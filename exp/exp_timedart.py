from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.TimeDART_paramsandtime_re_20260330103340 import FixedMultiScaleConv,FixedMultiScaleConv, LightWeightGenerator
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    transfer_weights,
    show_series,
    show_matrix, visual,
)
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from utils.augmentations import masked_data
from utils.metrics import metric
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm
from collections import defaultdict
import pickle

warnings.filterwarnings("ignore")
def _strip_module_prefix(state_dict):
    """去掉DDP/DP保存产生的 'module.' 前缀"""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    return new_sd

def _extract_model_state_dict(ckpt_obj):
    """
    兼容多种保存格式：
    - 直接是 state_dict（dict[str, Tensor]）
    - {'state_dict': ...}
    - {'model_state_dict': ...}
    """
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
    # 退化为“就当它本身是state_dict”
    return ckpt_obj

def _shape_adapt_if_scalar(param_tensor, ref_tensor):
    """
    当两边都是“标量语义”时，做 [] <-> [1] 的自适应：
    - 旧ckpt: []  新模型: [1]  -> unsqueeze(0)
    - 旧ckpt: [1] 新模型: []   -> squeeze(0)
    其他情况保持不动。
    """
    t = param_tensor
    want = ref_tensor
    try:
        if t.numel() == 1 and want.numel() == 1:
            # 只在两个张量都是单元素时才考虑自适应
            if t.dim() == 0 and want.dim() == 1 and want.shape == (1,):
                return t.unsqueeze(0)
            if t.dim() == 1 and t.shape == (1,) and want.dim() == 0:
                return t.squeeze(0)
    except Exception:
        pass
    return t

def load_checkpoint_compat(model, checkpoint_path, map_location="cpu", verbose=True):
    """
    兼容加载：
    1) 读取并提取state_dict；
    2) 去掉module.前缀；
    3) 针对“标量 <-> [1]”做形状自适应；
    4) 仅加载模型中存在的键；
    5) strict=False 加载并返回 missing/unexpected。
    """
    ckpt_obj = torch.load(checkpoint_path, map_location=map_location)
    raw_sd = _extract_model_state_dict(ckpt_obj)
    raw_sd = _strip_module_prefix(raw_sd)

    model_sd = model.state_dict()
    merged = OrderedDict()

    for k, v in raw_sd.items():
        if k in model_sd:
            vv = _shape_adapt_if_scalar(v, model_sd[k])
            # 若形状仍不匹配，跳过；否则记录
            if vv.shape == model_sd[k].shape:
                merged[k] = vv
            else:
                if verbose:
                    print(f"[skip:shape] {k}: ckpt{tuple(v.shape)} != model{tuple(model_sd[k].shape)}")
        else:
            # 可以在这里加更智能的匹配（比如尾部匹配/别名映射），默认忽略
            if verbose:
                pass  # print(f"[skip:key] {k} 不在当前模型中")

    # 用 merged 覆盖 model_sd 相同键
    model_sd.update(merged)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if verbose:
        print("[load_checkpoint_compat] loaded:", len(merged), "params")
        if missing:
            print("  missing keys:", missing)
        if unexpected:
            print("  unexpected keys:", unexpected)
    return missing, unexpected

class AdaptiveLossBalancer:
    def __init__(self, base_weights, momentum=0.9):
        """
        base_weights: dict 基础权重配置
        momentum: 滑动平均的动量因子
        """
        self.base_weights = base_weights
        self.momentum = momentum
        self.running_means = {k: None for k in base_weights.keys()}

    def __call__(self, losses,device):
        """
        losses: dict {loss_name: tensor}
        返回加权后的总损失
        """
        factor_hist = defaultdict(list)

        total_loss = torch.zeros((), device=device)
        adaptive_factors = {}
        # 计算自适应因子
        for name, loss in losses.items():
            if self.running_means[name] is None:
                self.running_means[name] = loss.detach()
            else:
                self.running_means[name] = (
                        self.momentum * self.running_means[name] +
                        (1 - self.momentum) * loss.detach()
                )

            # 相对大小因子 (0.5-2.0范围)
            rel_factor = loss.detach() / (self.running_means[name] + 1e-8)
            adaptive_factor = torch.clamp(rel_factor, 0.5, 2.0)
            adaptive_factors[name] = adaptive_factor

            # 应用基础权重和自适应因子
            # 3) 加权累加（全程 tensor 运算）
            weight = self.base_weights[name]

            # 如果 base_weights 是 Python float，loss*adaptive_factor*weight 会自动转成 tensor
            loss_weight_value = weight * adaptive_factor * loss
            total_loss = total_loss + loss_weight_value
            factor_hist[name].append(loss_weight_value.item())
        return total_loss, adaptive_factors,factor_hist



import torch

def count_fixed_multiscale_conv(m, x, y):
    x = x[0]
    B, C, L = x.shape

    total_ops = 0
    for conv in m.conv_layers:
        k = conv.kernel_size[0]
        total_ops += B * C * L * k

    # 一定要跟 m.total_ops 保持同 device / dtype
    if not hasattr(m, "total_ops"):
        return

    m.total_ops += m.total_ops.new_tensor([total_ops])


def count_lightweight_generator(m, x, y):
    x = x[0]
    B, C, L = x.shape
    K = m.num_scales

    # AdaptiveAvgPool1d(1): 近似 B*C*L
    pool_ops = B * C * L

    # Linear(C -> C*K): B * C * (C*K)
    linear_ops = B * C * (C * K)

    total_ops = pool_ops + linear_ops

    # 一定要跟 m.total_ops 保持同 device / dtype
    if not hasattr(m, "total_ops"):
        return

    m.total_ops += m.total_ops.new_tensor([total_ops])
class Exp_TimeDART(Exp_Basic):
    def __init__(self, args):
        super(Exp_TimeDART, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")
        self.loss_balancer = AdaptiveLossBalancer(
            # base_weights={
            #     'diff': 1.0,
            #     'freq': self.model.freq_weight,
            #     'orth': self.model.orth_weight,
            #     'smooth': self.model.smooth_weight,
            #     'season_freq': self.model.season_freq_weight,
            #     'recon': self.model.recon_weight,
            #     'period': self.model.period_weight,
            #     'reg': self.model.reg_weight
            # },
            # base_weights={
            #     'diff': 1.0,
            #     'freq': 0.3,
            #     'orth': 0.01,
            #     'smooth': 0,
            #     'season_freq': 0.05,
            #     'recon': 0,
            # },
            base_weights={
                'diff': 1.0,
                'freq': self.args.log_var_freq,
                'orth':self.args.log_var_orth,
                'smooth': self.args.log_var_smooth,
                'season_freq': self.args.log_var_season_freq,
                'recon': self.args.log_var_recon,
            },
            # base_weights={
            #     'diff': 0.9,  # 保持主导，但稍下调
            #     'freq': 0.6,  # 明显提高，强化全频约束
            #     'orth': 0.03,  # 提高，使正交性生效
            #     'smooth': 0.0,  # 若需平滑，可置小值；否则继续置0
            #     'season_freq': 0.12,  # 增强季节成分学习
            #     'recon': 0.0,  # 删除
            #     'period': 0.03,  # 若保留，可小幅度调整；否则删除
            #     'reg': 1e-4  # 启用模型参数正则
            # },
            momentum=0.95
        )
        self.loss_names = ['diff', 'freq', 'orth', 'smooth', 'season_freq', 'recon']
        self.project_path= os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    def _clear_thop_buffers_and_hooks(self, model):
        for module in model.modules():
            # 清掉 thop 常挂的统计 buffer
            if hasattr(module, "total_ops"):
                try:
                    delattr(module, "total_ops")
                except Exception:
                    pass
            if hasattr(module, "total_params"):
                try:
                    delattr(module, "total_params")
                except Exception:
                    pass

            # 清掉 forward hooks（thop失败时可能残留）
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks.clear()
            if hasattr(module, "_forward_pre_hooks"):
                module._forward_pre_hooks.clear()
            if hasattr(module, "_backward_hooks"):
                module._backward_hooks.clear()
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))

            transfer_device = self.args.device if torch.cuda.is_available() else "cpu"
            model = transfer_weights(
                self.args.load_checkpoints, model, device=transfer_device
            )

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print(
            "number of model params",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def pretrain(self):

        # data preparation
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        path = path + '_dln_' + str(self.args.denoise_layers_num)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        # model_optim.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        # model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,T_max=self.args.train_epochs)
        model_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=model_optim, gamma=self.args.lr_decay
        )

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            # current learning rate
            print("Current learning rate: {:.7f}".format(model_scheduler.get_last_lr()[0]))

            train_loss = self.pretrain_one_epoch(
                train_loader, model_optim, model_scheduler
            )
            vali_loss = self.valid_one_epoch(vali_loader)

            # log and Loss
            end_time = time.time()
            print(
                "Epoch: {}/{}, Time: {:.2f}, Train Loss: {:.4f}, Vali Loss: {:.4f}".format(
                    epoch + 1,
                    self.args.train_epochs,
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                )
            )

            loss_scalar_dict = {
                "train_loss": train_loss,
                "vali_loss": vali_loss,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model epoch{}...".format(
                        min_vali_loss, vali_loss, epoch
                    )
                )
                min_vali_loss = vali_loss

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if "encoder" in k or "enc_embedding" in k or "decomp_multi_learnable" in k or 'log_var_freq' in k or 'log_var_orth' in k or 'log_var_smooth' in k or 'log_var_season_freq' in k:
                        if "module." in k:
                            k = k.replace("module.", "")  # multi-gpu
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.encoder_state_dict,
                }
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if "encoder" in k or "enc_embedding" in k or "decomp_multi_learnable" in k or 'log_var_freq' in k or 'log_var_orth' in k or 'log_var_smooth' in k or 'log_var_season_freq' in k:
                        if "module." in k:
                            k = k.replace("module.", "")
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.encoder_state_dict,
                }
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

    def get_layer_weight(self,layer_idx, total_layers):
        return max(0.5, 1.0 - 0.2 * (layer_idx / total_layers))
    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):
        train_loss = []
        model_criterion = self._select_criterion()
        total_epoch = len(train_loader)

        self.model.train()
        if self.args.use_init_loss == 1:

            simple_batch_x, simple_batch_y, simple_batch_x_mark, simple_batch_y_mark = next(iter(train_loader))  # 获取一个批次
            simple_batch_x = simple_batch_x.float().to(self.device)

            self.model.init_adaptive_weights(simple_batch_x)




        # -------------------



        # loss_names = ['diff', 'freq', 'orth', 'smooth', 'season_freq', 'recon', 'period', 'reg','ps_loss']
        # loss_names = ['diff','ps_loss']
        resourcce_loss_dict_hist = defaultdict(list)
        factor_item_hist = defaultdict(list)

        # -------------------

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            train_loader
        ):
            model_optim.zero_grad()

            mask_rate = 0.5
            lm = 3
            positive_nums = 1
            # batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, mask_rate, lm,
            #                                               positive_nums)

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)


            # batch_x_m = batch_x_m.float().to(self.device)

            if self.args.model == 'TimeDART_my':
            # pred_x = self.model(batch_x)
                diff_loss = self.model(batch_x)
                # diff_loss.requires_grad = True
            # diff_loss = model_criterion(pred_x, batch_x)
            elif self.args.model == 'TimeDART' or self.args.model == 'TimeDART_paramsandtime_re_20260330103340':
                # pred_x,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss = self.model(batch_x,batch_y,i)
                # pred_x,freq_loss,orth_loss,smoothness,season_freq_loss = self.model(batch_x,batch_y,i)
                diff_loss,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss = self.model(batch_x,batch_y,i)

                # diff_loss = model_criterion(pred_x, batch_x)



                # 自适应权重计算（不确定权重法）
                # loss_freq = 1 / (2 * torch.exp(self.model.log_var_freq)) * freq_loss# + 0.5 * self.model.log_var_freq
                # loss_orth = 1 / (2 * torch.exp(self.model.log_var_orth)) * orth_loss #+ 0.5 * self.model.log_var_orth
                # loss_smooth = 1 / (
                #             2 * torch.exp(self.model.log_var_smooth)) * smoothness #+ 0.5 * self.model.log_var_smooth
                # loss_season_freq = 1 / (2 * torch.exp(
                #     self.model.log_var_season_freq)) * season_freq_loss #+ 0.5 * self.model.log_var_season_freq
                # loss_recon = 1 / (2 * torch.exp(self.model.log_var_recon)) * recon_loss #+ 0.5 * self.model.log_var_recon



                # loss_freq = 1 / (2 * self.model.log_var_freq) * freq_loss + 0.5 * torch.log(self.model.log_var_freq)
                #
                # loss_orth = 1 / (2 * self.model.log_var_orth) * orth_loss + 0.5 * torch.log(self.model.log_var_orth)
                #
                # loss_smooth = 1 / (
                #             2 * self.model.log_var_smooth) * smoothness + 0.5 * torch.log(self.model.log_var_smooth)
                # loss_season_freq = 1 / (2 *
                #     self.model.log_var_season_freq) * season_freq_loss + 0.5 * torch.log(self.model.log_var_season_freq)
                # loss_recon = 1 / (2 * self.model.log_var_recon) * recon_loss + 0.5 * torch.log(self.model.log_var_recon)


                loss_freq = 1 / (2 * self.model.log_var_freq) * freq_loss + 0.5*torch.log(self.model.log_var_freq)
                loss_orth = 1 / (2 * self.model.log_var_orth) * orth_loss + 0.5*torch.log(self.model.log_var_orth)
                loss_smooth = 1 / (2 * self.model.log_var_smooth) * smoothness + 0.5*torch.log(self.model.log_var_smooth)
                loss_season_freq = 1 / (2 * self.model.log_var_season_freq) * season_freq_loss + 0.5*torch.log(self.model.log_var_season_freq)
                loss_recon = 1 / (2 * self.model.log_var_recon) * recon_loss + 0.5*torch.log(self.model.log_var_recon)



                if self.args.del_orth_loss == 1:
                    loss_orth = torch.tensor(0.0)  # 618
                if self.args.del_smoothness_loss == 1:
                    loss_smooth = torch.tensor(0.0)  # 0.0025
                if self.args.del_season_freq_loss == 1:
                    loss_season_freq = torch.tensor(0.0)  # 10.6285
                if self.args.del_freq_loss == 1:
                    loss_freq = torch.tensor(0.0)  # 2.3
                if self.args.del_recon_loss == 1:
                    loss_recon = torch.tensor(0.0)
                # loss_freq = 0.01 * freq_loss
                # loss_orth = 0.1 * orth_loss
                # loss_smooth = 0.1 * smoothness

                # resourcce_loss_dict = {  # 原始输出loss
                #     'diff': diff_loss,
                #     'freq': freq_loss,
                #     'orth': orth_loss,
                #     'smooth': smoothness,
                #     'season_freq': season_freq_loss,
                #     'recon': recon_loss,
                # },
                log_loss_dict = {  # 原始输出loss
                    'diff': diff_loss,
                    'freq': loss_freq,
                    'orth': loss_orth,
                    'smooth': loss_smooth,
                    'season_freq': loss_season_freq,
                    'recon': loss_recon,
                },

                model_loss = diff_loss
                total_loss, adaptive_factors, factor_hist_item = self.loss_balancer(log_loss_dict[0],
                                                                               device=self.device)
                for name in self.loss_names:
                    resourcce_loss_dict_hist[name].append(log_loss_dict[0][name].item())
                    # adaptive_factors[name] 也是对应的权重
                    factor_item_hist[name].append(factor_hist_item[name])
                if self.args.use_loss_compute != 1:
                    total_loss = model_loss
                # diff_loss = total_loss

                # -----
                # 训练监控（前5个epoch打印详细信息）


            else:
                pred_x, = self.model(batch_x,None,i)
                # diff_loss = self.model(batch_x)
                diff_loss = model_criterion(pred_x, batch_x)
            # diff_loss.requires_grad = True
            diff_loss.backward()
            model_optim.step()
            train_loss.append(diff_loss.item())

        # ###############绘图

        # 解决中文显示问题
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 下采样函数：每隔step个点取一个点
        def downsample(data, step=10):
            return data[::step]

        # 保存历史数据的函数
        def save_hist(hist_dict, filename):
            with open(filename, 'wb') as f:
                pickle.dump(hist_dict, f)

        # ---- 1. 各loss分量随step变化 ----
        # plt.figure(figsize=(20, 6))
        # for name in self.loss_names:
        #     downsampled = downsample(resourcce_loss_dict_hist[name])
        #     plt.plot(downsampled, label=name)
        # plt.title('各 Loss 分量随 Step 的变化（源loss）')
        # plt.xlabel('Training Step')
        # plt.ylabel('Loss Value')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(str(self.args.data) + str(self.args.task_name) + str(self.args.pred_len)+'源loss.png')
        # save_hist(resourcce_loss_dict_hist, str(self.args.data) + str(self.args.task_name) + str(self.args.pred_len)+'resourcce_loss_dict_hist.pkl')  # 保存原始数据
        #
        # # ---- 2. 各adaptive factor随step变化 ----
        # plt.figure(figsize=(20, 6))
        # for name in self.loss_names:
        #     downsampled = downsample(factor_item_hist[name])
        #     plt.plot(downsampled, label=name)
        # plt.title('各 Adaptive Factor 随 Step 的变化（内部乘上权重之后的loss分量）')
        # plt.xlabel('Training Step')
        # plt.ylabel('Adaptive Factor')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(str(self.args.data) + str(self.args.task_name) + str(self.args.pred_len)+'内部乘上权重之后的loss分量.png')
        # save_hist(factor_item_hist, str(self.args.data) + str(self.args.task_name) + str(self.args.pred_len)+'factor_item_hist.pkl')  # 保存原始数据

        model_scheduler.step()
        train_loss = np.mean(train_loss)

        return train_loss

    def valid_one_epoch(self, vali_loader):
        vali_loss = []
        model_criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):

                mask_rate = 0.5
                lm = 3
                positive_nums = 1
                batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, mask_rate, lm,
                                                              positive_nums)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                batch_x_m = batch_x_m.float().to(self.device)
                if self.args.model == 'TimeDART_my':
                    # pred_x = self.model(batch_x)
                    diff_loss = self.model(batch_x)
                # diff_loss = model_criterion(pred_x, batch_x)
                elif self.args.model == 'TimeDART' or self.args.model == 'TimeDART_paramsandtime_re_20260330103340':
                    # pred_x, freq_loss, orth_loss, smoothness,season_freq_loss,recon_loss = self.model(batch_x, batch_x_m, i)
                    # diff_loss = self.model(batch_x)
                    # diff_loss = model_criterion(pred_x, batch_x)
                    diff_loss, freq_loss, orth_loss, smoothness,season_freq_loss,recon_loss = self.model(batch_x, batch_x_m, i)

                else:
                    pred_x = self.model(batch_x,batch_x_m)
                    # diff_loss = self.model(batch_x)
                    diff_loss = model_criterion(pred_x, batch_x)

                # diff_loss.backward()
                vali_loss.append(diff_loss.item())

        vali_loss = np.mean(vali_loss)

        return vali_loss
    def _sync_device(self):
        dev_str = str(self.device).lower()
        if "cuda" in dev_str and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif "musa" in dev_str and hasattr(torch, "musa"):
            try:
                torch.musa.synchronize()
            except Exception:
                pass

    def count_params_macs_once(
        self,
        model,
        flag="test",
        warmup_steps=10,
        measure_steps=50,
    ):
        from thop import profile, clever_format

        data_set, data_loader = self._get_data(flag=flag)
        model.eval()

        first_batch = next(iter(data_loader))
        batch_x, batch_y, batch_x_mark, batch_y_mark = first_batch
        batch_x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)

        # ===== Params =====
        if hasattr(model, "count_active_parameters"):
            params_raw = model.count_active_parameters(trainable_only=False)
            trainable_params_raw = model.count_active_parameters(trainable_only=True)
        else:
            params_raw = sum(p.numel() for p in model.parameters())
            trainable_params_raw = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # params_fmt = clever_format([params_raw], "%.3f")[0]
        params_fmt = clever_format([params_raw], "%.3f")
        if isinstance(params_fmt, (list, tuple)):
            params_fmt = params_fmt[0]

        # ===== 先 dummy forward，确保 lazy module 初始化 =====
        with torch.no_grad():
            _ = model(batch_x, batch_x_mark, 0)
        self._sync_device()

        # ===== MACs =====
        macs_raw = -1
        macs_fmt = "N/A"

        # 先清一次，避免上次失败残留
        self._clear_thop_buffers_and_hooks(model)

        try:
            with torch.no_grad():
                macs_raw, _ = profile(
                    model,
                    inputs=(batch_x, batch_x_mark, 0),
                    custom_ops={
                        FixedMultiScaleConv: count_fixed_multiscale_conv,
                        LightWeightGenerator: count_lightweight_generator,
                    },
                    verbose=False
                )
            macs_fmt = clever_format([macs_raw], "%.3f")
            if isinstance(macs_fmt, (list, tuple)):
                macs_fmt = macs_fmt[0]

        except Exception as e:
            print(f"[WARN] thop profile failed: {e}")

        finally:
            # 不管成功失败，都清残留
            self._clear_thop_buffers_and_hooks(model)

        # ===== Time =====
        total_time = 0.0
        total_samples = 0
        total_steps = 0

        with torch.no_grad():
            for step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                if step < warmup_steps:
                    _ = model(batch_x, batch_x_mark, step)
                    self._sync_device()
                    continue

                if total_steps >= measure_steps:
                    break

                self._sync_device()
                start_t = time.time()
                _ = model(batch_x, batch_x_mark, step)
                self._sync_device()
                end_t = time.time()

                total_time += (end_t - start_t)
                total_samples += batch_x.size(0)
                total_steps += 1

        ms_per_sample = (total_time / max(total_samples, 1)) * 1000.0
        samples_per_sec = total_samples / max(total_time, 1e-12)

        result = {
            "ablation_mode": getattr(model, "ablation_mode", "full"),
            "params_raw": params_raw,
            "trainable_params_raw": trainable_params_raw,
            "macs_raw": macs_raw,
            "params_fmt": params_fmt,
            "macs_fmt": macs_fmt,
            "ms_per_sample": ms_per_sample,
            "samples_per_sec": samples_per_sec,
            "measured_samples": total_samples,
            "measured_steps": total_steps,
        }
        return result

    def profile_efficiency_variants(self, flag="test", warmup_steps=10, measure_steps=50):
        """
        分别统计:
            - full
            - no_aasd
            - no_tcsr
        """
        assert hasattr(self.model, "set_ablation_mode"), \
            "Model 里没有 set_ablation_mode，请先按我给的 model 代码修改。"

        variants = [
            ("full", "Full"),
            ("no_aasd", "w/o AASD"),
            ("no_tcsr", "w/o TCSR"),
        ]

        old_mode = getattr(self.model, "ablation_mode", "full")
        all_results = []

        print("\n================ Efficiency Profiling ================\n")
        for mode, show_name in variants:
            self.model.set_ablation_mode(mode)
            result = self.count_params_macs_once(
                model=self.model,
                flag=flag,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
            )
            all_results.append((show_name, result))

            print(
                "[{}] Params: {} ({:,}) | Trainable Params: {:,} | MACs: {} ({}) | {:.3f} ms/sample | {:.1f} samples/s".format(
                    show_name,
                    result["params_fmt"],
                    result["params_raw"],
                    result["trainable_params_raw"],
                    result["macs_fmt"],
                    result["macs_raw"],
                    result["ms_per_sample"],
                    result["samples_per_sec"],
                )
            )

        self.model.set_ablation_mode(old_mode)

        print("\n=====================================================\n")
        return all_results

    def format_efficiency_results(self, results):
        lines = []
        lines.append("Efficiency profiling results:")
        for show_name, r in results:
            line = (
                f"{show_name}: "
                f"Params={r['params_fmt']} "
                f"(trainable={r['trainable_params_raw']:,}), "
                f"MACs={r['macs_fmt']}, "
                f"Time={r['ms_per_sample']:.3f} ms/sample, "
                f"Throughput={r['samples_per_sec']:.1f} samples/s"
            )
            lines.append(line)
        return "\n".join(lines)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        path = path + '_dln_' + str(self.args.denoise_layers_num)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_criteria = self._select_criterion()
        model_scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )
        resourcce_loss_dict_hist = defaultdict(list)
        factor_item_hist = defaultdict(list)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loader = tqdm(train_loader, desc="Training")

            print("Current learning rate: {:.7f}".format(model_scheduler.get_last_lr()[0]))

            self.model.train()
            if self.args.use_init_loss == 1:
                simple_batch_x, simple_batch_y, simple_batch_x_mark, simple_batch_y_mark = next(
                    iter(train_loader))  # 获取一个批次
                simple_batch_x = simple_batch_x.float().to(self.device)

                self.model.init_adaptive_weights(simple_batch_x)

            start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                pred_x, freq_loss, orth_loss, smoothness,season_freq_loss ,recon_loss= self.model(batch_x,batch_x_mark)

                # 自适应权重计算（不确定权重法）
                # loss_freq = 1/(2*torch.exp(self.model.log_var_freq)) * freq_loss #+ 0.5*self.model.log_var_freq
                # loss_orth = 1/(2*torch.exp(self.model.log_var_orth)) * orth_loss #+ 0.5*self.model.log_var_orth
                # loss_smooth = 1/(2*torch.exp(self.model.log_var_smooth)) * smoothness #+ 0.5*self.model.log_var_smooth
                # loss_season_freq = 1/(2*torch.exp(self.model.log_var_season_freq)) * season_freq_loss #+ 0.5*self.model.log_var_season_freq
                # loss_recon = 1/(2*torch.exp(self.model.log_var_recon)) * recon_loss #+ 0.5*self.model.log_var_recon

                # loss_freq = 1 / (2 * self.model.log_var_freq) * freq_loss + 0.5*torch.log(self.model.log_var_freq)
                # loss_orth = 1 / (2 * self.model.log_var_orth) * orth_loss + 0.5*torch.log(self.model.log_var_orth)
                # loss_smooth = 1 / (2 * self.model.log_var_smooth) * smoothness + 0.5*torch.log(self.model.log_var_smooth)
                # loss_season_freq = 1 / (2 * self.model.log_var_season_freq) * season_freq_loss + 0.5*torch.log(self.model.log_var_season_freq)
                # loss_recon = 1 / (2 * self.model.log_var_recon) * recon_loss + 0.5*torch.log(self.model.log_var_recon)

                loss_freq = 1 / (2 * self.model.log_var_freq) * freq_loss + 0.5*torch.log(self.model.log_var_freq)
                loss_orth = 1 / (2 * self.model.log_var_orth) * orth_loss + 0.5*torch.log(self.model.log_var_orth)
                loss_smooth = 1 / (2 * self.model.log_var_smooth) * smoothness + 0.5*torch.log(self.model.log_var_smooth)
                loss_season_freq = 1 / (2 * self.model.log_var_season_freq) * season_freq_loss + 0.5*torch.log(self.model.log_var_season_freq)
                loss_recon = 1 / (2 * self.model.log_var_recon) * recon_loss + 0.5*torch.log(self.model.log_var_recon)


                # loss_freq = 0.01 * freq_loss
                # loss_orth = 0.1 * orth_loss
                # loss_smooth = 0.1 * smoothness
                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                loss = model_criteria(pred_x, batch_y)
                if self.args.del_orth_loss == 1:
                    loss_orth = torch.tensor(0.0)  # 618
                if self.args.del_smoothness_loss == 1:
                    loss_smooth = torch.tensor(0.0)  # 0.0025
                if self.args.del_season_freq_loss == 1:
                    loss_season_freq = torch.tensor(0.0)  # 10.6285
                if self.args.del_freq_loss == 1:
                    loss_freq = torch.tensor(0.0)  # 2.3
                if self.args.del_recon_loss == 1:
                    loss_recon = torch.tensor(0.0)
                # model_loss = loss
                # log_loss_dict = {  # 原始输出loss
                #     'diff': loss,
                #     'freq': loss_freq,
                #     'orth': loss_orth,
                #     'smooth': loss_smooth,
                #     'season_freq': loss_season_freq,
                #     'recon': loss_recon,
                # },
                #
                # total_loss, adaptive_factors, factor_hist_item = self.loss_balancer(log_loss_dict[0],
                #                                                                     device=self.device)
                # for name in self.loss_names:
                #     resourcce_loss_dict_hist[name].append(log_loss_dict[0][name].item())
                #     # adaptive_factors[name] 也是对应的权重
                #     factor_item_hist[name].append(factor_hist_item[name])
                # if self.args.use_loss_compute != 1:
                #     total_loss = model_loss
                # loss = total_loss
                loss = loss + loss_freq + loss_orth + loss_smooth + loss_season_freq +loss_recon




                # -----
                # 训练监控（前5个epoch打印详细信息）







                loss.backward()
                model_optim.step()
                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)
            vali_loss = self.valid(vali_loader, model_criteria)
            test_loss = self.valid(test_loader, model_criteria)

            end_time = time.time()
            print(
                "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                    epoch + 1,
                    len(train_loader),
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                    test_loss,
                )
            )
            log_path = path + "/" + "log.txt"
            with open(log_path, "a") as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}\n".format(
                        epoch + 1,
                        len(train_loader),
                        end_time - start_time,
                        train_loss,
                        vali_loss,
                        test_loss,
                    )
                )
            early_stopping(vali_loss, self.model, path=path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "step":
                adjust_learning_rate(model_optim, model_scheduler, epoch + 1, self.args)
        # ###############绘图

        # 解决中文显示问题
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 下采样函数：每隔step个点取一个点
        def downsample(data, step=10):
            return data[::step]

        # 保存历史数据的函数
        def save_hist(hist_dict, filename):
            with open(filename, 'wb') as f:
                pickle.dump(hist_dict, f)

        # ---- 1. 各loss分量随step变化 ----
        # plt.figure(figsize=(20, 6))
        # for name in self.loss_names:
        #     downsampled = downsample(resourcce_loss_dict_hist[name])
        #     plt.plot(downsampled, label=name)
        # plt.title('各 Loss 分量随 Step 的变化（源loss）')
        # plt.xlabel('Training Step')
        # plt.ylabel('Loss Value')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(str(self.args.data) + str(self.args.task_name) + str(self.args.pred_len)+'train源loss.png')
        # save_hist(resourcce_loss_dict_hist, str(self.args.data) + str(self.args.task_name) + str(self.args.pred_len)+'trainresourcce_loss_dict_hist.pkl')  # 保存原始数据
        #
        # # ---- 2. 各adaptive factor随step变化 ----
        # plt.figure(figsize=(20, 6))
        # for name in self.loss_names:
        #     downsampled = downsample(factor_item_hist[name])
        #     plt.plot(downsampled, label=name)
        # plt.title('各 Adaptive Factor 随 Step 的变化（内部乘上权重之后的loss分量）')
        # plt.xlabel('Training Step')
        # plt.ylabel('Adaptive Factor')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(str(self.args.data) + str(self.args.task_name) + str(self.args.pred_len)+'train内部乘上权重之后的loss分量.png')
        # save_hist(factor_item_hist, str(self.args.data) + str(self.args.task_name) + str(self.args.pred_len)+'trainfactor_item_hist.pkl')  # 保存原始数据

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.args.device))

        self.lr = model_scheduler.get_last_lr()[0]

        return self.model

    def valid(self, vali_loader, model_criteria):
        vali_loss = []
        self.model.eval()
        vali_loader = tqdm(vali_loader, desc="Validation")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                pred_x, freq_loss, orth_loss, smoothness ,season_freq_loss,recon_loss= self.model(batch_x,batch_x_mark)
                # 自适应权重计算（不确定权重法）
                # loss_freq = 0.5 / (self.model.log_var_freq.exp()) * freq_loss + 0.5 * self.model.log_var_freq
                # loss_orth = 0.5 / (self.model.log_var_orth.exp()) * orth_loss + 0.5 * self.model.log_var_orth
                # loss_smooth = 0.5 / (self.model.log_var_smooth.exp()) * smoothness + 0.5 * self.model.log_var_smooth

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = pred_x.detach().cpu()
                true = batch_y.detach().cpu()

                loss = model_criteria(pred_x, batch_y)
                # loss = loss + + loss_freq + loss_orth + loss_smooth
                vali_loss.append(loss.item())

        vali_loss = np.mean(vali_loss)
        self.model.train()

        return vali_loss

    def test(self,settings=None,test=0):
        test_data, test_loader = self._get_data(flag="test")
        path = os.path.join(self.args.checkpoints, settings)
        path = path + '_dln_' + str(self.args.denoise_layers_num)
        output_dir = os.path.join(self.project_path, 'outputs/npz_results')
        npz_folder_path = os.path.join(output_dir, settings)  # 使用具体的字符串属性
        if not os.path.exists(npz_folder_path):
            os.makedirs(npz_folder_path)
        if test:
            print('loading model')
            ckpt_best = os.path.join(path, 'ckpt_best.pth')
            ckpt_default = os.path.join(path, 'checkpoint.pth')
            checkpoint_path = ckpt_best if os.path.exists(ckpt_best) else ckpt_default

            if os.path.exists(checkpoint_path):
                # 统一从cpu加载，避免显存/设备不一致问题；模型本身在外面再to(device)
                try:
                    load_checkpoint_compat(
                        model=self.model,
                        checkpoint_path=checkpoint_path,
                        map_location="cpu",
                        verbose=True
                    )
                except Exception as e:
                    print("[load_checkpoint_compat] 失败，降级为 strict=False 直接加载:", e)
                    # 兜底：直接尝试 strict=False（不做形状自适配）
                    ckpt_obj = torch.load(checkpoint_path, map_location="cpu")
                    raw_sd = _extract_model_state_dict(ckpt_obj)
                    raw_sd = _strip_module_prefix(raw_sd)
                    self.model.load_state_dict(raw_sd, strict=False)
            else:
                print(f"[warn] 找不到 checkpoint 文件: {checkpoint_path}")
            print("load success")

        preds = []
        trues = []

        folder_path = "./outputs/test_results/{}".format(self.args.data)
        folder_path = folder_path + '_dln_' + str(self.args.denoise_layers_num)

        # result save
        # npz_folder_path = os.path.join(self.project_path+ os.sep +'outputs' + os.sep, self.args)

        output_dir = os.path.join(self.project_path, 'outputs/npz_results')
        npz_folder_path = os.path.join(output_dir, settings)  # 使用具体的字符串属性


        if not os.path.exists(npz_folder_path):
            os.makedirs(npz_folder_path)


        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        infer_start = time.time()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)

                pred_x,freq_loss, orth_loss, smoothness,season_freq_loss,_ = self.model(batch_x,batch_x_mark)
                # 自适应权重计算（不确定权重法）
                # loss_freq = 0.5 / (self.model.log_var_freq.exp()) * freq_loss + 0.5 * self.model.log_var_freq
                # loss_orth = 0.5 / (self.model.log_var_orth.exp()) * orth_loss + 0.5 * self.model.log_var_orth
                # loss_smooth = 0.5 / (self.model.log_var_smooth.exp()) * smoothness + 0.5 * self.model.log_var_smooth
                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = pred_x.detach().cpu()
                true = batch_y.detach().cpu()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    fig_path = folder_path + os.sep + 'figs'
                    if not os.path.exists(fig_path):
                        os.makedirs(fig_path)
                    visual(gt, pd, os.path.join(fig_path, str(i) + '.png'))

        # preds = np.array(preds)
        # trues = np.array(trues)
        # 结束后统计整体推理耗时
        total_time = time.time() - infer_start
        n_samples_total = len(test_data)  # 或者循环里累加 batch_x.size(0)

        ms_per_sample = (total_time / n_samples_total) * 1000.0
        samples_per_sec = n_samples_total / total_time
        print(f"[Inference][Total] {ms_per_sample:.3f} ms/sample | {samples_per_sec:.1f} samples/s "
              f"(total_samples={n_samples_total})")
        preds = torch.stack(preds, dim=0).numpy()
        trues = torch.stack(trues, dim=0).numpy()
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(
            "{0}->{1}, mse:{2:.3f}, mae:{3:.3f}".format(
                self.args.input_len, self.args.pred_len, mse, mae
            )
        )
        f = open(folder_path + "/score.txt", "a")
        from datetime import datetime
        efficiency_results = self.profile_efficiency_variants(
            flag="test",
            warmup_steps=10,
            measure_steps=50
        )
        params_str = self.format_efficiency_results(efficiency_results)
        # f.write('params:{}'.format(params_str) + "  \n")

        # 获取当前时间
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

        f.write(
            "{0}->{1}, {2:.3f}, {3:.3f},{4},{5},{6},{7},log_var_orth={8},log_var_season_freq={9},log_var_freq={10},use_loss_compute={11},use_new_decomp={12},use_denoise={13},use_inner_new_decomp={14}\n{15}\n".format(
                self.args.input_len,
                self.args.pred_len,
                mse,
                mae,
                formatted_time,
                self.args.d_model,
                self.args.batch_size,
                self.args.n_heads,
                self.args.log_var_orth,
                self.args.log_var_season_freq,
                self.args.log_var_freq,
                self.args.use_loss_compute,
                self.args.use_new_decomp,
                self.args.use_denoise,
                self.args.use_inner_new_decomp,
                params_str
            )
        )


        f.close()
        # np.save(npz_folder_path+os.sep+ 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(npz_folder_path+os.sep+'pred.npy', preds)
        # np.save(npz_folder_path +os.sep+'true.npy', trues)
        return
