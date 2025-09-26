from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
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

@torch.no_grad()
def calibrate_aux_weights_for_pretrain(model, loader, K=32, device='cuda'):
    """
    针对“预训练阶段要加的分解正则”的量级校准（例如在 x_hat 上的冻结分解）
    """
    model.eval()
    sums = {"freq":0.0,"orth":0.0,"smooth":0.0,"season_freq":0.0,"recon":0.0}
    n = 0
    # 冻结分解器
    decomp_params = []
    for name, p in model.named_parameters():
        if 'decomp_multi' in name or 'decomp_multi_learnable' in name:
            decomp_params.append(p); p.requires_grad_(False)

    for i, (bx, *_) in enumerate(loader):
        if i >= K: break
        bx = bx.float().to(device)
        # 走一次预训练前向拿到重建（示意：B 版）
        loss_or_eps = model.pretrain_B(bx, None)  # 这里你也可以返回 x_hat 方便统计
        # 假设你在 pretrain_B 里临时把 x_hat 存在 model.last_x_hat（或直接改返回）
        x_hat = model.last_x_hat  # [B,L,C]
        # 在冻结分解器上统计
        s_dec, t_dec, Lf, Lo, Ls, Lsf, Lrec = model.decomp_multi_learnable(x_hat)  # 只作为统计
        sums["freq"]+=Lf.item(); sums["orth"]+=Lo.item(); sums["smooth"]+=Ls.item()
        sums["season_freq"]+=Lsf.item(); sums["recon"]+=Lrec.item(); n+=1

    for p in decomp_params: p.requires_grad_(True)
    means = {k: max(v/max(1,n),1e-8) for k,v in sums.items()}
    # 返回 log-variance 初值
    import math
    lg = lambda x: float(min(max(math.log(x), -6.0), 6.0))
    return { "log_var_freq":lg(means["freq"]), "log_var_orth":lg(means["orth"]),
             "log_var_smooth":lg(means["smooth"]), "log_var_season_freq":lg(means["season_freq"]),
             "log_var_recon":lg(means["recon"]) }

@torch.no_grad()
def calibrate_aux_weights_for_finetune(model, loader, K=64, device='cuda', use_kendall=True,
                          clamp=(-6.0, 6.0)):
    """
    在 K 个 mini-batches 上估计分解相关损失的典型量级，
    用于初始化 log_var_*（Kendall）或返回标量权重。
    要求：模型里有 decomp 模块；这一步不反传。
    """
    model.eval()
    sums = {"freq": 0.0, "orth": 0.0, "smooth": 0.0, "season_freq": 0.0, "recon": 0.0}
    n = 0

    # 可选：先冻结分解，保证统计稳定
    decomp_params = []
    for name, p in model.named_parameters():
        if 'decomp_multi' in name or 'decomp_multi_learnable' in name:
            decomp_params.append(p)
            p.requires_grad_(False)

    for i, (bx, *_) in enumerate(loader):
        if i >= K: break
        bx = bx.float().to(device)
        # 只做分解与正则项的前向统计；不要走你预训练的加噪/遮盖路径
        # 用干净 x 得到 seasonal/trend 与对应正则（与你现有 forward 的返回对齐即可）
        # 下面示例按你原 forward 的签名返回：
        _yhat, Lf, Lo, Ls, Lsf, Lrec = model.forecast(bx,None)  # 或显式调用分解函数统计
        sums["freq"] += float(Lf.item())
        sums["orth"] += float(Lo.item())
        sums["smooth"] += float(Ls.item())
        sums["season_freq"] += float(Lsf.item())
        sums["recon"] += float(Lrec.item())
        n += 1

    # 还原分解参数的 requires_grad
    for p in decomp_params:
        p.requires_grad_(True)

    means = {k: max(v / max(1, n), 1e-8) for k, v in sums.items()}

    if use_kendall:
        # Kendall: s_i = log( L_i ) 作为初始；并 clamp 防爆
        import math
        def lg(x):
            return float(min(max(math.log(x), clamp[0]), clamp[1]))

        return {
            "log_var_freq": lg(means["freq"]),
            "log_var_orth": lg(means["orth"]),
            "log_var_smooth": lg(means["smooth"]),
            "log_var_season_freq": lg(means["season_freq"]),
            "log_var_recon": lg(means["recon"]),
        }
    else:
        # 返回固定系数（如使各项乘权后量级相近）
        total = sum(means.values())
        return {k: (total / (5 * v)) for k, v in means.items()}  # 简单反比归一
@torch.no_grad()
def calibrate_aux_weights_for_finetune_coupled(model, loader, K=64, device='cuda', clamp=(-6,6)):
    model.eval()
    sums = dict(freq=0., orth=0., smooth=0., season_freq=0., recon=0.); n = 0
    for i, (bx, *_) in enumerate(loader):
        if i >= K: break
        bx = bx.float().to(device)
        # 与 forecast 完全一致的归一化
        means = bx.mean(1, keepdim=True); stds = bx.var(1, keepdim=True, unbiased=False).sqrt().clamp_min(1e-6)
        xnorm = (bx - means) / stds
        # 只跑分解器拿正则项（保留你 use_new_decomp 分支）
        if model.configs.use_new_decomp == 1:
            _, _, Lf, Lo, Ls, Lsf, Lrec = model.decomp_multi_learnable(xnorm)
        else:
            _, _ = model.decomp_multi(xnorm); Lf=Lo=Ls=Lsf=Lrec=bx.new_tensor(0.0)
        sums["freq"] += float(Lf); sums["orth"] += float(Lo); sums["smooth"] += float(Ls)
        sums["season_freq"] += float(Lsf); sums["recon"] += float(Lrec); n += 1

    means = {k: max(v/max(1,n), 1e-8) for k,v in sums.items()}
    import math; lg = lambda x: float(min(max(math.log(x), clamp[0]), clamp[1]))
    return { "log_var_freq":lg(means["freq"]), "log_var_orth":lg(means["orth"]),
             "log_var_smooth":lg(means["smooth"]), "log_var_season_freq":lg(means["season_freq"]),
             "log_var_recon":lg(means["recon"]) }


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
class Exp_TimeDART(Exp_Basic):
    def __init__(self, args):
        super(Exp_TimeDART, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")
        self.loss_balancer = AdaptiveLossBalancer(

            base_weights={
                'diff': 1.0,
                'freq': self.args.log_var_freq,
                'orth':self.args.log_var_orth,
                'smooth': self.args.log_var_smooth,
                'season_freq': self.args.log_var_season_freq,
                'recon': self.args.log_var_recon,
            },

            momentum=0.95
        )
        self.loss_names = ['diff', 'freq', 'orth', 'smooth', 'season_freq', 'recon']

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))

            transfer_device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

    def _save_pretrain_backbone(self, epoch, path, is_best=False):
        """
        只保存“骨干表示层”的权重：embedding/encoder/FiLM/denoise层/(可选)分解器/(可选)t_embed
        不保存：projection/regression/diffusion/log_var_* 等预训练专用或微调需重置的参数
        """
        ALLOW_PREFIX = [
            'enc_embedding', 'enc_embedding_trend',
            'positional_encoding',
            'encoder',
            'cond_to_gamma', 'cond_to_beta',
            'denoise_layers_cond',
            't_embed',  # 若 A 版才有；不存在不会出错
        ]
        if getattr(self.args, 'save_decomp_from_pretrain', 0) == 1:
            ALLOW_PREFIX.append('decomp_multi_learnable')

        to_save = {}
        for k, v in self.model.state_dict().items():
            k2 = k.replace('module.', '')
            if any(k2.startswith(p) for p in ALLOW_PREFIX):
                to_save[k2] = v.cpu()

        ckpt = {'epoch': epoch, 'model_state_dict': to_save}
        fname = 'backbone_best.pth' if is_best else f'backbone_epoch{epoch + 1}.pth'
        torch.save(ckpt, os.path.join(path, fname))

    @torch.no_grad()
    def valid_one_epoch(self, vali_loader):
        """
        预训练验证：直接跑一次“预训练前向”，拿到预训练目标的 loss（A=噪声预测；B=遮盖重建）。
        不做 masked_data 之类的数据增强，这些都在 model 内部完成。
        """
        self.model.eval()
        losses = []
        for i, (batch_x, _, batch_x_mark, _) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            # 预训练 forward 返回标量 loss
            diff_loss = self.model(batch_x, None, i)  # (x, x_mask=None, i)
            losses.append(diff_loss.item())
        return float(np.mean(losses)) if len(losses) > 0 else 0.0

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):
        """
        单 epoch 预训练循环：与微调循环解耦，简化成“取 batch → 计算预训练 loss → 反传”。
        """
        self.model.train()
        losses = []

        # 可选：仅在第一个 epoch 做一次 Kendall 初始化（针对“预训练阶段要加的分解正则”的量级）
        if getattr(self.args, 'use_init_loss_pretrain', 0) == 1:
            init = calibrate_aux_weights_for_pretrain(self.model, train_loader, K=64, device=self.device)
            self.model.log_var_freq.data = torch.tensor(init["log_var_freq"], device=self.device)
            self.model.log_var_orth.data = torch.tensor(init["log_var_orth"], device=self.device)
            self.model.log_var_smooth.data = torch.tensor(init["log_var_smooth"], device=self.device)
            self.model.log_var_season_freq.data = torch.tensor(init["log_var_season_freq"], device=self.device)
            self.model.log_var_recon.data = torch.tensor(init["log_var_recon"], device=self.device)

        for i, (batch_x, _, batch_x_mark, _) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device)

            # 预训练前向（A 或 B），直接返回预训练目标的 loss
            diff_loss = self.model(batch_x, None, i)  # (x, x_mask=None, i)
            diff_loss.backward()
            model_optim.step()

            losses.append(diff_loss.item())

        # epoch 末尾调度
        model_scheduler.step()
        return float(np.mean(losses)) if len(losses) > 0 else 0.0

    def pretrain(self):
        """
        总控：跑多 epoch，保存最优/按周期保存骨干权重。
        """
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        # 保存路径
        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        path = path + '_dln_' + str(self.args.denoise_layers_num)
        os.makedirs(path, exist_ok=True)

        # 优化器 & 学习率
        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=model_optim, gamma=self.args.lr_decay
        )

        min_vali = None
        for epoch in range(self.args.train_epochs):
            st = time.time()
            print("Current learning rate: {:.7f}".format(model_scheduler.get_last_lr()[0]))

            train_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            vali_loss = self.valid_one_epoch(vali_loader)

            print(f"Epoch {epoch + 1}/{self.args.train_epochs} | "
                  f"Time {time.time() - st:.2f}s | Train {train_loss:.4f} | Val {vali_loss:.4f}")

            self.writer.add_scalars("/pretrain_loss",
                                    {"train_loss": train_loss, "vali_loss": vali_loss}, epoch)

            # 保存最优骨干
            if (min_vali is None) or (vali_loss <= min_vali - 1e-9):
                print(f"Validation loss improved: {min_vali} -> {vali_loss}. Save best backbone.")
                min_vali = vali_loss
                self._save_pretrain_backbone(epoch, path, is_best=True)

            # 每 N 个 epoch 另存一份（可选）
            if (epoch + 1) % max(1, int(getattr(self.args, 'save_every', 10))) == 0:
                print(f"Saving backbone at epoch {epoch + 1} ...")
                self._save_pretrain_backbone(epoch, path, is_best=False)

    def _load_pretrained_backbone(self, ckpt_path):
        print(f"[finetune] load backbone from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        pre_sd = ckpt['model_state_dict']
        cur_sd = self.model.state_dict()

        # 过滤：两边都存在且 shape 匹配的键
        filtered = {k: v for k, v in pre_sd.items() if k in cur_sd and cur_sd[k].shape == v.shape}
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        print("[finetune] missing:", missing)  # 正常会包含 head/regression 等
        print("[finetune] unexpected:", unexpected)  # 一般为空

        # 重置微调头，避免旧权重影响
        def _reset_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

        self.model.head.apply(_reset_linear)
        for reg in self.model.regression:
            reg.apply(_reset_linear)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        path = path + '_dln_' + str(self.args.denoise_layers_num)
        if not os.path.exists(path):
            os.makedirs(path)
            # —— 加载预训练骨干（新增）—— #
        if getattr(self.args, 'pretrained_backbone', ''):
            self._load_pretrained_backbone(self.args.load_checkpoints)
        # optimizer
        # model_optim = self._select_optimizer()
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
            # 建议：预训练基线不做 init；若需要，做“多批次校准”
            if self.args.use_init_loss_finetune == 1:
                # init = calibrate_aux_weights_for_finetune(self.model, train_loader, K=64,
                #                              device=self.device, use_kendall=True)
                init = calibrate_aux_weights_for_finetune_coupled(self.model, train_loader, K=64,
                                             device=self.device)
                self.model.log_var_freq.data = torch.tensor(init["log_var_freq"], device=self.device)
                self.model.log_var_orth.data = torch.tensor(init["log_var_orth"], device=self.device)
                self.model.log_var_smooth.data = torch.tensor(init["log_var_smooth"], device=self.device)
                self.model.log_var_season_freq.data = torch.tensor(init["log_var_season_freq"], device=self.device)
                self.model.log_var_recon.data = torch.tensor(init["log_var_recon"], device=self.device)

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
                loss_freq = torch.exp(-self.model.log_var_freq) * freq_loss + 0.5 * self.model.log_var_freq
                loss_orth = torch.exp(-self.model.log_var_orth) * freq_loss + 0.5 * self.model.log_var_orth
                loss_smooth = torch.exp(-self.model.log_var_smooth) * freq_loss + 0.5 * self.model.log_var_smooth
                loss_season_freq = torch.exp(-self.model.log_var_season_freq) * freq_loss + 0.5 * self.model.log_var_season_freq
                loss_recon = torch.exp(-self.model.log_var_recon) * freq_loss + 0.5 * self.model.log_var_recon


                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                model_loss = model_criteria(pred_x, batch_y)
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
                loss = model_loss + loss_freq + loss_orth + loss_smooth + loss_season_freq +loss_recon




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
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
        self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0"))

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

    def test(self):
        test_data, test_loader = self._get_data(flag="test")

        preds = []
        trues = []

        folder_path = "./outputs/test_results/{}".format(self.args.data)
        folder_path = folder_path + '_dln_' + str(self.args.denoise_layers_num)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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

        # 获取当前时间
        now = datetime.now()

        # 方式1：标准格式化输出（示例：2023-10-25 15:30:45）
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        # print("当前时间:", formatted_time)
        f.write(
            "{0}->{1}, {2:.3f}, {3:.3f},{4},{5},{6},{7},log_var_orth={8},log_var_season_freq={9},log_var_freq={10},use_loss_compute={11},use_new_decomp={12},use_denoise={13},log_var_season_freq={14},{15}_{16}_{17}_{18}inner_{19}_{20}_{21}_{22} \n".format(
                self.args.input_len, self.args.pred_len, mse, mae,formatted_time,self.args.d_model,
                self.args.batch_size,self.args.n_heads,self.args.log_var_orth,self.args.log_var_season_freq,
                self.args.log_var_freq,
                self.args.use_loss_compute,self.args.use_new_decomp,self.args.use_denoise,self.args.use_inner_new_decomp,

                self.args.max_lag,
                self.args.num_scales,
                self.args.peak_threshold,
                self.args.distance,
                self.args.max_lag_inner,
                self.args.num_scales_inner,
                self.args.peak_threshold_inner,
                self.args.distance_inner,





            ))
        f.close()
        np.save(folder_path+os.sep+ 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+os.sep+'pred.npy', preds)
        np.save(folder_path +os.sep+'true.npy', trues)
        return
