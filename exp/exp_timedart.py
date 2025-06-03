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

warnings.filterwarnings("ignore")


class Exp_TimeDART(Exp_Basic):
    def __init__(self, args):
        super(Exp_TimeDART, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")

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
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            train_loader
        ):
            model_optim.zero_grad()

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
                # diff_loss.requires_grad = True
            # diff_loss = model_criterion(pred_x, batch_x)
            elif self.args.model == 'TimeDART':
                pred_x,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss = self.model(batch_x,batch_x_m,i)
                # diff_loss = self.model(batch_x)

                diff_loss = model_criterion(pred_x, batch_x)
                print("before:")

                print("diff_loss="+str(diff_loss.item()),"freq_loss=" + str(freq_loss.item()),"orth_loss=" + str(orth_loss.item()),"smoothness="+str(smoothness.item()),"season_freq_loss=" + str(season_freq_loss.item()),"recon_loss="+str(recon_loss.item()))

                # 自适应权重计算（不确定权重法）
                loss_freq = 1 / (2 * torch.exp(self.model.log_var_freq)) * freq_loss + 0.5 * self.model.log_var_freq
                loss_orth = 1 / (2 * torch.exp(self.model.log_var_orth)) * orth_loss + 0.5 * self.model.log_var_orth
                loss_smooth = 1 / (
                            2 * torch.exp(self.model.log_var_smooth)) * smoothness + 0.5 * self.model.log_var_smooth
                loss_season_freq = 1 / (2 * torch.exp(
                    self.model.log_var_season_freq)) * season_freq_loss + 0.5 * self.model.log_var_season_freq
                loss_recon = 1 / (2 * torch.exp(self.model.log_var_recon)) * recon_loss + 0.5 * self.model.log_var_recon
                if self.args.del_orth_loss == 1:
                    loss_orth = 0  # 618
                elif self.args.del_smoothness_loss == 1:
                    loss_smooth = 0  # 0.0025
                elif self.args.del_season_freq_loss == 1:
                    loss_season_freq = 0  # 10.6285
                elif self.args.del_freq_loss == 1:
                    loss_freq = 0  # 2.3
                elif self.args.del_recon_loss == 1:
                    loss_recon = 0
                # loss_freq = 0.01 * freq_loss
                # loss_orth = 0.1 * orth_loss
                # loss_smooth = 0.1 * smoothness
                print("after:")
                print("diff_loss="+str(diff_loss.item()),"loss_freq=" + str(loss_freq.item()),"loss_orth=" + str(loss_orth.item()),"loss_smooth="+str(loss_smooth.item()),"loss_season_freq=" + str(loss_season_freq.item()),"loss_recon="+str(loss_recon.item()))

                diff_loss = diff_loss + loss_freq  + loss_orth + loss_smooth +loss_season_freq+loss_recon

                # -----
                # 训练监控（前5个epoch打印详细信息）
                if i < 5:
                    # 打印自适应权重
                    print(f"\nEpoch {i} Adaptive Weights:")
                    print(f"  freq_weight: {loss_freq.item():.6f} (log_var={self.model.log_var_freq.item():.4f})")
                    print(f"  orth_weight: {loss_orth.item():.6f} (log_var={self.model.log_var_orth.item():.4f})")
                    print(f"  smooth_weight: {loss_smooth.item():.6f} (log_var={self.model.log_var_smooth.item():.4f})")
                    print(
                        f"  season_weight: {loss_season_freq.item():.6f} (log_var={self.model.log_var_season_freq.item():.4f})")
                    print(f"  recon_weight: {loss_recon.item():.6f} (log_var={self.model.log_var_recon.item():.4f})")

                    # 打印正则项贡献
                    print("\nRegularization Terms:")
                    print(f"  freq_reg: {0.5 * self.model.log_var_freq.item():.4f}")
                    print(f"  orth_reg: {0.5 * self.model.log_var_orth.item():.4f}")
                    print(f"  smooth_reg: {0.5 * self.model.log_var_smooth.item():.4f}")
                    print(f"  season_reg: {0.5 * self.model.log_var_season_freq.item():.4f}")
                    print(f"  recon_reg: {0.5 * self.model.log_var_recon.item():.4f}")

                    # 打印总损失组成
                    print("\nTotal Loss Breakdown:")
                    print(f"  total_loss: {diff_loss.item():.4f}")
                    print(f"    freq_component: {loss_freq.item() / diff_loss.item():.2%}")
                    print(f"    orth_component: {loss_orth.item() / diff_loss.item():.2%}")
                    print(f"    smooth_component: {loss_smooth.item() / diff_loss.item():.2%}")
                    print(f"    season_component: {loss_season_freq.item() / diff_loss.item():.2%}")
                    print(f"    recon_component: {loss_recon.item() / diff_loss.item():.2%}")
                # -----

            else:
                pred_x, = self.model(batch_x,batch_x_m,i)
                # diff_loss = self.model(batch_x)
                diff_loss = model_criterion(pred_x, batch_x)
            # diff_loss.requires_grad = True
            diff_loss.backward()
            model_optim.step()
            train_loss.append(diff_loss.item())

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
                elif self.args.model == 'TimeDART':
                    pred_x, freq_loss, orth_loss, smoothness,season_freq_loss,recon_loss = self.model(batch_x, batch_x_m, i)
                    # diff_loss = self.model(batch_x)
                    diff_loss = model_criterion(pred_x, batch_x)

                else:
                    pred_x = self.model(batch_x,batch_x_m)
                    # diff_loss = self.model(batch_x)
                    diff_loss = model_criterion(pred_x, batch_x)

                # diff_loss.backward()
                vali_loss.append(diff_loss.item())

        vali_loss = np.mean(vali_loss)

        return vali_loss

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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loader = tqdm(train_loader, desc="Training")

            print("Current learning rate: {:.7f}".format(model_scheduler.get_last_lr()[0]))

            self.model.train()
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
                loss_freq = 1/(2*torch.exp(self.model.log_var_freq)) * freq_loss + 0.5*self.model.log_var_freq
                loss_orth = 1/(2*torch.exp(self.model.log_var_orth)) * orth_loss + 0.5*self.model.log_var_orth
                loss_smooth = 1/(2*torch.exp(self.model.log_var_smooth)) * smoothness + 0.5*self.model.log_var_smooth
                loss_season_freq = 1/(2*torch.exp(self.model.log_var_season_freq)) * season_freq_loss + 0.5*self.model.log_var_season_freq
                loss_recon = 1/(2*torch.exp(self.model.log_var_recon)) * recon_loss + 0.5*self.model.log_var_recon
                # loss_freq = 0.01 * freq_loss
                # loss_orth = 0.1 * orth_loss
                # loss_smooth = 0.1 * smoothness
                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                loss = model_criteria(pred_x, batch_y)
                if self.args.del_orth_loss == 1:
                    loss_orth = 0  # 618
                elif self.args.del_smoothness_loss == 1:
                    loss_smooth = 0  # 0.0025
                elif self.args.del_season_freq_loss == 1:
                    loss_season_freq = 0  # 10.6285
                elif self.args.del_freq_loss == 1:
                    loss_freq = 0  # 2.3
                elif self.args.del_recon_loss == 1:
                    loss_recon = 0
                loss = loss + loss_freq + loss_orth + loss_smooth + loss_season_freq +loss_recon




                # -----
                # 训练监控（前5个epoch打印详细信息）
                if epoch < 5:
                    # 打印自适应权重
                    print(f"\nEpoch {epoch} Adaptive Weights:")
                    print(f"  freq_weight: {loss_freq.item():.6f} (log_var={self.model.log_var_freq.item():.4f})")
                    print(f"  orth_weight: {loss_orth.item():.6f} (log_var={self.model.log_var_orth.item():.4f})")
                    print(f"  smooth_weight: {loss_smooth.item():.6f} (log_var={self.model.log_var_smooth.item():.4f})")
                    print(
                        f"  season_weight: {loss_season_freq.item():.6f} (log_var={self.model.log_var_season_freq.item():.4f})")
                    print(f"  recon_weight: {loss_recon.item():.6f} (log_var={self.model.log_var_recon.item():.4f})")


                    # 打印正则项贡献
                    print("\nRegularization Terms:")
                    print(f"  freq_reg: {0.5 * self.model.log_var_freq.item():.4f}")
                    print(f"  orth_reg: {0.5 * self.model.log_var_orth.item():.4f}")
                    print(f"  smooth_reg: {0.5 * self.model.log_var_smooth.item():.4f}")
                    print(f"  season_reg: {0.5 * self.model.log_var_season_freq.item():.4f}")
                    print(f"  recon_reg: {0.5 * self.model.log_var_recon.item():.4f}")

                    # 打印总损失组成
                    print("\nTotal Loss Breakdown:")
                    print(f"  total_loss: {loss.item():.4f}")
                    print(f"    freq_component: {loss_freq.item() / loss.item():.2%}")
                    print(f"    orth_component: {loss_orth.item() / loss.item():.2%}")
                    print(f"    smooth_component: {loss_smooth.item() / loss.item():.2%}")
                    print(f"    season_component: {loss_season_freq.item() / loss.item():.2%}")
                    print(f"    recon_component: {loss_recon.item() / loss.item():.2%}")
                # -----






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
            "{0}->{1}, {2:.3f}, {3:.3f},{4} \n".format(
                self.args.input_len, self.args.pred_len, mse, mae,formatted_time
            )
        )
        f.close()
        np.save(folder_path+os.sep+ 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+os.sep+'pred.npy', preds)
        np.save(folder_path +os.sep+'true.npy', trues)
        return
