import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial

# from layers.linear import Linear
from layers.model_utils import default, extract, identity
from models.TimeDART import TimeDART, plot_tensors


# gaussian diffusion trainer class

# pred_len = 96


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Model(nn.Module):
    def __init__(
            self,
            configs,
    ):
        super(Model, self).__init__()

        self.configs = configs
        self.eta = configs.eta
        self.seq_length = configs.seq_len
        self.feature_size = configs.c_out

        self.model = TimeDART(self.configs)

        if configs.beta_schedule == 'linear':
            betas = linear_beta_schedule(configs.timesteps)
        elif configs.beta_schedule == 'cosine':
            betas = cosine_beta_schedule(configs.timesteps)
        else:
            raise ValueError(f'unknown beta schedule {configs.beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = configs.loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            configs.sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting

        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(self, x, t, training=False):
        model_output = self.model(x, t, training=training)
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, training=False):
        if training:
            training = False  # padding masks = 1
        maybe_clip = partial(torch.clamp, min=-2, max=2) if clip_x_start else identity
        x_start = self.output(x, t, training)
        # x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def sample(self, x):
        device = self.betas.device
        shape = x.shape
        img = x[:, :self.seq_length, :]
        # img = torch.randn(shape, device=device)

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, x, clip_denoised=True):
        shape = x.shape
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # img = torch.randn(shape, device=device)
        img = x[:, :self.seq_length, :]

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)
            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            sigma = 0
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = 0
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    @torch.no_grad()
    def fast_sample_tsr_old(self, x_T):
        """返回分解后的三个分量 (T, S, R)"""
        shape = x_T.shape
        batch, device = shape[0], self.betas.device
        T = self.num_timesteps

        # 时间步处理（确保非负）
        times = torch.linspace(T - 1, 0, steps=self.sampling_timesteps + 1)
        times = torch.round(times).long().clamp(min=0)
        time_pairs = list(zip(times[:-1], times[1:]))

        # 初始化分量存储
        x = x_T
        # 初始化分量累积器
        trend_accum = torch.zeros_like(x_T)
        season_accum = torch.zeros_like(x_T)
        resid_accum = torch.zeros_like(x_T)

        for t_now, t_next in time_pairs:
            t_next_clamped = max(0, t_next.item()) if t_next < 0 else t_next

            # 获取当前时间步的条件
            time_cond = torch.full((batch,), t_now, device=device, dtype=torch.long)

            # 模型预测三个分量（关键修改点1：同时获取三个分量）
            trend_pred, season_pred, resid_pred = self.model(x, time_cond)
            # 累积分量（按时间权重）
            weight = (1 - self.alphas_cumprod[t_now])  # 时间相关权重
            trend_accum += weight * trend_pred
            season_accum += weight * season_pred
            resid_accum += weight * resid_pred

            # 计算alpha系数（使用累积乘积）
            alpha_now = self.alphas_cumprod[int(t_now)]
            alpha_next = self.alphas_cumprod[int(t_next)] if t_next > 0 else 1.0

            # 确定性方向（对应公式3.3的S+R分离）
            pred_dir = (x - torch.sqrt(1 - alpha_now) * trend_pred) / torch.sqrt(alpha_now)

            # 噪声预测（对应季节+残差项）
            pred_noise = season_pred + resid_pred

            # 随机性控制（DDIM的eta参数）
            if self.eta > 0:
                sigma = self.eta * ((1 - alpha_now / alpha_next) * (1 - alpha_next) / (1 - alpha_now)).sqrt()
                noise = torch.randn_like(x)
            else:
                sigma = 0
                noise = 0

            # 更新x（对应公式逆向过程）
            x = (alpha_next.sqrt() * pred_dir
                 + (1 - alpha_next - sigma ** 2).sqrt() * pred_noise
                 + sigma * noise)

            # 最终分解结果（通过归一化保证x0 = T + S + R）
        total_weight = torch.sum(1 - self.alphas_cumprod[times.long()])
        final_trend = trend_accum / total_weight
        final_season = season_accum / total_weight
        final_resid = resid_accum / total_weight

        # 保证分解的守恒性
        reconstructed = final_trend + final_season + final_resid
        residual_error = x - reconstructed
        final_resid += residual_error  # 将重建误差归入残差项

        return final_trend, final_season, final_resid

    @torch.no_grad()
    def fast_sample_tsr(self, x_T):
        x_shape, device = x_T.shape, self.betas.device
        x_T = x_T.permute(0,2,1)
        x_T = torch.reshape(x_T, (x_T.shape[0]*x_T.shape[1],1,-1))

        # 生成时间步序列（从T-1到0）
        times = torch.linspace(self.num_timesteps - 1, 0,
                               steps=self.sampling_timesteps + 1)
        time_pairs = list(zip(times[:-1], times[1:]))

        x = x_T

        for t_now, t_next in tqdm(time_pairs, desc='sampling loop time step'):
            # 时间条件（当前步）
            time_cond = torch.full((x_T.shape[0]*x_T.shape[1],), t_now, device=device, dtype=torch.long)

            # 模型预测趋势项（核心修改）
            trend, season, residule = self.model.decomp_func(x, time_cond)

            # 计算alpha系数（使用累积乘积）
            alpha_now = self.alphas_cumprod[int(t_now)]
            alpha_next = self.alphas_cumprod[int(t_next)] if t_next > 0 else torch.tensor(1.0)

            # 确定性方向（对应公式3.3的S+R分离）
            pred_dir = (x - torch.sqrt(1 - alpha_now) * trend) / torch.sqrt(alpha_now)

            # 噪声预测（对应季节+残差项）
            pred_noise = season + residule

            # 随机性控制（DDIM的eta参数）
            if self.eta > 0:
                sigma = self.eta * ((1 - alpha_now / alpha_next) * (1 - alpha_next) / (1 - alpha_now)).sqrt()
                noise = torch.randn_like(x)
            else:
                sigma = 0
                noise = 0

            # 更新x（对应公式逆向过程）
            x = (alpha_next.sqrt() * pred_dir
                 + (1 - alpha_next - sigma ** 2).sqrt() * pred_noise
                 + sigma * noise)

        # 最终分解结果（使用t=0时的模型预测）
        final_trend, final_season, final_residule = self.model.decomp_func(x, torch.zeros_like(time_cond))
        return final_trend, final_season, final_residule

    def sample_tsr(self, x_start,noise=None):
        # 初始化纯噪声 x_T
        # xt = torch.randn(num_samples, config.seq_len, device=config.device)
        xt = default(noise, lambda: torch.randn_like(x_start)).to(x_start.device)
        B,S,C = xt.shape
        # 获取预定义的噪声调度参数
        alphas = self.alphas  # [timesteps], 对应公式中的 α_t
        betas = self.betas  # [timesteps], β_t = 1 - α_t

        for t in reversed(range(self.configs.timesteps)):
            # 当前时间步 t 的完整计算流程
            t_tensor = torch.full((B,), t, device=x_start.device)

            ######################################################
            # 逆向过程步骤1: 预测趋势项 T_θ(x_t, t)
            ######################################################
            trend, season, resid = self.model(xt, t_tensor)  # 模型同时预测 T/S/R

            ######################################################
            # 逆向过程步骤2: 计算均值 μ_θ (严格按公式实现)
            ######################################################
            alpha_t = alphas[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            # 核心计算公式 μ_θ = (x_t - sqrt(1-α_t)T_θ) / sqrt(α_t)
            mu_theta = (xt - sqrt_one_minus_alpha_t * trend) / sqrt_alpha_t

            ######################################################
            # 逆向过程步骤3: 采样 x_{t-1} (带噪声项)
            ######################################################
            if t > 0:
                # 计算噪声方差 σ_t (此处采用DDPM默认设置 σ_t = sqrt(β_t))
                sigma_t = torch.sqrt(betas[t])
                noise = torch.randn_like(xt)
                xt = mu_theta + sigma_t * noise
            else:
                # t=0时不添加噪声
                xt = mu_theta

        ######################################################
        # 最终成分分离 (严格按理论公式3.3)
        ######################################################
        # 最后一次迭代时 t=0, 此时:
        # x_0 = μ_θ = (x_1 - sqrt(1-α_1)T_θ) / sqrt(α_1)
        # 通过模型直接输出最终分解结果
        final_trend, final_season, final_resid = self.model(xt, torch.zeros_like(t_tensor))

        # 验证分解结果满足 x0 = T + S + R
        reconstructed = final_trend + final_season + final_resid
        assert torch.allclose(xt, reconstructed, atol=1e-3), "分解结果不满足 x0 = T + S + R"

        return final_trend, final_season, final_resid
    def generate_mts(self, x):
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        # return sample_fn(x)
        return self.fast_sample_tsr(x)


    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        index = int(t[0]) + 1
        x_middle = x_start[:, self.seq_length - index:-index, :]
        return x_middle

    def _train_loss(self, x_start, t, target=None, noise=None, training=True):
        # 输入维度检查 [B, L, C]
        b, seq_len, n_vars = x_start.shape

        # 噪声初始化
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 目标设定为原始信号
        target = x_start

        # 获取alpha系数（关键修改）
        alpha_cumprod = self.alphas_cumprod[t]  # [B]
        sqrt_alpha = torch.sqrt(alpha_cumprod)[:, None, None]  # 广播维度
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod)[:, None, None]

        # 单步前向过程（闭合公式）
        with torch.no_grad():
            # 初始分解（t=0时的分解）
            trend_0, season_0, resid_0 = self.model.decomp_func(x_start, torch.zeros_like(t))

        # 前向扩散公式（核心修改）
        xt = sqrt_alpha * (x_start - trend_0) + sqrt_one_minus_alpha * noise

        # 模型预测当前步分解
        pred_trend, pred_season, pred_resid = self.model.decomp_func(xt, t)

        # 重建信号
        reconstructed = pred_trend + pred_season + pred_resid

        # 损失计算（增强物理约束）
        loss_dict = {}

        # 1. 主重建损失（时域）
        # loss_recon = F.l1_loss(reconstructed, target)
        loss_recon = F.l1_loss(reconstructed, noise)
        loss_dict['recon'] = loss_recon

        # 2. 趋势平滑约束（二阶差分正则）
        d2_trend = torch.diff(pred_trend, n=2, dim=-1)
        loss_trend = 0.1 * torch.mean(d2_trend ** 2)
        loss_dict['trend'] = loss_trend

        # 3. 季节项频域约束（动态掩码）
        season_fft = torch.fft.rfft(pred_season, dim=-1)
        freq_mask = self.model._generate_lowpass_mask(
            self.model.freq_predictor(torch.mean(self.model.encoder(xt), dim=2))).to(season_fft.device)
        loss_season = 0.01 * torch.mean(torch.abs(season_fft * (1 - freq_mask[..., None])) ** 2)
        loss_dict['season'] = loss_season

        # 4. 残差稀疏性（L1正则 + 自相关约束）
        # 翻转序列用于模拟互相关 (沿时间轴翻转)
        input_reshaped = pred_resid.permute(1, 0, 2)  # (112,1,336) => (1,112,336)
        flipped = torch.flip(pred_resid, dims=[2])  # 形状保持 (B, 1, L)

        # 使用深度可分离卷积实现批量计算
        autocorr = F.conv1d(
            input=input_reshaped,  # 原始信号 (B, 1, L)
            weight=flipped,  # 翻转后的核 (B, 1, L)
            padding=seq_len - 1,  # 全填充模式
            groups=b  # 关键：每个样本独立计算
        )  # 输出形状 (B, 1, 2L-1)

        # 压缩通道维度并计算损失
        # 调整维度并计算损失
        autocorr = autocorr.permute(1, 0, 2).squeeze(1)  # 恢复形状 (B, 2L-1)
        loss_autocorr = 0.1 * torch.mean(torch.abs(autocorr))
        loss_resid = 0.01 * (torch.mean(torch.abs(pred_resid)) + loss_autocorr)
        loss_dict['resid'] = loss_resid

        # 5. 分量正交性约束
        orth_loss = 0.001 * (torch.mean(pred_trend * pred_season) +
                             torch.mean(pred_trend * pred_resid) +
                             torch.mean(pred_season * pred_resid))
        loss_dict['orth'] = orth_loss

        total_loss = sum(loss_dict.values())

        return total_loss

    def _train_loss_old(self, x_start, t, target=None, noise=None, training=True):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start[:, :, -self.seq_length:]
        target = x_start[:, :, -self.seq_length:]

        origin_trend, origin_season, origin_resi = self.model.decomp_func(x_start, t)

        # plot_tensors([x_start.permute(0,2,1),origin_trend.permute(0,2,1), origin_season.permute(0,2,1), origin_resi.permute(0,2,1)],'first_dec',1)

        alpha = self.sqrt_alphas_cumprod[t[0]]
        minus_alpha = self.sqrt_one_minus_alphas_cumprod[t[0]]

        xt = alpha * (x_start -origin_trend)+ minus_alpha * noise

        model_out,t_out,s_out,r_out = self.model(xt, t)

        # 计算损失
        loss_recon = F.l1_loss(model_out, target)
        loss_trend = F.mse_loss(torch.diff(t_out, n=2, dim=1), torch.zeros_like(t_out[:, 2:,:]))
        season_fft = torch.fft.rfft(s_out, dim=1)
        H_low = self.model._generate_lowpass_mask(
            self.model.freq_predictor(torch.mean(self.model.encoder(xt), dim=2)))
        loss_season = torch.mean(torch.abs(season_fft * (1 - H_low)) ** 2)
        loss_resid = torch.mean(torch.abs(r_out))
        total_loss = loss_recon + 0.1 * loss_trend + 0.01 * loss_season + 0.01 * loss_resid

        return total_loss

    def forward(self, x, **kwargs):
        b, n, c, device, feature_size, = *x.shape, x.device, self.feature_size
        x_ci = torch.reshape(x.permute(0,2,1),[b*c,1,n])
        if self.configs.task_name == "pretrain":


            assert c == feature_size, f'number of variable must be {feature_size}'
            t = torch.randint(0, self.num_timesteps, (b*c,), device=device).long()
            return self._train_loss(x_start=x_ci, t=t, **kwargs)
        else:
            t = torch.randint(0, self.num_timesteps, (b*c,), device=device).long()
            trend,season,resid = self.model(x_ci, t)



def get_config():
    import argparse
    import torch
    from exp.exp_timedart import Exp_TimeDART
    import random
    import numpy as np
    import os
    import platform
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="TimeDART")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        required=False,
        default="pretrain",
        help="task name, options:[pretrain, finetune]",
    )
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument(
        "--model_id", type=str, required=False, default="TimeDART", help="model id"
    )
    parser.add_argument(
        "--model", type=str, required=False, default="TimeDART", help="model name"
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=False, default="ETTh1", help="dataset type"
    )
    parser.add_argument(
        "--root_path", type=str, default="./datasets", help="root path of the data file"
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./outputs/checkpoints/",
        help="location of model fine-tuning checkpoints",
    )
    parser.add_argument(
        "--pretrain_checkpoints",
        type=str,
        default="./outputs/pretrain_checkpoints/",
        help="location of model pre-training checkpoints",
    )
    parser.add_argument(
        "--transfer_checkpoints",
        type=str,
        default="ckpt_best.pth",
        help="checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]",
    )
    parser.add_argument(
        "--load_checkpoints", type=str, default=None, help="location of model checkpoints"
    )
    parser.add_argument(
        "--select_channels",
        type=float,
        default=1,
        help="select the rate of channels to train",
    )

    # forecasting task
    parser.add_argument("--input_len", type=int, default=336, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=0, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )

    # model define
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=3, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--fc_dropout", type=float, default=0, help="fully connected dropout"
    )
    parser.add_argument("--head_dropout", type=float, default=0.1, help="head dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in ecoder",
    )
    parser.add_argument(
        "--individual", type=int, default=0, help="individual head; True 1 False 0"
    )
    parser.add_argument("--pct_start", type=float, default=0.3, help="pct_start")
    parser.add_argument("--patch_len", type=int, default=12, help="path length")
    parser.add_argument("--stride", type=int, default=12, help="stride")

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=5, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="decay", help="adjust learning rate")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0", help="device ids of multile gpus"
    )

    # Pre-train
    parser.add_argument(
        "--time_steps", type=int, default=1000, help="time steps in diffusion"
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", help="scheduler in diffusion"
    )

    parser.add_argument("--lr_decay", type=float, default=0.5, help="learning rate decay")
    parser.add_argument(
        "--real_scheduler", type=str, default="cosine", help="real_scheduler in diffusion"
    )
    parser.add_argument(
        "--imag_scheduler", type=str, default="quad", help="imag_scheduler in diffusion"
    )

    parser.add_argument("--down_sampling_method", type=str, default='avg', help="down_sampling_method")
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')

    parser.add_argument('--GT_d_model', type=int, default=512)
    parser.add_argument('--GT_d_ff', type=int, default=2048)
    parser.add_argument('--token_len', type=int, default=48)
    parser.add_argument('--GT_pooling_rate', type=list, default=[8, 4, 2, 1])
    parser.add_argument('--GT_e_layers', type=int, default=3)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument("--device", default='cuda:0', help="device")
    parser.add_argument('--positive_nums', type=int, default=3, help='masking series numbers')
    parser.add_argument('--rbtp', type=int, default=1,
                        help='0: rebuild the embedding of oral series; 1: rebuild oral series')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
    parser.add_argument('--masked_rule', type=str, default='geometric',
                        help='geometric, random, masked tail, masked head')
    parser.add_argument('--mask_rate', type=float, default=0.5, help='mask ratio')
    parser.add_argument('--seq_len', type=int, default=96, help='seq_len')
    parser.add_argument('--denoise_layers_num', type=int, default=3, help='denoise_layers_num')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # ---  扩散分解
    parser.add_argument('--timesteps', type=int, help='timesteps', default=100)
    parser.add_argument('--beta_schedule', type=str, help='beta_schedule', default='linear')
    parser.add_argument('--eta', type=int, help='eta', default=0)
    parser.add_argument('--loss_type', type=str, help='loss_type', default='l1')
    parser.add_argument('--sampling_timesteps', type=int, help='sampling_timesteps', default=1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    configs = get_config()

    model = Model(configs)
    x = torch.randn(configs.batch_size,configs.input_len,configs.c_out)
    res = model.generate_mts(x)
    res = torch.stack(res,dim=-1).sum(-1)
    res = torch.reshape(res,[batchsize,cout,seqlen])
    pass
