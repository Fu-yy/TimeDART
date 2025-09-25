from scipy.signal import find_peaks
import math

from layers.Autoformer_EncDec import moving_avg, series_decomp

import torch
import os
import torch.nn as nn

import matplotlib.pyplot as plt
# TimeDART_version2
# 只保留15分解，用新的分解策略
class ChannelIndependence(nn.Module):
    def __init__(
        self,
        input_len: int,
    ):
        super(ChannelIndependence, self).__init__()
        self.input_len = input_len

    def forward(self, x):
        """
        :param x: [batch_size, input_len, num_features]
        :return: [batch_size * num_features, input_len, 1]
        """
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, self.input_len, 1)
        return x
class AbsolutePositionEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
    ):
        super(AbsolutePositionEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        return self.pe[:, : x.size(1)]



class AddSosTokenAndDropLast(nn.Module):
    def __init__(self, sos_token: torch.Tensor):
        super(AddSosTokenAndDropLast, self).__init__()
        assert sos_token.dim() == 3
        self.sos_token = sos_token

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        sos_token_expanded = self.sos_token.expand(
            x.size(0), -1, -1
        )  # [batch_size * num_features, 1, d_model]
        x = torch.cat(
            [sos_token_expanded, x], dim=1
        )  # [batch_size * num_features, seq_len + 1, d_model]
        x = x[:, :-1, :]  # [batch_size * num_features, seq_len, d_model]
        return x

class Patch(nn.Module):
    def __init__(
        self,
        patch_len: int,
        stride: int,
    ):
        super(Patch, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        """
        :param x: [batch_size * num_features, input_len, 1]
        :return: [batch_size * num_features, num_patches, d_model]
                num_patches = seq_len = (input_len - patch_len) // stride + 1
        """
        x = x.squeeze(-1)  # [batch_size * num_features, input_len]
        x = x.unfold(-1, self.patch_len, self.stride)
        return x

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
    ):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.position_encoding = AbsolutePositionEncoding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        x = x + self.position_encoding(x)
        return self.dropout(x)
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_len: int,
        d_model: int,
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_embedding = nn.Linear(patch_len, d_model, bias=True)

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, patch_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        x = self.patch_embedding(x)
        return x



def generate_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    # in nn.MultiheadAttention
    # binary mask is used and True means not allowed to attend
    # so we use triu instead of tril
    return mask


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(x)
        output = self.norm2(x + self.dropout(ff_output))

        return output


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super(CausalTransformer, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, is_mask=True):
        # x: [batch_size * num_features, seq_len, d_model]
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len).to(x.device) if is_mask else None
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x



class Diffusion(nn.Module):
    def __init__(
        self,
        time_steps: int,
        device: torch.device,
        scheduler: str = "cosine",
    ):
        super(Diffusion, self).__init__()
        self.device = device
        self.time_steps = time_steps

        if scheduler == "cosine":
            self.betas = self._cosine_beta_schedule().to(self.device)
        elif scheduler == "linear":
            self.betas = self._linear_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Invalid scheduler: {scheduler=}")

        self.alpha = 1 - self.betas
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, self.time_steps)
        return betas

    def sample_time_steps(self, shape):
        return torch.randint(0, self.time_steps, shape, device=self.device)

    def noise(self, x, t):
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1)  # [batch_size * num_features, seq_len, 1]
        # x_t = sqrt(gamma_t) * x + sqrt(1 - gamma_t) * noise
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x):
        # x: [batch_size * num_features, seq_len, patch_len]
        t = self.sample_time_steps(x.shape[:2])  # [batch_size * num_features, seq_len]
        noisy_x, noise = self.noise(x, t)
        return noisy_x, noise, t

    def noise_with_t(self, x, t):
        """手动指定时间步t添加噪声"""
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1)  # [batch*features, seq_len, 1]
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

def calculate_scales_optimized(data, num_scales=3, max_lag=63, peak_threshold=0.3):
    """完全基于PyTorch的优化实现"""
    # 数据预处理 [B, L, C] -> [B, C, L]
    x = data.permute(0, 2, 1)
    B, C, L = x.shape

    # 批标准化
    x_mean = x.mean(dim=2, keepdim=True)
    x_centered = x - x_mean
    x_norm = x_centered / (x_centered.std(dim=2, keepdim=True) + 1e-8)

    # FFT加速自相关计算
    pad_size = L
    x_padded = torch.nn.functional.pad(x_norm, (0, pad_size))
    fft_x = torch.fft.rfft(x_padded, dim=2)
    acf = torch.fft.irfft(fft_x * fft_x.conj(), dim=2)[..., :L]
    acf = acf / (acf[..., :1] + 1e-8)

    # 聚合所有通道和批次
    mean_acf = acf.mean(dim=(0, 1))  # [L]

    # 峰值检测（PyTorch实现）
    # peaks = find_peaks_torch(
    #     mean_acf[:max_lag],
    #     height=peak_threshold,
    #     distance=10,
    #     max_num=num_scales * 2
    # )

    peaks, _ = find_peaks(mean_acf[:max_lag].cpu().numpy(),
                          height=peak_threshold,
                          distance=10)
    # peaks
    # 选择主要尺度
    if len(peaks) == 0:
        return [15, 31, 63][:num_scales]
    peaks =  torch.from_numpy(peaks).to(mean_acf.device)
    # 密度估计选择
    hist = torch.histc(
        peaks.float(),
        bins=max_lag,
        min=0,
        max=max_lag - 1
    )
    scales = []
    for _ in range(num_scales):
        max_bin = hist.argmax()
        if hist[max_bin] == 0:
            break
        scales.append(max_bin.item())
        hist[max(0, max_bin - 5):max_bin + 6] = 0
    last_scales = [s if s % 2 else s + 1 for s in scales[:num_scales]]
    # 补足长度
    if len(scales) < num_scales:
        default_scales = [15, 31, 63]
        for s in default_scales:
            if s not in last_scales:
                last_scales.append(s)
                if len(last_scales) == num_scales:
                    break
        # 若仍不足，循环填充
        while len(last_scales) < num_scales:
            last_scales.append(default_scales[-1])

    # 确保奇数尺寸
    return sorted(last_scales[:num_scales])

def find_peaks_torch(acf, height, distance=10, max_num=6):
    # 计算局部极大值 (比左右邻居大)
    peaks = torch.zeros_like(acf, dtype=torch.bool)
    peaks[1:-1] = (acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:])
    peaks &= (acf >= height)

    # 提取候选索引和值
    candidate_indices = torch.where(peaks)[0]
    if len(candidate_indices) == 0:
        return torch.tensor([], device=acf.device)
    peak_values = acf[candidate_indices]

    # 按峰值高度降序排序
    sorted_indices = torch.argsort(peak_values, descending=True)
    sorted_candidates = candidate_indices[sorted_indices]

    # 根据距离筛选
    selected = []
    for idx in sorted_candidates:
        if all(abs(idx - s) > distance for s in selected):
            selected.append(idx)
            if len(selected) >= max_num:
                break
    return torch.tensor(selected, device=acf.device, dtype=torch.long)
def calculate_scales_optimized_toech(data,init_conv_kernel=[15, 31, 63,95], num_scales=3, max_lag=63, peak_threshold=0.3,distance=10):
    # 数据预处理 [B, L, C] -> [B, C, L]
    x = data.permute(0, 2, 1).contiguous()
    B, C, L = x.shape
    max_lag = min(max_lag, L - 1)
    # 批标准化
    x_mean = x.mean(dim=2, keepdim=True)
    x_centered = x - x_mean
    x_norm = x_centered / (x_centered.std(dim=2, keepdim=True) + 1e-8)

    # FFT加速自相关计算（优化填充尺寸）
    pad_size = L - 1  # 原为 L
    x_padded = torch.nn.functional.pad(x_norm, (0, pad_size))
    fft_x = torch.fft.rfft(x_padded, dim=2)
    acf = torch.fft.irfft(fft_x * fft_x.conj(), dim=2)[..., :L]
    acf = acf / (acf[..., :1] + 1e-8)

    # 聚合所有通道和批次
    mean_acf = acf.mean(dim=(0, 1))  # [L]

    # # GPU峰值检测（替换SciPy）
    # peaks = find_peaks_torch(
    #     mean_acf[:max_lag],
    #     height=peak_threshold,
    #     distance=distance,
    #     max_num=num_scales * 2
    # )
    # peaks += 1  # 滞后值修正

    # GPU峰值检测（替换SciPy）
    peaks = find_peaks_torch(
        mean_acf[1:max_lag],
        height=peak_threshold,
        distance=distance,
        max_num=num_scales * 2
    )
    peaks += 1  # 滞后值修正
    # 选择主要尺度
    if len(peaks) == 0:
        return init_conv_kernel[:num_scales]

    # 密度估计选择（向量化优化）
    hist = torch.histc(peaks.float(), bins=max_lag, min=0, max=max_lag - 1)
    hist = hist.to(device=peaks.device)
    scales = []
    for _ in range(num_scales):
        max_bin = hist.argmax()
        if hist[max_bin] <= 0:
            break
        scales.append(max_bin.item())
        start = max(0, max_bin - 5)
        end = max_bin + 6
        hist[start:end] = 0

    last_scales = [s if s % 2 else s + 1 for s in scales[:num_scales]]
    # 补足长度
    if len(scales) < num_scales:
        default_scales = init_conv_kernel
        for s in default_scales:
            if s not in last_scales:
                last_scales.append(s)
                if len(last_scales) == num_scales:
                    break
        # 若仍不足，循环填充
        while len(last_scales) < num_scales:
            last_scales.append(default_scales[-1])

    return sorted(last_scales[:num_scales])
class FixedMultiScaleConv(nn.Module):
    def __init__(self, nvar, scales):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(nvar, nvar, kernel_size=k, padding=(k - 1) // 2,
                      groups=nvar, bias=False)
            for k in scales
        ])
        # 固定为均值滤波且不更新权重
        for conv in self.conv_layers:
            conv.weight.data = torch.ones_like(conv.weight) / conv.kernel_size[0]
            conv.weight.requires_grad = False

    def forward(self, x):
        return [conv(x) for conv in self.conv_layers]


class LightWeightGenerator(nn.Module):
    def __init__(self, nvar, num_scales):
        super().__init__()
        self.nvar = nvar
        self.num_scales = num_scales
        self.gen = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B,C,1]
            nn.Flatten(start_dim=1),  # [B,C]
            nn.Linear(nvar, nvar * num_scales),  # [B, C*K]
            nn.Unflatten(-1, (nvar, num_scales)),  # [B,C,K]
            nn.Softmax(dim=-1)  # 沿尺度维度归一化
        )

    def forward(self, x):
        weights = self.gen(x)  # [B,C,K]
        return weights.unsqueeze(-1)  # [B,C,K,1]


class StopLearnableMultiScaleDecomp(nn.Module):
    def __init__(self, nvar,num_scales,max_lag,peak_threshold,distance):
        super().__init__()
        self.nvar = nvar
        # 延迟初始化的组件
        self.peak_threshold = peak_threshold
        self.distance =distance
        self.fixed_convs = None  # 将在第一次forward时初始化
        self.scales = None  # 保存计算得到的scales
        self.num_scales = num_scales
        self.max_lag = max_lag
        # 多尺度卷积组（固定参数）
        # self.fixed_convs = FixedMultiScaleConv(nvar, scales)

        # 轻量权重生成器
        self.weight_gen = LightWeightGenerator(nvar, self.num_scales)
        # self.weight_gen = None
        # self.local_encoder = nn.Conv1d(nvar, nvar, 3, padding=1, groups=nvar)
    def forward(self, x):
        # 输入x形状: [Batch, Length, Channels]
        if self.fixed_convs is None:
            # 动态计算scales（基于当前批次数据）
            with torch.no_grad():  # 不参与梯度计算
                # self.scales = calculate_scales_optimized(
                #     x,
                #     num_scales=self.num_scales,
                #     # max_lag=self.max_lag,
                #     # peak_threshold=self.peak_threshold
                # )
                self.scales = calculate_scales_optimized_toech(
                    x,
                    num_scales=self.num_scales,
                    distance=self.distance,
                    peak_threshold=self.peak_threshold,
                    max_lag=self.max_lag,
                    # peak_threshold=self.peak_threshold
                )

            # 创建多尺度卷积模块
            self.fixed_convs = FixedMultiScaleConv(self.nvar, self.scales)
            self.add_module('fixed_convs', self.fixed_convs)  # 注册到模块

            # self.weight_gen = LightWeightGenerator(self.nvar, len(self.scales))
            # self.add_module('weight_gen', self.weight_gen)  # 注册到模块

            # 确保模块参数与输入数据在同一设备
            self.fixed_convs.to(x.device)
            # self.weight_gen.to(x.device)



        # 输入形状: [Batch, Length, Channels]
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        # x_local = self.local_encoder(x.permute(0, 2, 1))  # 捕捉局部模式
        # x = 0.8 * x.permute(0, 2, 1) + 0.2 * x_local  # 特征融合
        # 多尺度趋势提取




        trends = self.fixed_convs(x)  # list of [B,C,L]
        trends = torch.stack(trends, dim=-1)  # [B,C,L,K]

        # 生成融合权重
        weights = self.weight_gen(x)  # [B,C,K,1]
        weights = weights.permute(0,1,3,2)
        # 加权融合（广播机制）
        fused_trend = (trends * weights).sum(dim=-1)  # [B,C,L]

        # 季节项
        seasonal = x - fused_trend

        # 频域约束（抑制高频）
        # trend_fft = torch.fft.rfft(fused_trend, dim=-1)
        # freq_loss = torch.mean(torch.abs(trend_fft[..., 2:]))  # 忽略前5个低频
        # trend_freq_loss = freq_loss
        # # 平滑性约束
        # smooth_loss = torch.mean(torch.diff(fused_trend, n=2, dim=-1) ** 2)
        # # smooth_loss = torch.tensor(0.0)
        # # 正交约束
        # orth_loss = torch.mean((seasonal * fused_trend).sum(dim=-1) ** 2)
        # # orth_loss = torch.tensor(0.0)
        # # total_loss = freq_loss + 0.1 * smooth_loss + 0.1 * orth_loss
        # # 季节项高频激励（可选）
        # seasonal_fft = torch.fft.rfft(seasonal, dim=2)  # [B,C, L//2+1]
        # season_freq_loss = -torch.mean(torch.abs(seasonal_fft[..., 5:]))  # 激励高频
        # recon_loss=torch.tensor(0.0)
        # # === 科学修正的损失计算 ===
        # 1. 趋势项低频保护 + 高频抑制
        trend_fft = torch.fft.rfft(fused_trend, dim=-1)
        # 频域处理 (动态比例)
        n_freq = trend_fft.size(-1)
        k_protect = max(1, int(n_freq * 0.1))
        k_mid = min(n_freq, int(n_freq * 0.8))

        # 趋势损失: 保护极低频，抑制其他
        trend_freq_loss = torch.mean(torch.abs(trend_fft[..., k_protect:]))



        # 2. 季节项低频抑制（非高频激励！）
        seasonal_fft = torch.fft.rfft(seasonal, dim=-1)
        # 季节损失: 抑制极低频+高频，保护中频
        season_freq_loss = torch.mean(torch.abs(seasonal_fft[..., :k_protect])) + \
                           torch.mean(torch.abs(seasonal_fft[..., k_mid:]))

        # 3. 正交约束（科学修正）
        # 正交约束 (双方中心化)
        centered_trend = fused_trend - fused_trend.mean(dim=-1, keepdim=True)
        centered_seasonal = seasonal - seasonal.mean(dim=-1, keepdim=True)
        orth_loss = torch.mean((centered_seasonal * centered_trend).sum(dim=-1) ** 2)

        # 平滑约束 (混合一阶/二阶)
        smooth_loss = 0.0
        if L >= 3:
            first_diff = torch.mean(torch.diff(fused_trend, dim=-1) ** 2)
            second_diff = torch.mean(torch.diff(fused_trend, n=2, dim=-1) ** 2)
            smooth_loss = 0.6 * second_diff + 0.4 * first_diff
        # # 重构约束
        recon_loss = F.l1_loss(x, seasonal + fused_trend)
        return seasonal.permute(0, 2, 1), fused_trend.permute(0, 2, 1), trend_freq_loss,orth_loss,smooth_loss,season_freq_loss,recon_loss
def plot_tensors(tensor_list,file_name,index):
    project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    fig_path = project_path + os.sep + 'trend_figs'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.figure(figsize=(10, 6))  # 设置画布大小

    # 遍历每个tensor并绘制
    for i, tensor in enumerate(tensor_list):
        # 将tensor转换为numpy数组（自动处理GPU/CPU设备）
        data = tensor.cpu().detach().numpy()  # 兼容PyTorch张量
        # 如果是其他框架如TensorFlow，使用 data = tensor.numpy()

        plt.plot(data[0,:,-1],
                 label=f'Tensor {i + 1}',  # 自动生成图例标签
                 linestyle='-',  # 实线连接
                 alpha=0.7)  # 半透明效果

    # 添加图表元素
    plt.title('Tensor Line Plots', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)  # 网格线
    plt.legend()  # 显示图例

    # 自动调整布局并显示
    plt.tight_layout()
    plt.savefig( fig_path + os.sep + file_name + '_' +str(index)  +'_figure.png')

class FlattenHead(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        pred_len: int,
        dropout: float,
    ):
        super(FlattenHead, self).__init__()
        self.pred_len = pred_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        x = self.flatten(x)  # (batch_size, num_features, seq_len * d_model)
        x = self.forecast_head(x)  # (batch_size, num_features, pred_len)
        x = self.dropout(x)  # (batch_size, num_features, pred_len)
        x = x.permute(0, 2, 1)  # (batch_size, pred_len, num_features)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F



class TrendExtractorConv(nn.Module):
    def __init__(self, input_dim, d_model, kernel_size=3):
        super(TrendExtractorConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.activation = nn.ReLU()
        # self.pool = nn.AdaptiveAvgPool1d(1)  # 将每个特征图压缩为一个值

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        x = self.conv(x)  # [batch_size, d_model, seq_len]
        x = self.activation(x)
        return x.permute(0, 2, 1).contiguous()


class ConditionalEncoding(nn.Module):
    def __init__(self, input_dim, d_model):
        super(ConditionalEncoding, self).__init__()
        self.trend_extractor = TrendExtractorConv(input_dim, d_model)
        self.condition_proj = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        返回: [batch_size, 1, d_model]
        """
        trend = self.trend_extractor(x)  # [batch_size, 1, d_model]
        cond_encoded = self.condition_proj(trend)  # [batch_size, 1, d_model]
        cond_encoded = self.activation(cond_encoded)
        return cond_encoded




class DenoisingConditionDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads=1,dropout=0.1):
        super(DenoisingConditionDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*2, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.sigmoid = nn.Sigmoid()



    def compute_attention(self, q, k, v):
        scores = torch.matmul(q, k) / torch.sqrt(torch.tensor(q.size(-1)))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self,Noise_x,X,cond):

        # ------------------------------
        # 融合 Q 和 cond
        combined = torch.cat([Noise_x, cond], dim=-1)  # [batch_size, seq_len, embed_dim * 2]
        gate = self.sigmoid(self.gate(combined))  # [batch_size, seq_len, embed_dim]
        fused = gate * Noise_x + (1 - gate) * cond  # 融合后的表示
        # ------------------------------


        A,B,C,D = Noise_x[0, :, 0], cond[0, :, 0],fused[0,:,0], X[0, :, 0]

        query= fused
        key = X.permute(0,2,1).contiguous()
        value = X
        attn_output,_ = self.compute_attention(query, key, value)

        query = self.norm1(query + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(query)
        output = self.norm2(query + self.dropout(ff_output))


        # res0 = torch.stack((A, B,C, D,output[0,:,0]), dim=-1)
        # df0 = pd.DataFrame(res0.cpu().detach().numpy())  # 先转移到CPU
        # df0.to_excel('output2.xlsx', index=False, header=False)


        return output






class Model(nn.Module):
    """
    TimeDART
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.input_len = configs.input_len

        # For Model Hyperparameters
        self.d_model = configs.d_model
        self.num_heads = configs.n_heads
        self.feedforward_dim = configs.d_ff
        self.dropout = configs.dropout
        self.device = configs.device
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.inverse_embedding = nn.Linear(self.input_len, self.input_len)
        self.channel_independence = nn.ModuleList(
            [ChannelIndependence(
                input_len=self.input_len // (configs.down_sampling_window ** i),
            )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )



        # Patch
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch = Patch(
            patch_len=self.patch_len,
            stride=self.stride,
        )
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 1

        # Embedding
        self.enc_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            d_model=self.d_model,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
        )

        sos_token = torch.randn(1, 1, self.d_model, device=self.device)
        self.sos_token = nn.Parameter(sos_token, requires_grad=True)

        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(
            sos_token=self.sos_token,
        )

        # Encoder (Casual Trasnformer)
        self.diffusion = Diffusion(
            time_steps=configs.time_steps,
            device=self.device,
            scheduler=configs.scheduler,
        )
        self.encoder = CausalTransformer(
            d_model=configs.d_model,
            num_heads=configs.n_heads,
            feedforward_dim=configs.d_ff,
            dropout=configs.dropout,
            num_layers=configs.e_layers,
        )
        self.encoder_noise = CausalTransformer(
            d_model=configs.d_model,
            num_heads=configs.n_heads,
            feedforward_dim=configs.d_ff,
            dropout=configs.dropout,
            num_layers=configs.e_layers,
        )

        # 条件编码模块
        # 假设条件信息维度为 configs.condition_dim
        self.conditional_encoding = ConditionalEncoding(
            input_dim=self.d_model,
            d_model=self.d_model
        )

        # Decoder
        if self.task_name == "pretrain":


            self.projection = nn.ModuleList(
                [FlattenHead(
                seq_len=self.seq_len // (configs.down_sampling_window ** i),
                d_model=self.d_model,
                pred_len=configs.input_len,
                dropout=configs.head_dropout,
            )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            self.regression = nn.ModuleList([
                nn.Linear(self.input_len, self.input_len)
                for i in range(configs.down_sampling_layers + 1)
            ])



        elif self.task_name == "finetune":
            # self.head = FlattenHead(
            #     seq_len=self.seq_len,
            #     d_model=configs.d_model,
            #     pred_len=configs.pred_len,
            #     dropout=configs.head_dropout,
            # )


            # self.regression = nn.Linear(self.input_len, configs.pred_len)
            self.regression = nn.ModuleList([
                nn.Linear(self.input_len // (configs.down_sampling_window ** i), configs.pred_len)
                for i in range(configs.down_sampling_layers + 1)
            ])

            self.head = FlattenHead(
                    seq_len=self.seq_len,
                    d_model=self.d_model,
                    pred_len=configs.pred_len,
                    dropout=configs.head_dropout,
                )


            self.regression = nn.ModuleList([
                nn.Linear(self.input_len, configs.pred_len)
                for i in range(configs.down_sampling_layers + 1)
            ])

        # 自适应可学习权重参数
        # 自适应损失权重参数

        # self.log_var_freq = nn.Parameter(torch.log(torch.tensor(self.configs.log_var_freq)), requires_grad=True)
        # self.log_var_orth = nn.Parameter(torch.log(torch.tensor(self.configs.log_var_orth)), requires_grad=True)
        # self.log_var_smooth = nn.Parameter(torch.log(torch.tensor(self.configs.log_var_smooth)), requires_grad=True)
        # self.log_var_season_freq = nn.Parameter(torch.log(torch.tensor(self.configs.log_var_season_freq)),
        #                                         requires_grad=True)
        # self.log_var_recon = nn.Parameter(torch.log(torch.tensor(self.configs.log_var_recon)), requires_grad=True)


        # self.log_var_freq = nn.Parameter(torch.log(torch.tensor(1.0)))  # 频率损失初始权重 ≈0.5
        # self.log_var_orth = nn.Parameter(torch.log(torch.tensor(300.0)))  # 正交损失初始权重 ≈0.0017
        # self.log_var_smooth = nn.Parameter(torch.log(torch.tensor(0.001)))  # 平滑损失初始权重 ≈500
        # self.log_var_season_freq = nn.Parameter(torch.log(torch.tensor(5.0)))  # 季节频率初始权重 ≈0.1

        # self.log_var_freq = torch.tensor(self.configs.log_var_freq)
        # self.log_var_orth = torch.tensor(self.configs.log_var_orth)
        # self.log_var_smooth = torch.tensor(self.configs.log_var_smooth)
        # self.log_var_season_freq = torch.tensor(self.configs.log_var_season_freq)
        if self.configs.use_init_loss == 1:
            self.log_var_freq = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.log_var_orth = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.log_var_smooth = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.log_var_season_freq = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.log_var_recon = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            # 修正后的初始化参数（基于损失量级平衡）  ##  注意 这里的每个self.configs.log_var_freq表示的是方差
            self.log_var_freq = nn.Parameter(torch.tensor(self.configs.log_var_freq), requires_grad=True)
            self.log_var_orth = nn.Parameter(torch.tensor(self.configs.log_var_orth), requires_grad=True)
            self.log_var_smooth = nn.Parameter(torch.tensor(self.configs.log_var_smooth), requires_grad=True)
            self.log_var_season_freq = nn.Parameter(torch.tensor(self.configs.log_var_season_freq), requires_grad=True)
            self.log_var_recon = nn.Parameter(torch.tensor(self.configs.log_var_recon), requires_grad=True)


        self.decomp_multi = series_decomp(95)
        self.decomp_multi_learnable = StopLearnableMultiScaleDecomp(self.configs.c_out,max_lag=self.configs.max_lag,num_scales=self.configs.num_scales,peak_threshold=self.configs.peak_threshold,distance=self.configs.distance)
        # self.decomp_multi_learnable_second = StopLearnableMultiScaleDecomp(self.patch_len,num_scales=4)
        self.decomp_multi_learnable_third = StopLearnableMultiScaleDecomp(self.d_model, max_lag=self.configs.max_lag_inner,num_scales=self.configs.num_scales_inner,peak_threshold=self.configs.peak_threshold_inner,distance=self.configs.distance_inner)
        #

        # self.decomp_multi_learnable = StopLearnableMultiScaleDecomp(self.configs.c_out,max_lag=63,num_scales=1,peak_threshold=0.3,distance=10)
        # # self.decomp_multi_learnable_second = StopLearnableMultiScaleDecomp(self.patch_len,num_scales=4)
        # self.decomp_multi_learnable_third = StopLearnableMultiScaleDecomp(self.d_model, max_lag=31,num_scales=1,peak_threshold=0.1,distance=3)

        # self.decomp_multi_learnable = LearnableMultiScaleDecomp(self.configs.c_out)
        # self.decomp_multi_learnable_second = LearnableMultiScaleDecomp(self.patch_len,scales=[5, 13, 25])
        # self.decomp_multi_learnable_third = LearnableMultiScaleDecomp(self.d_model,scales=[5, 13, 25])
        self.denoise_layers_num = configs.denoise_layers_num

        self.denoise_layers_cond = nn.ModuleList([
            DenoisingConditionDecoder(
                embed_dim=configs.d_model,
                num_heads=configs.n_heads,
                dropout=configs.dropout,
            )
            for _ in range(self.denoise_layers_num)
        ])
    def init_adaptive_weights(self, sample_batch):
        """ 用样本数据初始化自适应权重 """
        with torch.no_grad():
            _, freq_loss, orth_loss, smooth_loss, season_freq_loss, recon_loss = self.forward(sample_batch,None)

            # 核心公式：log_var = log(损失值)
            self.log_var_freq.data = freq_loss + 1e-8
            self.log_var_orth.data = orth_loss + 1e-8
            self.log_var_smooth.data = smooth_loss + 1e-8
            self.log_var_season_freq.data = season_freq_loss + 1e-8
            self.log_var_recon.data = recon_loss + 1e-8
            # self.log_var_freq.data = torch.log(freq_loss + 1e-8)
            # self.log_var_orth.data = torch.log(orth_loss + 1e-8)
            # self.log_var_smooth.data = torch.log(smooth_loss + 1e-8)
            # self.log_var_season_freq.data = torch.log(season_freq_loss + 1e-8)
            # self.log_var_recon.data = torch.log(recon_loss + 1e-8)

    def pretrain(self, x,x_mask,i=0):

        # [batch_size, input_len, num_features]
        # Instance Normalization

        # x = torch.fft.fft(x,dim=-2).real
        mask_rate = 0.5
        lm=3
        positive_nums=1
        e_x =x
        batch_size, input_len, num_features = x.size()
        means = torch.mean(
            x, dim=1, keepdim=True
        ).detach()  # [batch_size, 1, num_features], detach from gradient
        x = x - means  # [batch_size, input_len, num_features]
        stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # [batch_size, 1, num_features]
        x = x / stdevs  # [batch_size, input_len, num_features]
        # x = self.inverse_embedding(x.permute(0,2,1)).permute(0,2,1)
        # 分解  1
        if self.configs.use_new_decomp == 1:
            x, trend,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss = self.decomp_multi_learnable(x)
        else:
            x, trend = self.decomp_multi(x)
            freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0),torch.tensor(0.0)

        # x, trend = x,x
        # Channel Independence
        x = self.channel_independence[0](x)  # [batch_size * num_features, input_len, 1]
        # Patch
        x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]
        # x_patch_f = torch.fft.fft(x_patch,dim=-2).imag

        # For Casual Transformer
        x_embedding = self.enc_embedding(
            x_patch
        )  # [batch_size * num_features, seq_len, d_model]




        if self.configs.use_sostoken == 1:
            x_embedding_bias = self.add_sos_token_and_drop_last(
                x_embedding
            )  # [batch_size * num_features, seq_len, d_model]
        else:

            x_embedding_bias=x_embedding
        if self.configs.use_positional_encoding == 1:

            x_embedding_bias = self.positional_encoding(x_embedding_bias)

        else:
            x_embedding_bias = x_embedding_bias
        # 分解  2
        # x_embedding_bias, _ = self.decomp_multi(x_embedding_bias)
        # x_embedding_bias, _ = x_embedding_bias,x_embedding_
        x_out = self.encoder(
            x_embedding_bias,
            is_mask=True,
        )  # [batch_size * num_features, seq_len, d_model]

        freq_loss_inner_list = []
        orth_loss_inner_list = []
        smoothness_inner_list = []
        season_freq_loss_inner_list = []
        recon_loss_inner_list = []
        # 获取总去噪层数和扩散模型总时间步
        num_denoise_layers = len(self.denoise_layers_cond)
        total_time_steps = self.diffusion.time_steps

        res_pred = []
        for layer_idx, layer in enumerate(self.denoise_layers_cond):
            x = self.channel_independence[0](x)  # [batch_size * num_features, input_len, 1]
            # Patch
            x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

            # 分解  3
            # x_patch, default_trend1 = self.decomp_multi_learnable_second(x_patch)
            # x_patch, default_trend = x_patch,x_patch
            # 动态计算当前层的时间步（退火策略）
            current_time_step = int((total_time_steps - 1) * (1 - layer_idx / num_denoise_layers))
            # 创建与输入形状匹配的时间步张量 [batch*features, seq_len]
            t = torch.full(
                (x_patch.size(0), x_patch.size(1)),
                current_time_step,
                device=self.device,
                dtype=torch.long
            )
            if self.configs.use_defire_noise:
                noise_x_patch, _ = self.diffusion.noise_with_t(x_patch, t)  # 使用指定t的噪声生成
            else:

                noise_x_patch, _, _ = self.diffusion(
                    x_patch
                )  # [batch_size * num_features, seq_len, patch_len]
            # 添加分层退火噪声

            noise_x_embedding = self.enc_embedding(
                noise_x_patch
            )  # [batch_size * num_features, seq_len, d_model]
            if self.configs.use_positional_encoding == 1:

                noise_x_embedding = self.positional_encoding(noise_x_embedding)
            else:
                noise_x_embedding = noise_x_embedding
            # noise_x_embedding = self.add_sos_token_and_drop_last(
            #     noise_x_embedding
            # )  # [batch_size * num_features, seq_len, d_model]

            noise_x_embedding_res = noise_x_embedding
            # denoise_input = noise_x_embedding
            if self.configs.use_inner_encoder == 1:

                denoise_input = self.encoder_noise(
                    noise_x_embedding,
                    is_mask=False,
                )  # [batch_size * num_features, seq_len, d_model]
            else:
                denoise_input = noise_x_embedding
            # noise end --------------------------
            # 分解  4
            # _, default_trend2 = self.decomp_multi_learnable_third(noise_x_embedding)
            # noise_x_embedding = noise_x_embedding-default_trend2


            # 分解  5
            # noise_x_embedding, _ = noise_x_embedding,noise_x_embedding
            if self.configs.use_inner_new_decomp == 1:

                x_out, x_out_trend,freq_loss_inner,orth_loss_inner,smoothness_inner ,season_freq_loss_inner,recon_loss_inner= self.decomp_multi_learnable_third(x_out)
            else:
                x_out, x_out_trend = self.decomp_multi(x_out)
                freq_loss_inner, orth_loss_inner, smoothness_inner, season_freq_loss_inner, recon_loss_inner =  torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0),torch.tensor(0.0)

            # x_out, x_out_trend = x_out,x_out

            # --------------------------- 添加条件 begin
            # 获取条件编码
            # cond_encoded = self.conditional_encoding_att(x_out_trend)  # [batch_size, 1, d_model]
            if self.configs.use_trend_layer == 1:

                cond_encoded = self.conditional_encoding(x_out_trend)  # [batch_size, 1, d_model]
            else:
                cond_encoded = x_out_trend
            # 扩展条件编码以匹配批次和特征维度
            # cond_encoded = cond_encoded.repeat_interleave(x_out_trend.size(0) // cond_encoded.size(0),
            #                                               dim=0)  # [batch_size * num_features, 1, d_model]
            # cond_encoded = cond_encoded.expand(-1, x_out_trend.size(1), -1)  # [batch_size * num_features, seq_len, d_model]

            # 将条件编码添加到嵌入中

            # --------------------------- 添加条件 end
            if self.configs.use_denoise == 1:
                # For Denoising Patch Decoder
                denoise_out = layer(
                    Noise_x=denoise_input,
                    X=x_out,
                    cond=cond_encoded
                )  # [batch_size * num_features, seq_len, d_model]

            else:
                denoise_out = denoise_input
            # default_trend2_reg = self.trend2_regression(default_trend2.permute(0,2,1)).permute(0,2,1)
            # denoise_out = denoise_out + default_trend2_reg
            denoise_out = denoise_out.reshape(
                batch_size, num_features, -1, self.d_model
            )  # [batch_size, num_features, seq_len, d_model]
            denoise_out = self.projection[0](denoise_out)  # [batch_size, input_len, num_features]
            x = denoise_out
            freq_loss_inner_list.append(freq_loss_inner)
            orth_loss_inner_list.append(orth_loss_inner)
            smoothness_inner_list.append(smoothness_inner)
            season_freq_loss_inner_list.append(season_freq_loss_inner)
            recon_loss_inner_list.append(recon_loss_inner)

            # predict_x = denoise_out + self.regression[0](trend.permute(0,2,1)).permute(0,2,1).contiguous()
            predict_x = denoise_out
            res_pred.append(predict_x)
        res_pred = torch.stack(res_pred,dim=-1).mean(dim=-1)
        predict_x = res_pred + self.regression[0](trend.permute(0,2,1)).permute(0,2,1).contiguous()
        # if i % 20 == 0:
        #     no_trend = [default_trend1,default_trend2,x_patch,noise_x_embedding]
        #     plot_tensors(no_trend,'notrend',i)
        #     trend_list = [default_trend1,default_trend2,x_patch,noise_x_embedding,trend]
        #     plot_tensors(trend_list,'withtrend',i)


        # Instance Denormalization
        predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
            1, input_len, 1
        )  # [batch_size, input_len, num_features]
        predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
            1, input_len, 1
        )  # [batch_size, input_len, num_features]
        # predict_x = torch.fft.ifft(predict_x,dim=-2).real
        total_freq = freq_loss + sum(freq_loss_inner_list)
        total_orth = orth_loss + sum(orth_loss_inner_list)
        total_smooth = smoothness + sum(smoothness_inner_list)
        total_season_freq = season_freq_loss + sum(season_freq_loss_inner_list)

        total_recon_loss = recon_loss + sum(recon_loss_inner_list)
        # return predict_x,0,0,0
        # return predict_x,total_freq,total_orth,total_smooth,total_season_freq

        return predict_x,total_freq,total_orth,total_smooth,total_season_freq,total_recon_loss

    def forecast(self, x,x_mark):
        # x = torch.fft.fft(x,dim=-2).real


        batch_size, _, num_features = x.size()
        means = torch.mean(x, dim=1, keepdim=True).detach()
        x = x - means
        stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / stdevs
        # x, trend = self.decomp_multi(x)
        # x = self.inverse_embedding(x.permute(0,2,1)).permute(0,2,1)

        if self.configs.use_new_decomp == 1:
            x, trend,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss = self.decomp_multi_learnable(x)
        else:
            x, trend = self.decomp_multi(x)
            freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0),torch.tensor(0.0)

        # x, trend = x,x
        x = self.channel_independence[0](x)  # [batch_size * num_features, input_len, 1]
        x = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

        # x = torch.fft.fft(x,dim=-2).imag
        x = self.enc_embedding(x)  # [batch_size * num_features, seq_len, d_model]


        # --------------------------- 添加条件 begin
        # 获取条件编码
        # cond_encoded = self.conditional_encoding(x)  # [batch_size, 1, d_model]
        # # 扩展条件编码以匹配批次和特征维度
        # cond_encoded = cond_encoded.repeat_interleave(x.size(0) // cond_encoded.size(0),
        #                                               dim=0)  # [batch_size * num_features, 1, d_model]
        # cond_encoded = cond_encoded.expand(-1, x.size(1), -1)  # [batch_size * num_features, seq_len, d_model]
        #
        # # 将条件编码添加到嵌入中
        # x = x + cond_encoded  # 结合条件编码

        # --------------------------- 添加条件 end

        if self.configs.use_positional_encoding == 1:

            x = self.positional_encoding(x)  # [batch_size * num_features, seq_len, d_model]
        else:
            x = x

        # x, _ = self.decomp_multi(x)
        # x, _ = x,x

        x = self.encoder(
            x,
            is_mask=False,
        )  # [batch_size * num_features, seq_len, d_model]
        x = x.reshape(
            batch_size, num_features, -1, self.d_model
        )  # [batch_size, num_features, seq_len, d_model]
        # x = torch.fft.ifft(x,dim=-2).real
        # forecast
        x = self.head(x)  # [bs, pred_len, n_vars]
        x = x + self.regression[0](trend.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # denormalization
        x = x * (stdevs[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
        x = x + (means[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)


        return x,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss
        # return x,0,0,0

    def forward(self, batch_x,x_mask,i=0):

        if self.task_name == "pretrain":
            return self.pretrain(batch_x,x_mask,i)
        elif self.task_name == "finetune":
            dec_out,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss = self.forecast(batch_x,x_mask)
            return dec_out[:, -self.pred_len: , :],freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")

def get_config():
    import argparse
    import torch
    import random
    import numpy as np
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='SimMTM')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    parser.add_argument('--train_only', type=bool, required=False, default=False,
                        help='perform training on full input dataset without validation and testing')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./datasets', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./outputs/checkpoints/',
                        help='location of model fine-tuning checkpoints')
    parser.add_argument('--pretrain_checkpoints', type=str, default='./outputs/pretrain_checkpoints/',
                        help='location of model pre-training checkpoints')
    parser.add_argument('--transfer_checkpoints', type=str, default='ckpt_best.pth',
                        help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
    parser.add_argument('--load_checkpoints', type=str, default=None, help='location of model checkpoints')
    parser.add_argument('--select_channels', type=float, default=1, help='select the rate of channels to train')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--patch_len', type=int, default=12, help='path length')
    parser.add_argument('--stride', type=int, default=12, help='stride')

    # optimization
    parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # Pre-train
    parser.add_argument('--lm', type=int, default=3, help='average masking length')
    parser.add_argument('--positive_nums', type=int, default=3, help='masking series numbers')
    parser.add_argument('--rbtp', type=int, default=1,
                        help='0: rebuild the embedding of oral series; 1: rebuild oral series')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
    parser.add_argument('--masked_rule', type=str, default='geometric',
                        help='geometric, random, masked tail, masked head')
    parser.add_argument('--mask_rate', type=float, default=0.5, help='mask ratio')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--time_steps', default=1000,type=int, help='device')
    # Pre-train

    parser.add_argument(
        "--scheduler", type=str, default="cosine", help="scheduler in diffusion"
    )

    parser.add_argument("--lr_decay", type=float, default=0.5, help="learning rate decay")
    parser.add_argument("--down_sampling_method", type=str, default='avg', help="down_sampling_method")
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
    parser.add_argument('--denoise_layers_num', type=int, default=3, help='num of denoise_layers_num')

    parser.add_argument(
        "--real_scheduler", type=str, default="cosine", help="real_scheduler in diffusion"
    )
    parser.add_argument(
        "--imag_scheduler", type=str, default="quad", help="imag_scheduler in diffusion"
    )

    configs = parser.parse_args()

    return configs



if __name__ == '__main__':
    configs = get_config()

    configs.task_name = 'pretrain'

    configs.seq_len = 336
    configs.e_layers = 3
    configs.enc_in = 7
    configs.dec_in = 7
    configs.c_out = 7
    configs.n_heads = 16
    configs.d_model = 32
    configs.d_ff = 64
    configs.positive_nums = 3
    configs.mask_rate = 0.5
    configs.learning_rate = 0.001
    configs.batch_size = 16
    configs.train_epochs = 5
    configs.input_len = 336

    configs.down_sampling_layers = 2
    configs.down_sampling_window = 2

    x= torch.randn(1,336,7)
    x_mark_enc= torch.randn(16,336,4)
    x_res= torch.randn(16,336,7)


    configs.device = x.device
    model = Model(configs)
    mask = torch.ones_like(x)
    # # x_enc 64 336 7 ; x_mark_enc 16 336 4 ； batch_x 16 336 7  mask 64 336 7
    c = model(x,x_mark_enc)
    d = 'end'
