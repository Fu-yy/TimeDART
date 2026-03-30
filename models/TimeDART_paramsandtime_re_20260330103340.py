import torch
import torch.nn as nn
from einops import rearrange, repeat
import pandas as pd
from scipy.signal import find_peaks

from layers.Autoformer_EncDec import moving_avg, series_decomp
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
# from layers.SelfAttention_Family import DSAttention, AttentionLayer, FullAttention
from layers.TimeDART_EncDec import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    Diffusion,
    DenoisingPatchDecoder,
)
from layers.Embed import Patch, PatchEmbedding, PositionalEncoding
from utils.augmentations import masked_data
import torch.nn.functional as F
import torch
import os
import torch.nn as nn

import matplotlib.pyplot as plt

from utils.masking import generate_causal_mask


# TimeDART_version2
# 只保留15分解，用新的分解策略

def geometric_block_mask_torch(B, L, C, masking_ratio=0.5, lm=13,
                               shared_channels=True, device='cuda'):
    """
    返回: M (bool) 形状 [B, L, C]，True=遮盖（drop），False=保留（keep）
    近似使得遮盖比例≈masking_ratio、平均段长≈lm
    """
    # 反推种子概率 p_seed，使膨胀后期望遮盖率≈masking_ratio  （独立近似）
    # 1 - (1 - p_seed)^lm ≈ masking_ratio => p_seed ≈ 1 - (1 - masking_ratio)^(1/lm)
    p_seed = 1.0 - (1.0 - masking_ratio) ** (1.0 / max(1, lm))

    G = 1 if shared_channels else C               # 共享通道 or 每通道独立
    seeds = (torch.rand(B, G, L, device=device) < p_seed).float()  # [B,G,L]

    # 用全 1 核做“膨胀”，得到连续段
    kernel = torch.ones(G, 1, lm, device=device)  # depthwise conv 权重
    # padding 使长度不变
    pad = lm // 2
    seg = F.conv1d(seeds, kernel, padding=pad, groups=G)  # [B,G,L]
    M = (seg > 0)  # bool，True=被覆盖区域

    if shared_channels:
        M = M.expand(-1, C, -1)  # [B,C,L]
    # 转回 [B,L,C] 且 True=遮盖
    M = M.permute(0, 2, 1).contiguous()
    return M


def block_mask_torch(B, L, C, masking_ratio=0.5, block=12,
                     shared_channels=True, variable_block=False, device='cuda'):
    """
    返回: M (bool) 形状 [B, L, C]，True=遮盖，False=保留
    固定或随机块遮盖，纯 Torch，无 Python 循环。
    """
    # 计算期望块数
    num_mask = int(L * masking_ratio)
    num_blocks = max(1, num_mask // max(1, block))
    # 起点均匀采样
    starts = torch.randint(0, max(1, L - block + 1), (B, num_blocks), device=device)  # [B, Nb]

    if variable_block:
        # 每个块随机长度（[block//2, 3*block//2]）
        lengths = torch.randint(max(1, block//2), 3*block//2 + 1, (B, num_blocks), device=device)
    else:
        lengths = torch.full((B, num_blocks), block, device=device, dtype=torch.long)

    # 构造每块的索引：start + [0..len-1]
    # 先取最大的长度，构造一个 base，再用掩码屏蔽超出部分
    max_len = int(lengths.max().item())
    base = torch.arange(max_len, device=device)[None, None, :]              # [1,1,K]
    idx = starts[..., None] + base                                          # [B,Nb,K]
    valid = (base < lengths[..., None])                                     # [B,Nb,K]  bool
    idx = torch.clamp(idx, max=L-1)

    # scatter 到 [B,L] 的平面 mask
    M2d = torch.zeros(B, L, device=device, dtype=torch.bool)                # [B,L]
    # 展平后 scatter
    flat_idx = idx.view(B, -1)
    flat_val = valid.view(B, -1)
    # 需要把 bool 转为同形 src（True->1）
    src = flat_val
    # 用 advanced indexing（比 scatter 好理解）
    for b in range(B):
        M2d[b, flat_idx[b][src[b]]] = True

    # 升到通道维
    if shared_channels:
        M = M2d[:, :, None].expand(-1, -1, C)     # [B,L,C]
    else:
        # 每通道独立：复制每个 b 的 mask 到不同 c（也可为每 c 各采一份 starts/lengths）
        M = M2d[:, :, None].expand(-1, -1, C).clone()

    return M

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
            with torch.no_grad():
                self.scales = calculate_scales_optimized_toech(
                    x,
                    num_scales=self.num_scales,
                    distance=self.distance,
                    peak_threshold=self.peak_threshold,
                    max_lag=self.max_lag,
                )

            self.fixed_convs = FixedMultiScaleConv(self.nvar, self.scales).to(x.device)



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

        def _band_abs_mean(spec, s, e):
            # spec: [..., F]
            s = int(s);
            e = int(e)
            if e <= s:
                return spec.new_tensor(0.0)
            band = spec[..., s:e]
            # 最后一层兜底：把 NaN/Inf 变成 0
            return torch.nan_to_num(band.abs().mean(), nan=0.0, posinf=0.0, neginf=0.0)

        # # === 科学修正的损失计算 ===
        # 1. 趋势项低频保护 + 高频抑制

        trend_fft = torch.fft.rfft(fused_trend, dim=-1)
        seasonal_fft = torch.fft.rfft(seasonal, dim=-1)
        n_freq = trend_fft.size(-1)

        # 合理的边界（确保有至少1个频点可用）
        k_protect = max(1, min(n_freq - 1, int(n_freq * 0.1)))
        k_mid = max(k_protect + 1, min(n_freq - 1, int(n_freq * 0.8)))

        # k_protect = max(1, int(n_freq * 0.1))
        # k_mid = min(n_freq, int(n_freq * 0.8))

        # 趋势损失: 保护极低频，抑制其他
        # trend_freq_loss = torch.mean(torch.abs(trend_fft[..., k_protect:]))



        # 2. 季节项低频抑制（非高频激励！）
        # seasonal_fft = torch.fft.rfft(seasonal, dim=-1)
        # # 季节损失: 抑制极低频+高频，保护中频
        # season_freq_loss = torch.mean(torch.abs(seasonal_fft[..., :k_protect])) + \
        #                    torch.mean(torch.abs(seasonal_fft[..., k_mid:]))

        trend_freq_loss = _band_abs_mean(trend_fft, k_protect, n_freq)  # 抑制保护区以外
        season_low = _band_abs_mean(seasonal_fft, 0, k_protect)  # 季节项抑制极低频
        season_high = _band_abs_mean(seasonal_fft, k_mid, n_freq)  # 以及极高频
        season_freq_loss = season_low + season_high

        # 3. 正交约束（科学修正）
        # 正交约束 (双方中心化)
        # centered_trend = fused_trend - fused_trend.mean(dim=-1, keepdim=True)
        # centered_seasonal = seasonal - seasonal.mean(dim=-1, keepdim=True)
        # orth_loss = torch.mean((centered_seasonal * centered_trend).sum(dim=-1) ** 2)
        # 正交/平滑加兜底
        centered_trend = fused_trend - fused_trend.mean(dim=-1, keepdim=True)
        centered_seasonal = seasonal - seasonal.mean(dim=-1, keepdim=True)
        orth_loss = torch.nan_to_num(((centered_seasonal * centered_trend).sum(dim=-1) ** 2).mean(),
                                     nan=0.0, posinf=0.0, neginf=0.0)

        # 平滑约束 (混合一阶/二阶)
        # smooth_loss = 0.0
        if L >= 3:
            first_diff = torch.diff(fused_trend, dim=-1).pow(2).mean()
            second_diff = torch.diff(fused_trend, n=2, dim=-1).pow(2).mean()
            smooth_loss = 0.6 * first_diff + 0.4 * second_diff
        else:
            smooth_loss = fused_trend.new_tensor(0.0)
        # # 重构约束
        recon_loss = torch.nn.functional.l1_loss(x, seasonal + fused_trend)
        # 最终再统一 nan_to_num 一次（双保险）
        trend_freq_loss = torch.nan_to_num(trend_freq_loss, nan=0.0, posinf=0.0, neginf=0.0)
        season_freq_loss = torch.nan_to_num(season_freq_loss, nan=0.0, posinf=0.0, neginf=0.0)
        smooth_loss = torch.nan_to_num(smooth_loss, nan=0.0, posinf=0.0, neginf=0.0)
        orth_loss = torch.nan_to_num(orth_loss, nan=0.0, posinf=0.0, neginf=0.0)
        recon_loss = torch.nan_to_num(recon_loss, nan=0.0, posinf=0.0, neginf=0.0)


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


class StableCondDenoiser(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1, use_film=1):
        super().__init__()
        self.use_film = use_film

        # 1. Self-Attention 层 (处理 noisy input 内部关系)
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # 2. Cross-Attention 层 (引入 trend 上下文)
        self.ln_cross = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)  # 对 context 做 norm
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # 3. FFN
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.Dropout(dropout)
        )

        # FiLM 仅用于调制输入 (可选，或者用于调制 Norm 层 - AdaLN)
        if use_film == 1:
            self.film = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.SiLU(),
                nn.Linear(d_model, 2 * d_model)
            )

    def forward(self, noisy, context, cond=None, key_padding_mask=None):
        # noisy: [B, S, D] (Seasonal / X)
        # context: [B, S, D] (Trend)

        x = noisy

        # (0) FiLM 调制 (可选，保持你原有的逻辑，但在 Self-Attn 之前)
        if self.use_film == 1 and cond is not None:
            # 注意：此处建议只调制 x，或者使用 AdaLN。
            # 这里沿用你简单的仿射变换逻辑
            gamma_beta = self.film(torch.cat([x, cond], dim=-1))  # 这里可能需要调整 cat 的逻辑，或者 cond 已经是 projection 过的
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            x = (1.0 + 0.1 * torch.tanh(gamma)) * x + 0.1 * torch.tanh(beta)

        # (1) Self-Attention Block (关键新增！)
        # 残差连接 x = x + Attention(Norm(x))
        residual = x
        # x_norm = self.ln_self(x)
        # 这里的 Q=K=V 都是 noisy 本身，这样才能学会序列内部的高频周期性
        # attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        # x = residual + attn_out

        # (2) Cross-Attention Block (Trend 指导)
        # Q = Noisy(processed), K=V = Trend
        # residual = x
        x_norm = self.ln_cross(x)
        k_val = self.ln_kv(context)
        v=x_norm
        attn_out, _ = self.cross_attn(x_norm, k_val, v, key_padding_mask=key_padding_mask)
        x = residual + attn_out

        # (3) FFN
        residual = x
        # x = residual + self.ff(self.ln_ff(x))

        return x
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attention_cross = nn.MultiheadAttention(
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

    def forward(self, x, cond,mask):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        if cond is not None:
            cross_x,_ = self.attention_cross(x, cond, x, attn_mask=mask)
            x = self.norm1(x + self.dropout(cross_x))

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
        # 1. 定义那个 "长度为1的 parameter"
        # 这就是你的 [Trend] Token，类似 BERT 的 [CLS]
        self.trend_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 2. Cross-Attention: 让 Token 去读 Trend 序列
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.film = nn.Sequential(
            nn.Linear(d_model, 2*d_model)
            # nn.Linear(d_model, 2*d_model)  # -> [gamma, beta]
        )

    def forward(self, x, context,is_mask=True):
        # x: [batch_size * num_features, seq_len, d_model]
        seq_len = x.size(1)

        B = x.shape[0]
        if context is not None:
            # h = torch.cat([x, context], dim=-1)
            h = x+ context
            gamma_beta = self.film(h)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            q = (1.0 + 0.5 * torch.tanh(gamma)) * x + 0.5 * torch.tanh(beta)
            q = q + h
            # --- Step 1: 学习 Trend 信息 (Summary) ---
            # 扩展 token 到 batch 大小: [B, 1, D]
            # token = self.trend_token.expand(B, -1, -1)

            # Q = Token, K = Trend, V = Trend
            # 这一步把 S 长度的 Trend 压缩成了 1 长度的 Summary
            trend_summary, _ = self.cross_attn(
                query=h,
                key=q,
                value=q
            )
            # trend_summary = self.norm(trend_summary)  # [B, 1, D]
            # --- Step 2: 拼接 (Prompt Injection) ---
            # 把 Summary 拼在 Noisy Input 前面 -> [B, 1+S, D]
            # combined_input = torch.cat([trend_summary, x], dim=1)
            combined_input = trend_summary+x
        else:
            combined_input=x
        mask = generate_causal_mask(seq_len).to(x.device) if is_mask else None
        for layer in self.layers:
            x = layer(x, combined_input,mask)
        # 扔掉第一个 token，只保留去噪后的序列
        # if context is not None:
        #     x = x[:, 1:, :]  # [B, S, D]
        x = self.norm(x)
        return x



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
        self.decomp_multi_learnable = StopLearnableMultiScaleDecomp(self.configs.c_out,max_lag=self.configs.max_lag,num_scales=self.configs.num_scales,peak_threshold=self.configs.peak_threshold,distance=10)

        self.denoise_layers_num = configs.denoise_layers_num


        # 在 __init__ 尾部、其它属性旁边
        self.pretrain_noise = getattr(configs, 'pretrain_noise', 'mask')  # 'mask' 或 'tembed'
        self.mask_block = getattr(configs, 'lm', 12)
        self.mask_ratio = getattr(configs, 'mask_ratio', 0.5)
        # ------- t-embedding（A 版） -------
        if getattr(configs, 'use_t_embed', 1) == 1:
            self.t_embed = nn.Sequential(
                nn.Embedding(configs.time_steps, self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.SiLU(),
                nn.Linear(self.d_model, self.d_model),
            )
        else:
            self.t_embed = None

        # ------- FiLM 条件调制（两版都可用） -------
        self.cond_to_gamma = nn.Linear(self.d_model * 2, self.d_model)  # concat(t_emb, cond)
        self.cond_to_beta = nn.Linear(self.d_model * 2, self.d_model)
        self.cat_2 = nn.Linear(self.d_model * 2, self.d_model)
        self.enc_embedding_trend = PatchEmbedding(
            patch_len=self.patch_len,
            d_model=self.d_model,
        )
        self.stable_denoiser = StableCondDenoiser(d_model=self.d_model, n_heads=self.num_heads, dropout=self.dropout,
                                                  use_film=True)
        self.cond_proj = nn.Linear(self.d_model, self.d_model)  # 轻量投影，避免尺度打架

        # 在 __init__ 中添加
        self.ib_projector = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),  # 输出 [mu, logvar]
            nn.Tanh()  # 限制范围，防止梯度爆炸
        )

        # ===== ablation mode =====
        # full / no_aasd / no_tcsr / no_aasd_no_tcsr
        self.ablation_mode = self.configs.ablation_mode


    def set_ablation_mode(self, mode="full"):
        """
        mode:
            - full
            - no_aasd
            - no_tcsr
            - no_aasd_no_tcsr
        """
        assert mode in ["full", "no_aasd", "no_tcsr", "no_aasd_no_tcsr"]
        self.ablation_mode = mode

    def _disable_aasd(self):
        return self.ablation_mode in ["no_aasd", "no_aasd_no_tcsr"]

    def _disable_tcsr(self):
        return self.ablation_mode in ["no_tcsr", "no_aasd_no_tcsr"]

    def _zero_loss_like(self, x):
        z = torch.zeros(1, device=x.device, dtype=x.dtype)
        return z, z, z, z, z

    def _run_decomposition(self, x):
        """
        统一管理是否启用 AASD
        返回:
            seasonal, trend, freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss
        """
        if self._disable_aasd():
            seasonal = x
            trend = torch.zeros_like(x)
            freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = self._zero_loss_like(x)
            return seasonal, trend, freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss

        # ===== 原始逻辑 =====
        if self.configs.use_new_decomp == 1:
            seasonal, trend, freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = self.decomp_multi_learnable(x)
        elif self.configs.use_new_decomp == 2:
            # 你原来这里 seasonal, trend = x, x 不太合理，会重复叠加
            # 改成 seasonal=x, trend=0 更适合作为无分解/无AASD基线
            seasonal = x
            trend = torch.zeros_like(x)
            freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = self._zero_loss_like(x)
        else:
            seasonal, trend = self.decomp_multi(x)
            freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = self._zero_loss_like(x)

        return seasonal, trend, freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss

    def _use_pretrain_encoder_now(self):
        return (self.configs.use_pretrain_encoder == 1) and (not self._disable_tcsr())

    def _use_finetune_encoder_now(self):
        return (self.configs.use_finetune_encoder == 1) and (not self._disable_tcsr())

    def count_active_parameters(self, trainable_only=False):
        """
        按当前 ablation_mode 统计“实际启用模块”的参数数目。
        注意：不用 sum(self.parameters())，否则去掉的模块参数仍会被统计进去。
        """
        skip_prefixes = []

        if self._disable_aasd():
            skip_prefixes.append("decomp_multi_learnable.")

        if self._disable_tcsr():
            skip_prefixes.append("encoder.")

        total = 0
        for name, p in self.named_parameters():
            if trainable_only and (not p.requires_grad):
                continue

            skip_flag = False
            for prefix in skip_prefixes:
                if name.startswith(prefix):
                    skip_flag = True
                    break
            if skip_flag:
                continue

            total += p.numel()
        return total
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

    def pretrain(self, x, x_mask, i=0):
        mask_rate = 0.5
        lm = 3
        positive_nums = 1
        e_x = x
        device = x.device
        batch_size, input_len, num_features = x.size()

        means = torch.mean(x, dim=1, keepdim=True).detach()
        x = x - means
        stdevs = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdevs

        B, L, C = x.shape

        # ===== 统一由辅助函数控制 AASD 是否启用 =====
        seasonal, trend, freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = self._run_decomposition(x)

        M = None
        eps = None

        if self.configs.pretrain_mode == 'mask':
            if self.configs.use_geo_mask:
                M = geometric_block_mask_torch(
                    B, L, C,
                    masking_ratio=self.configs.mask_ratio,
                    lm=self.configs.mask_block,
                    shared_channels=True,
                    device=x.device
                )
                if self.configs.destroy_mode == 'season':
                    seasonal_tilde = torch.where(M, torch.zeros_like(seasonal), seasonal)
                    x_tilde = seasonal_tilde
                elif self.configs.destroy_mode == 'trend':
                    trend_tilde = torch.where(M, torch.zeros_like(trend), trend)
                    x_tilde = trend_tilde + seasonal
                elif self.configs.destroy_mode == 'x':
                    x_tilde = torch.where(M, torch.zeros_like(x), x)
            else:
                M = block_mask_torch(
                    B, L, C,
                    masking_ratio=self.configs.mask_ratio,
                    block=self.configs.mask_block,
                    shared_channels=True,
                    variable_block=True,
                    device=x.device
                )
                if self.configs.destroy_mode == 'season':
                    seasonal_tilde = seasonal.masked_fill(M, 0.0)
                    x_tilde = seasonal_tilde
                elif self.configs.destroy_mode == 'trend':
                    trend_tilde = trend.masked_fill(M, 0.0)
                    x_tilde = trend_tilde + seasonal
                elif self.configs.destroy_mode == 'x':
                    x_tilde = x.masked_fill(M, 0.0)

        elif self.configs.pretrain_mode == 'noise':
            t = self.diffusion.sample_time_steps((B,))
            gamma_t = self.diffusion.gamma[t].view(B, 1, 1).to(device)

            if self.configs.destroy_mode == 'season':
                eps = torch.randn_like(seasonal)
                noisy_seasonal = torch.sqrt(gamma_t) * seasonal + torch.sqrt(1 - gamma_t) * eps
                x_tilde = noisy_seasonal
            elif self.configs.destroy_mode == 'trend':
                eps = torch.randn_like(trend)
                noisy_trend = torch.sqrt(gamma_t) * trend + torch.sqrt(1 - gamma_t) * eps
                x_tilde = seasonal + noisy_trend
            elif self.configs.destroy_mode == 'x':
                eps = torch.randn_like(x)
                x_tilde = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * eps
        else:
            x_tilde = seasonal + trend

        # ===== seasonal branch =====
        x_ci = self.channel_independence[0](x_tilde)
        x_patch = self.patch(x_ci)
        x_emb = self.enc_embedding(x_patch)

        if self.configs.use_sostoken == 1:
            x_embedding_bias = self.add_sos_token_and_drop_last(x_emb)
        else:
            x_embedding_bias = x_emb

        if self.configs.use_positional_encoding == 1:
            x_embedding_bias = self.positional_encoding(x_embedding_bias)

        # ===== trend condition =====
        trend_ci = self.channel_independence[0](trend)
        trend_patch = self.patch(trend_ci)
        trend_emb = self.enc_embedding_trend(trend_patch)

        # ===== 统一由辅助函数控制 TCSR/encoder 是否启用 =====
        if self._use_pretrain_encoder_now():
            x_emb = self.encoder(
                x_embedding_bias,
                context=trend_emb,
                is_mask=False,
            )
        else:
            x_emb = x_embedding_bias

        cond_trend = trend_emb

        if self.configs.pretrain_mode == 'noise':
            if self.t_embed is not None:
                t_big = t.view(B, 1).repeat(1, x_emb.size(1))
                t_big = t_big.repeat_interleave(C, dim=0)
                t_emb = self.t_embed(t_big)
            else:
                t_emb = torch.zeros_like(x_emb)
        else:
            t_emb = torch.zeros_like(x_emb)

        feat = x_emb
        denoised_bsd = None

        if getattr(self.configs, 'use_pretrain_in_ft', 0) == 1:
            if self.configs.pretrain_mode == 'noise':
                cond_bc = trend_emb + t_emb
            else:
                cond_bc = trend_emb

            cond_bc = self.cond_proj(cond_bc)
            context_bc = trend_emb.detach()

            feat_bsd = feat.view(B, C, -1, self.d_model).reshape(B * C, -1, self.d_model)
            context_bsd = context_bc.view(B * C, -1, self.d_model)
            cond_bsd = cond_bc.view(B * C, -1, self.d_model)

            denoised_bsd = self.stable_denoiser(
                noisy=x_emb,
                context=context_bsd,
                cond=None,
                key_padding_mask=None
            )
            feat = denoised_bsd + x_emb

        kl_loss = 0.0

        feat = feat.view(B, C, -1, self.d_model)
        seasonal_hat = self.projection[0](feat)

        x_hat_norm = seasonal_hat + trend
        x_clean_norm = seasonal + trend

        if self.configs.pretrain_mode == 'mask' and (M is not None):
            M_float = M.float()
            loss_rec = ((x_hat_norm - x_clean_norm) ** 2 * M_float).sum() / (M_float.sum() + 1e-6)
        elif self.configs.pretrain_mode == 'noise':
            if self.configs.predict_eps == 1:
                loss_rec = F.mse_loss(seasonal_hat, eps)
            else:
                loss_rec = F.mse_loss(x_hat_norm, x_clean_norm)
        else:
            loss_rec = F.mse_loss(x_hat_norm, x)

        total_freq = freq_loss
        total_orth = orth_loss
        total_smooth = smoothness
        total_season_freq = season_freq_loss
        total_recon_loss = recon_loss

        return loss_rec + 0.02 * kl_loss, total_freq, total_orth, total_smooth, total_season_freq, total_recon_loss

    def forecast(self, x, x_mark):
        batch_size, _, num_features = x.size()

        means = torch.mean(x, dim=1, keepdim=True).detach()
        x = x - means
        stdevs = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdevs

        # ===== 统一由辅助函数控制 AASD 是否启用 =====
        seasonal, trend, freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = self._run_decomposition(x)

        seasonal_ci = self.channel_independence[0](seasonal)
        seasonal_ci = self.patch(seasonal_ci)
        seasonal_emb = self.enc_embedding(seasonal_ci)

        if self.configs.use_positional_encoding == 1:
            seasonal_emb_pos = self.positional_encoding(seasonal_emb)
        else:
            seasonal_emb_pos = seasonal_emb

        trend_ci = self.channel_independence[0](trend)
        trend_patch = self.patch(trend_ci)
        trend_emb = self.enc_embedding_trend(trend_patch)

        # ===== 统一由辅助函数控制 TCSR/encoder 是否启用 =====
        if self._use_finetune_encoder_now():
            emb = self.encoder(
                seasonal_emb_pos,
                context=trend_emb,
                is_mask=False,
            )
        else:
            emb = seasonal_emb_pos

        if getattr(self.configs, 'use_film_in_ft', 0) == 1:
            ti = self.channel_independence[0](trend)
            tp = self.patch(ti)
            t_emb = self.enc_embedding_trend(tp)
            if self.configs.use_positional_encoding == 1:
                t_emb = self.positional_encoding(t_emb)

            zeros_t = torch.zeros_like(emb)
            h = torch.cat([zeros_t, t_emb], dim=-1)

            if getattr(self.configs, 'film_mode', 'full') == 'full':
                gamma = 1.0 + 0.1 * torch.tanh(self.cond_to_gamma(h))
                beta = 0.1 * torch.tanh(self.cond_to_beta(h))
            elif self.configs.film_mode == 'none':
                gamma = torch.ones_like(h[..., :self.d_model])
                beta = torch.zeros_like(h[..., :self.d_model])
            elif self.configs.film_mode == 'random':
                gamma = 1.0 + 0.1 * torch.randn_like(h[..., :self.d_model])
                beta = 0.1 * torch.randn_like(h[..., :self.d_model])
            elif self.configs.film_mode == 'trend_only':
                gamma = 1.0 + 0.1 * torch.tanh(self.cond_to_gamma(t_emb))
                beta = 0.1 * torch.tanh(self.cond_to_beta(t_emb))
            elif self.configs.film_mode == 't_only':
                gamma = 1.0 + 0.1 * torch.tanh(self.cond_to_gamma(zeros_t))
                beta = 0.1 * torch.tanh(self.cond_to_beta(zeros_t))

            emb = gamma * emb + beta

        seasonal_enc = emb.reshape(batch_size, num_features, -1, self.d_model)
        seasonal_enc = self.head(seasonal_enc)

        y_enc = seasonal_enc + self.regression[0](trend.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        y_enc = y_enc * (stdevs[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
        y_enc = y_enc + (means[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)

        return y_enc, freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss
    def forward(self, batch_x, x_mask, i=0):
        if self.task_name == "pretrain":
            return self.pretrain(batch_x, x_mask, i)
        elif self.task_name == "finetune":
            dec_out, freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = self.forecast(batch_x, x_mask)
            return dec_out[:, -self.pred_len:, :], freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss
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
