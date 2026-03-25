from scipy.signal import find_peaks
from data_provider.data_factory_draw import data_provider
import torch, numpy as np

from layers.Autoformer_EncDec import moving_avg, series_decomp


from layers.Embed import Patch, PatchEmbedding, PositionalEncoding

import os

from models.TimeDART_draw import plot_line_charts
# TimeDART_version2
# 只保留15分解，用新的分解策略
from utils.masking import generate_causal_mask, generate_self_only_mask
import math
import torch
import torch.nn as nn
from torch.nn import init
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

# ---------------- 全局样式设置 ----------------
FONTSIZE_BASE = 14   # 基础字号
FONTSIZE_LABEL = 14  # 坐标轴标签字号
FONTSIZE_TITLE = 20  # 图标题字号
FONTSIZE_TICK = 14   # 坐标轴刻度字号
FONTSIZE_LEGEND = 14 # 图例字号

# 你原来的数字显示函数保持不变
def format_number(val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return ""
    if abs(val) >= 1e2 or (0 < abs(val) < 0.1):
        return f"{val:.2e}"
    else:
        return f"{val:.2f}"


class WeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4):
        super(WeightGenerator, self).__init__()
        #print('FACTORIZED {}'.format(factorized))
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim
        if self.factorized:
            self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cpu')
            # self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cuda:0')
            self.generator = self.generator = nn.Sequential(*[
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 100)
            ])

            self.mem_dim = 10
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.Q = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
        else:
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True) for _ in range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True) for _ in range(number_of_weights)])
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))
            bias = [torch.matmul(memory, self.B[i]).squeeze(1) for i in range(self.number_of_weights)]
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]
            return weights, bias
        else:
            return self.P, self.B


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

    def forward(self, x, mask,index,configs):
        project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        path = project_path + os.sep + "vis" + os.sep + configs.data + os.sep + configs.task_name + os.sep
        os.makedirs(path, exist_ok=True)
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, att_weight = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(x)
        output = self.norm2(x + self.dropout(ff_output))

        plot_cross_attn_heatmap(att_weight, sample_idx=0,
                                savepath=os.path.join(path, str(index) + '_' + 'encoder_weight.png'),title="Encoder Attention")
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

    def forward(self, x, is_mask=True,configs=None,index=0):
        # x: [batch_size * num_features, seq_len, d_model]
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len).to(x.device) if is_mask else None
        for layer in self.layers:
            x = layer(x, mask,configs=configs,index=index)

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

class DiffusionForComp(nn.Module):
    def __init__(
        self,
        time_steps: int,
        device: torch.device,
        real_scheduler: str = "cosine",
        imag_scheduler: str = "quad",
    ):
        super(DiffusionForComp, self).__init__()
        self.device = device
        self.time_steps = time_steps

        if real_scheduler == "cosine":
            self.real_betas = self._cosine_beta_schedule().to(self.device)
        elif real_scheduler == "linear":
            self.betas = self._linear_beta_schedule().to(self.device)
        elif real_scheduler == "quad":
            self.betas = self._quad_beta_schedule().to(self.device)
        elif real_scheduler == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            self.betas = self._jsd_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Invalid scheduler: {real_scheduler=}")
        self.real_alpha = 1 - self.real_betas
        self.real_gamma = torch.cumprod(self.real_alpha, dim=0).to(self.device)


        if imag_scheduler == "cosine":
            self.imag_betas = self._cosine_beta_schedule().to(self.device)
        elif imag_scheduler == "linear":
            self.imag_betas = self._linear_beta_schedule().to(self.device)
        elif imag_scheduler == "quad":
            self.imag_betas = self._quad_beta_schedule().to(self.device)
        elif imag_scheduler == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            self.imag_betas = self._jsd_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Invalid scheduler: {imag_scheduler=}")
        self.imag_alpha = 1 - self.imag_betas
        self.imag_gamma = torch.cumprod(self.imag_alpha, dim=0).to(self.device)





        # self.betas_real = self._quad_beta_schedule().to(self.device)
        # self.betas_imag = self._jsd_beta_schedule().to(self.device)
        #
        # self.alpha = 1 - self.betas
        # self.alpha_real = 1 - self.betas_real
        # self.alpha_imag = 1 - self.betas_imag
        # self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)
        # # Calculate alpha and gamma for real and imaginary parts separately
        #
        # self.gamma_real = torch.cumprod(self.alpha_real, dim=0).to(self.device)
        # self.gamma_imag = torch.cumprod(self.alpha_imag, dim=0).to(self.device)

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

    def _quad_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, self.time_steps) ** 2
        return betas
    def _jsd_beta_schedule(self):
        betas = 1.0 /torch.linspace(self.time_steps, 1, self.time_steps)
        return betas

    def sample_time_steps(self, shape):
        return torch.randint(0, self.time_steps, shape, device=self.device)

    def noise(self, real,imag, t):
        real_noise = torch.randn_like(real)

        real_gamma_t = self.real_gamma[t].unsqueeze(-1)  # [batch_size * num_features, seq_len, 1]
        # x_t = sqrt(gamma_t) * x + sqrt(1 - gamma_t) * noise
        real_noisy_x = torch.sqrt(real_gamma_t) * real + torch.sqrt(1 - real_gamma_t) * real_noise



        imag_noise = torch.randn_like(imag)

        imag_gamma_t = self.imag_gamma[t].unsqueeze(-1)  # [batch_size * num_features, seq_len, 1]
        # x_t = sqrt(gamma_t) * x + sqrt(1 - gamma_t) * noise
        imag_noisy_x = torch.sqrt(imag_gamma_t) * imag + torch.sqrt(1 - imag_gamma_t) * imag_noise

        return real_noisy_x, real_noise,imag_noisy_x,imag_noise




    def forward(self, real,imag):
        # x: [batch_size * num_features, seq_len, patch_len]
        t = self.sample_time_steps(real.shape[:2])  # [batch_size * num_features, seq_len]
        real_noisy_x, real_noise,imag_noisy_x, imag_noise = self.noise(real=real,imag=imag, t=t)
        return real_noisy_x, real_noise,imag_noisy_x, imag_noise, t


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerDecoderBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, tgt_mask, src_mask):
        """
        :param query: [batch_size * num_features, seq_len, d_model]
        :param key: [batch_size * num_features, seq_len, d_model]
        :param value: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.self_attention(query, query, query, attn_mask=tgt_mask)
        query = self.norm1(query + self.dropout(attn_output))

        # Encoder attention
        attn_output, _ = self.encoder_attention(query, key, value, attn_mask=src_mask)
        query = self.norm2(query + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(query)
        x = self.norm3(query + self.dropout(ff_output))

        return x


class DenoisingPatchDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super(DenoisingPatchDecoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, is_tgt_mask=True, is_src_mask=True):
        seq_len = query.size(1)
        tgt_mask = (
            generate_self_only_mask(seq_len).to(query.device) if is_tgt_mask else None
        )
        src_mask = (
            generate_self_only_mask(seq_len).to(query.device) if is_src_mask else None
        )
        for layer in self.layers:
            query = layer(query, key, value, tgt_mask, src_mask)
        x = self.norm(query)
        return x



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
def calculate_scales_optimized_toech_for_fig(data, init_conv_kernel=[15, 31, 63, 95], num_scales=3, max_lag=63, peak_threshold=0.3, distance=10):
    # 数据预处理 [B, L, C] -> [B, C, L]
    x = data.permute(0, 2, 1).contiguous()
    B, C, L = x.shape
    max_lag = min(max_lag, L - 1)
    # 批标准化
    x_mean = x.mean(dim=2, keepdim=True)
    x_centered = x - x_mean
    x_norm = x_centered / (x_centered.std(dim=2, keepdim=True) + 1e-8)

    # FFT加速自相关计算（优化填充尺寸）
    pad_size = L - 1
    x_padded = torch.nn.functional.pad(x_norm, (0, pad_size))
    fft_x = torch.fft.rfft(x_padded, dim=2)
    acf = torch.fft.irfft(fft_x * fft_x.conj(), dim=2)[..., :L]
    acf = acf / (acf[..., :1] + 1e-8)

    # 聚合所有通道和批次
    mean_acf = acf.mean(dim=(0, 1))  # [L]

    # GPU峰值检测
    peaks = find_peaks_torch(
        mean_acf[1:max_lag],
        height=peak_threshold,
        distance=distance,
        max_num=num_scales * 2
    )
    peaks += 1  # 滞后值修正

    # 选择主要尺度
    if len(peaks) == 0:
        last_scales = init_conv_kernel[:num_scales]
    else:
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
            while len(last_scales) < num_scales:
                last_scales.append(default_scales[-1])

    # ====== 裁剪到 ACF 范围内 ======
    max_valid_idx = len(mean_acf) - 1
    last_scales = [s for s in last_scales if s <= max_valid_idx]
    peaks = peaks[peaks <= max_valid_idx]

    return mean_acf, sorted(last_scales[:num_scales]), peaks

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
    def __init__(self, nvar,num_scales,max_lag,peak_threshold,distance,configs):
        super().__init__()
        self.nvar = nvar
        # 延迟初始化的组件
        self.peak_threshold = peak_threshold
        self.distance =distance
        self.fixed_convs = None  # 将在第一次forward时初始化
        self.scales = None  # 保存计算得到的scales
        self.num_scales = num_scales
        self.max_lag = max_lag
        self.configs = configs
        # 多尺度卷积组（固定参数）
        # self.fixed_convs = FixedMultiScaleConv(nvar, scales)

        # 轻量权重生成器
        self.weight_gen = LightWeightGenerator(nvar, self.num_scales)
        # self.weight_gen = None
        # self.local_encoder = nn.Conv1d(nvar, nvar, 3, padding=1, groups=nvar)
    def forward(self, x,i):

        project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        path = project_path + os.sep + "vis" + os.sep + configs.data + os.sep + configs.task_name + os.sep
        os.makedirs(path, exist_ok=True)



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


        # ========================== visio for peaks and scales ==========================


        import torch.nn.functional as F
        import matplotlib.pyplot as plt

        def plot_acf_with_scales(mean_acf, peaks, scales, save_path=None,
                                 title="Autocorrelation, Peaks, and Selected Scales"):
            """
            绘制完整ACF、自相关峰值以及最终选择的尺度

            参数：
                mean_acf : 1D array-like，自相关曲线 (长度 L)
                peaks    : 1D array-like，检测到的局部峰值位置
                scales   : 1D array-like，最终选择的主要尺度
                save_path: str，可选，保存路径 (如果为None则不保存)
                title    : str，图标题
            """
            # 转 numpy
            mean_acf = np.array(mean_acf.detach().numpy())
            peaks = np.array(peaks.detach().numpy())
            scales = np.array(scales)

            plt.figure(figsize=(7, 4))
            # 画ACF
            plt.plot(mean_acf, color='blue', linewidth=2, label="Mean ACF")
            # 画所有峰值
            if len(peaks) > 0:
                plt.scatter(peaks, mean_acf[peaks], color='red', s=50, zorder=5, label="Detected Peaks")
            # 画最终选择的尺度
            if len(scales) > 0:
                plt.scatter(scales, mean_acf[scales], color='green', s=100, marker='*', zorder=6,
                            label="Selected Scales")

            # 美化
            plt.xlabel("Lag")
            plt.ylabel("ACF")
            plt.title(title)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            # 保存或显示
            if save_path:
                plt.savefig(save_path , dpi=300)

        out = calculate_scales_optimized_toech_for_fig(
            x,
            num_scales=self.num_scales,
            distance=self.distance,
            peak_threshold=self.peak_threshold,
            max_lag=self.max_lag,
            # peak_threshold=self.peak_threshold
        )


        if len(out) == 3:
            mean_acf, scales, peaks = out


            plot_acf_with_scales(mean_acf,peaks,scales,save_path=os.path.join(path , str(i) + '_'+'_acf.png'))

        else:
            scales, peaks = out

        # ======== 绘图 ========
        # if len(peaks) != 0:
        #     plt.figure(figsize=(6, 4))
        #     scales = torch.tensor(scales) if isinstance(scales, list) else scales
        #     peaks = torch.tensor(peaks) if isinstance(peaks, list) else peaks
        #
        #     plt.plot(scales.numpy(), label="Mean ACF")
        #     plt.scatter(peaks.numpy(), scales.numpy(), color='red', zorder=5, label="Detected Peaks")
        #     plt.xlabel("Lag")
        #     plt.ylabel("ACF")
        #     plt.title("Autocorrelation & Detected Peaks ")
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(path + str(i) + '_peaks.png')
        #
        #     # ======== 绘制直方图 ========
        #     plt.figure(figsize=(6, 4))
        #     plt.hist(peaks.numpy(), bins=np.arange(0, 64, 2), color="gray", edgecolor="black")
        #     plt.title("Peak Lag Distribution")
        #     plt.xlabel("Lag (bin)")
        #     plt.ylabel("Count")
        #     plt.tight_layout()
        #     plt.savefig(path + str(i) + '_Lag.png')

        # ========================== visio for peaks and scales ==========================



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



        # -------------------------------------------  绘制 heatmap for weight------------------


        # ======== 获取权重 ========
        weights_mean = weights.squeeze(-1).mean(dim=0)
        weights_mean  = weights_mean.unsqueeze(0)  # [C,K]
        plot_cross_attn_heatmap(weights_mean, sample_idx=0, savepath=os.path.join(path, str(i) +'_'+'cross_attn_heatmap1_AASD.png'),title="AASD Weight" )
        # # ======== 自动生成横轴（1,2,3,4） ========
        # xticks = [f"S{i + 1}" for i in range(weights_mean.shape[1])]
        # yticks = [f"Var{i}" for i in range(weights_mean.shape[0])]
        #
        # # ======== 绘制紧凑型 heatmap ========
        # plt.figure(figsize=(1 + weights_mean.shape[1], 0.8 + weights_mean.shape[0]))
        # ax = sns.heatmap(weights_mean, annot=False, cmap="Blues",
        #             xticklabels=xticks, yticklabels=yticks,
        #             cbar_kws={'label': 'Weight'}, square=False)
        # h, w = weights_mean.shape
        # ax.set_aspect(w / h)  # 关键：让整个热图是正方形
        #
        # # 反转 y 轴，让 Var0 在最上面
        # plt.gca().invert_yaxis()
        #
        # plt.title("Scale Weights (averaged over batch)")
        # plt.xlabel("Scales (Index)")
        # plt.ylabel("Variables")
        # plt.tight_layout()
        # plt.savefig(path + str(i) + '_heatmap.png')

        # -------------------------------------------  绘制 heatmap for weight------------------









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



        # ------------------------------- 绘制 frequency map

        sample_idx = 0
        var_idx = 0
        # ======== 选同一个样本 & 变量 ========
        trend_fft = torch.fft.rfft(fused_trend[sample_idx, var_idx, :]).abs().detach().numpy()
        seasonal_fft = torch.fft.rfft(seasonal[sample_idx, var_idx, :]).abs().detach().numpy()
        freqs = np.fft.rfftfreq(L, d=1)  # 频率刻度

        # ======== 绘图 ========
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, trend_fft, label="Trend FFT", linewidth=2)
        plt.plot(freqs, seasonal_fft, label="Seasonal FFT", linestyle="--")
        plt.title(f"Frequency Spectrum (Sample {sample_idx}, Var {var_idx})")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + str(i)+'_' + self.configs.task_name +  self.configs.data + str(self.configs.pred_len) +'_frequency.png')

        # ------------------------------- 绘制 frequency map
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
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2*d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2*d_model, d_model), nn.Dropout(dropout)
        )
        if use_film == 1:
            self.film = nn.Sequential(
                nn.Linear(2*d_model, d_model), nn.SiLU(),
                nn.Linear(d_model, 2*d_model)  # -> [gamma, beta]
            )
        self.ln_out = nn.LayerNorm(d_model)

    @staticmethod
    def _orth_residual(x, delta, eps=1e-6):
        # 从 delta 中去掉在 x 方向的投影，避免“复制输入”的退化
        # 逐 token 做：proj = <delta,x>/<x,x> * x
        num = (delta * x).sum(-1, keepdim=True)
        den = (x * x).sum(-1, keepdim=True) + eps
        proj = num / den * x
        return delta - proj

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().float().cpu().numpy()

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.ticker import MaxNLocator

    def viz_film_scatter(
            self,
            gamma: torch.Tensor,  # [B,S,D]
            beta: torch.Tensor,  # [B,S,D]
            savepath: str = "FiLM_scatter.png",
            weak_thr: float = 0.02,  # 仅在 show_weak_box=True 时使用
            dpi: int = 300,
            hexbin_cutover: int = 20000,  # 点数超过阈值用 hexbin

            # —— 可读性/诊断开关（默认全部关闭：极简风格）——
            show_weak_box: bool = False,
            show_zero_axes: bool = False,
            show_stats_box: bool = False,
            show_colorbar: bool = False  # 仅 hexbin 有效
    ):
        """
        极简版 FiLM γ–β 散点/蜂窝图（期刊友好）。
        - 默认只绘制点/密度 + 轴标签/标题。
        - 需要时可打开诊断元素（弱调制框、零轴、统计角标、colorbar）。
        """
        # 有效调制
        g_eff = 0.1 * torch.tanh(gamma)
        b_eff = 0.1 * torch.tanh(beta)
        gx = self._to_numpy(g_eff).ravel()
        by = self._to_numpy(b_eff).ravel()

        # 统计（可用于角标）
        if gx.size > 1 and by.size > 1:
            r = float(np.corrcoef(gx, by)[0, 1])
        else:
            r = np.nan
        gm, gs = float(np.mean(gx)), float(np.std(gx))
        bm, bs = float(np.mean(by)), float(np.std(by))

        # 画图
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)

        # 大样本用 hexbin；否则散点
        cbar = None
        if gx.size > hexbin_cutover:
            hb = ax.hexbin(gx, by, gridsize=55, mincnt=1, cmap='viridis')
            if show_colorbar:
                cbar = plt.colorbar(hb, ax=ax)
                cbar.set_label("count", fontsize=FONTSIZE_LABEL)
                cbar.ax.tick_params(labelsize=FONTSIZE_TICK)
        else:
            ax.scatter(gx, by, s=5, alpha=0.25, color='#1f77b4', edgecolors='none')

        # 可选：零轴
        if show_zero_axes:
            ax.axhline(0.0, color='k', lw=0.9)
            ax.axvline(0.0, color='k', lw=0.9)

        # 可选：弱调制理想区域
        if show_weak_box:
            ax.axhspan(-weak_thr, weak_thr, xmin=0.0, xmax=1.0,
                       color='tab:green', alpha=0.10)
            ax.axvspan(-weak_thr, weak_thr, ymin=0.0, ymax=1.0,
                       color='tab:green', alpha=0.10)
            ax.plot([-weak_thr, -weak_thr], [-weak_thr, weak_thr], color='tab:green', lw=1.0, ls='--')
            ax.plot([weak_thr, weak_thr], [-weak_thr, weak_thr], color='tab:green', lw=1.0, ls='--')
            ax.plot([-weak_thr, weak_thr], [-weak_thr, -weak_thr], color='tab:green', lw=1.0, ls='--')
            ax.plot([-weak_thr, weak_thr], [weak_thr, weak_thr], color='tab:green', lw=1.0, ls='--')

        # 可选：角标统计
        if show_stats_box:
            txt = (f"Pearson r = {r: .3f}\n"
                   f"γ_eff: μ={gm: .3e}, σ={gs: .3e}\n"
                   f"β_eff: μ={bm: .3e}, σ={bs: .3e}")
            ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                    va='top', ha='left', fontsize=FONTSIZE_BASE,
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

        # 轴/标题/刻度（与你的全局字体策略一致）
        ax.set_xlabel(r"effective $\gamma$ = $0.1\,\tanh(\gamma)$", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(r"effective $\beta$  = $0.1\,\tanh(\beta)$", fontsize=FONTSIZE_LABEL)
        ax.set_title("FiLM γ–β correlation", fontsize=FONTSIZE_TITLE)
        ax.tick_params(axis='x', labelsize=FONTSIZE_TICK, direction='in')
        ax.tick_params(axis='y', labelsize=FONTSIZE_TICK, direction='in')
        ax.xaxis.set_major_locator(MaxNLocator(nbins='auto'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins='auto'))

        plt.tight_layout()
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    def viz_film_bar_per_feature(
            self,
            gamma: torch.Tensor,  # [B,S,D]
            savepath: str = "FiLM_avg_gamma_per_feature.png",
            weak_thr: float = 0.02,  # 期望的“理想上限线”（可调）
            warn_thr: float = 0.05,  # 告警线（超过说明调制偏强）
            topk: int = 8,
            dpi: int = 220
    ):
        """
        画“按特征维度聚合”的平均 |γ_eff| 柱状图：
          v[d] = mean_{B,S} | 0.1*tanh(γ_{b,s,d}) |
        并画两条参考线：理想上限 & 告警线，且高亮 Top-K 维度。
        """
        g_eff = 0.1 * torch.tanh(gamma)  # [B,S,D]
        vals_t = g_eff.detach().abs().mean(dim=(0, 1))  # [D]
        vals = self._to_numpy(vals_t)  # -> np.ndarray [D]

        D = vals.shape[0]
        idx = np.arange(D)
        # Top-K 维度
        topk = min(topk, D)
        top_idx = np.argsort(vals)[-topk:][::-1]

        # 画图
        plt.figure(figsize=(6.6, 3.6), dpi=dpi)
        # 先画全体
        plt.bar(idx, vals, color='#9ecae1', edgecolor='none')
        # 高亮 Top-K
        plt.bar(top_idx, vals[top_idx], color='#e34a33', edgecolor='none', label=f'Top-{topk}')

        # 参考线：理想上限 & 告警线
        plt.axhline(weak_thr, color='tab:green', ls='--', lw=1.2, label=f'weak-thr={weak_thr}')
        plt.axhline(warn_thr, color='tab:red', ls='--', lw=1.2, label=f'warn-thr={warn_thr}')

        # 给 Top-K 标注数值
        for i in top_idx:
            plt.text(i, vals[i] + 0.002, f"{vals[i]:.3f}", ha='center', va='bottom', fontsize=8, rotation=90)

        plt.xlabel('Feature dimension (D)')
        plt.ylabel(r'$\mathbb{E}_{B,S}\left[\,|0.1\,\tanh(\gamma)|\,\right]$')
        plt.title('Average effective |γ| per feature (lower is better)')
        plt.legend(loc='upper right', framealpha=0.75)
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()
    def forward(self, noisy, context, cond=None, key_padding_mask=None,configs=None,index=0):
        """
        noisy:   [B, S, D]  待去噪的序列（掩码重建或扩散噪声）
        context: [B, S, D]  供检索的上下文（如季节/趋势/clean encoder）
        cond:    [B, S, D]  额外条件（如趋势嵌入、t-embedding），可为 None
        """
        q = self.ln_q(noisy)
        kv = self.ln_kv(context)
        # self.viz_prefix = "vis" + os.sep + configs.data+os.sep+configs.task_name+os.sep  # 你自己的输出目录
        project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        path = project_path + os.sep + "vis" + os.sep + configs.data + os.sep + configs.task_name + os.sep
        os.makedirs(path, exist_ok=True)

        # (1) 可选 FiLM：只对 q 做仿射，不把 cond 与 noisy 直接拼接/相加
        if self.use_film == 1 and cond is not None:
            h = torch.cat([q, cond], dim=-1)
            gamma_beta = self.film(h)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            q = (1.0 + 0.1 * torch.tanh(gamma)) * q + 0.1 * torch.tanh(beta)
            # os.makedirs(self.viz_prefix, exist_ok=True)
            self.viz_film_scatter(gamma, beta, savepath=os.path.join(path, str(index) +'_'+'FiLM_scatter.png'),show_zero_axes=True,show_weak_box=True,show_stats_box=True)

        # (2) 跨注意力：Q=noisy(经LN), K=V=context；确保 K/V 是“有结构的表征”，不是标量门控
        attn_out, attn_w = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=True)
        # attn_w_3d: [B, L, L]
        # asi = plot_cross_attn_heatmap(attn_w, sample_idx=0, savepath='cross_attn_heatmap.png')
        plot_cross_attn_heatmap(attn_w, sample_idx=0, savepath=os.path.join(path, str(index) +'_'+'cross_attn_heatmap1_in_TCSR.png') ,title="TCSR Attention")
        # (3) 正交化残差 + FFN
        delta = self._orth_residual(noisy, attn_out)         # 去掉与输入共线的部分
        y = noisy + delta                                    # 残差1
        y = y + self.ff(self.ln_out(y))                      # 残差2（前归一化）

        return y

# 绘制热图
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_cross_attn_heatmap(
    attn_w: torch.Tensor,
    sample_idx: int = 0,
    head_reduce: str = "mean",      # 对 [B,H,L,L] 时有效: 'mean' | 'max'
    row_normalize: bool = True,     # 是否对每一行做归一化，突出“查询→键”的分布
    clip_percentiles=(1, 99),       # 分位裁剪增强对比
    show_diag: bool = True,         # 画对角基准线
    title: str = "TCSR Cross-Attn",
    savepath: str = "heatmap.png",
    dpi: int = 200
):
    """
    支持输入:
      - [B, L, L]
      - [B, H, L, L] (会先按 head_reduce 聚合到 [B, L, L])

    返回:
      asi: Attention Shift Index (上三角平均 - 下三角平均)，>0 代表上三角偏高，<0 代表下三角偏高
    """
    if not torch.is_tensor(attn_w):
        raise TypeError("attn_w must be a torch.Tensor")

    A = attn_w
    if A.dim() == 4:
        # [B,H,L,L] -> [B,L,L]
        if head_reduce == "mean":
            A = A.mean(dim=1)
        elif head_reduce == "max":
            A = A.max(dim=1).values
        else:
            raise ValueError("head_reduce must be 'mean' or 'max' for 4D attention")
    elif A.dim() == 3:
        # [B,L,L] 直接用
        pass
    else:
        raise ValueError(f"Unexpected attn_w shape {tuple(A.shape)}; expected [B,L,L] or [B,H,L,L].")

    # 取样本
    if sample_idx < 0 or sample_idx >= A.shape[0]:
        raise IndexError(f"sample_idx out of range: got {sample_idx}, but batch size is {A.shape[0]}")
    A = A[sample_idx].detach().float().cpu().numpy()  # [L, L]

    # 可选：按行归一化，让每个 Query 的分布可比
    if row_normalize:
        row_sum = A.sum(axis=1, keepdims=True) + 1e-12
        A = A / row_sum

    # 分位裁剪，增强对比同时稳健处理极值
    lo, hi = np.percentile(A, clip_percentiles)
    A = np.clip(A, lo, hi)

    # 计算注意力偏移指数 ASI（上三角 - 下三角）
    # 只统计非对角线的上/下三角
    L = A.shape[0]
    iu = np.triu_indices(L, k=1)  # 上三角（不含对角）
    il = np.tril_indices(L, k=-1) # 下三角（不含对角）
    up_mean = A[iu].mean() if A[iu].size > 0 else 0.0
    lo_mean = A[il].mean() if A[il].size > 0 else 0.0
    asi = float(up_mean - lo_mean)  # >0: 上三角更强；<0: 下三角更强

    # 绘图
    plt.figure(figsize=(5.2, 4.2), dpi=dpi)
    im = plt.imshow(A, aspect='auto', origin='lower')  # cmap 默认即可
    cbar = plt.colorbar(im)
    cbar.set_label('Attention weight', rotation=90)

    if show_diag:
        # 画对角线参考
        xs = np.arange(L)
        plt.plot(xs, xs, linewidth=0.8)

    plt.xlabel('Key (trend/context)')
    plt.ylabel('Query (season/noisy)')
    plt.title(f"{title}\nASI={asi:+.3f}  (upper-lower)")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

    return asi
# === 通用字体与样式控制 ===
plt.rcParams.update({
    "font.size": FONTSIZE_BASE,
    "axes.titlesize": FONTSIZE_TITLE,
    "axes.labelsize": FONTSIZE_LABEL,
    "xtick.labelsize": FONTSIZE_TICK,
    "ytick.labelsize": FONTSIZE_TICK,
    "legend.fontsize": FONTSIZE_LEGEND,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "figure.dpi": 200
})

# ========== 1. TCSR Cross-Attention Heatmap ==========
from matplotlib.ticker import MaxNLocator, MultipleLocator

def plot_cross_attn_heatmap(
        attn_w,
        sample_idx=0,
        head_reduce='mean',
        title='TCSR Cross-Attn',
        savepath: str = "heatmap.png",
        max_ticks: int = 6  # 控制坐标轴最多显示多少个刻度
):
    """
    绘制 TCSR 的交叉注意力热力图。
    attn_w: [B, n_heads, S_q, S_k]
    """
    A = attn_w
    if head_reduce == 'mean':
        A = A.mean(0)
    elif head_reduce == 'max':
        A = A.max(0).values
    A = A.detach().cpu().numpy()

    # 稳健裁剪
    lo, hi = np.percentile(A, [1, 99])
    A = np.clip(A, lo, hi)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A, aspect='auto', origin='lower', cmap='viridis')

    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention weight', fontsize=FONTSIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICK)

    # 轴标签
    ax.set_xlabel('Key (trend/context)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Query (season/noisy)', fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE)

    # 让刻度更稀疏、可读
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, integer=True))

    # 或者固定每隔若干步（例如每 5 或 10）
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    # ax.yaxis.set_major_locator(MultipleLocator(10))

    ax.tick_params(axis='x', labelsize=FONTSIZE_TICK)
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


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
        self.decomp_multi_learnable = StopLearnableMultiScaleDecomp(self.configs.c_out,max_lag=self.configs.max_lag,num_scales=self.configs.num_scales,peak_threshold=self.configs.peak_threshold,distance=10,configs=configs)

        self.denoise_layers_num = configs.denoise_layers_num

        # self.denoise_layers_cond = nn.ModuleList([
        #     DenoisingConditionDecoder(
        #         embed_dim=configs.d_model,
        #         num_heads=configs.n_heads,
        #         dropout=configs.dropout,
        #     )
        #     for _ in range(self.denoise_layers_num)
        # ])
        self.denoise_layers_cond = nn.ModuleList([
            StableCondDenoiser(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                dropout=configs.dropout,
                use_film=self.configs.use_film,
            )
            for _ in range(self.denoise_layers_num)
        ])

        # 在 __init__ 尾部、其它属性旁边
        self.pretrain_noise = getattr(configs, 'pretrain_noise', 'mask')  # 'mask' 或 'tembed'
        self.mask_block = getattr(configs, 'lm', 12)
        self.mask_ratio = getattr(configs, 'mask_rate', 0.5)
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
        self.enc_embedding_trend = PatchEmbedding(
            patch_len=self.patch_len,
            d_model=self.d_model,
        )
        self.stable_denoiser = StableCondDenoiser(d_model=self.d_model, n_heads=self.num_heads, dropout=self.dropout,
                                                  use_film=True)
        self.cond_proj = nn.Linear(self.d_model, self.d_model)  # 轻量投影，避免尺度打架


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

    def pretrain(self, x,x_mask,i=0):
        project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        path = project_path + os.sep + "vis" + os.sep + self.configs.data + os.sep + self.configs.task_name + os.sep
        os.makedirs(path, exist_ok=True)

        # [batch_size, input_len, num_features]
        # Instance Normalization

        # x = torch.fft.fft(x,dim=-2).real
        mask_rate = 0.5
        lm=3
        positive_nums=1
        e_x =x
        device = x.device
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
        B, L, C = x.shape  # 此时 x 是 seasonal，trend 是趋势

        list_ts = []
        list_ts_mov = []
        list_ts.append(x)
        list_ts_mov.append(x)


        seasonal, trend = self.decomp_multi(x)
        freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = torch.tensor(0.0), torch.tensor(
            0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        list_ts_mov.append(seasonal)
        list_ts_mov.append(trend)

        if self.configs.use_new_decomp == 1:
            seasonal ,trend,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss = self.decomp_multi_learnable(x,i)
            list_ts.append(seasonal)
            list_ts.append(trend)
        else:
            seasonal, trend = self.decomp_multi(x)
            freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0),torch.tensor(0.0)

        plot_line_charts(list_ts, ['source', 'season', 'trend'], path=path,
                         name=str(i) + '_draw_source_dynamic_fig_')
        plot_line_charts(list_ts_mov, ['source', 'season', 'trend'], path=path,
                         name=str(i) + '_draw_source_mov_fig_')


        if self.configs.pretrain_mode == 'mask':
            if self.configs.use_geo_mask:
                # 在 pretrain_B 内部、拿到 seasonal/trend 之后：
                M = geometric_block_mask_torch(
                    B, L, C,
                    masking_ratio=self.configs.mask_ratio,
                    lm=self.configs.mask_block,
                    shared_channels=True,
                    device=x.device
                )  # [B,L,C], True=遮盖
                if self.configs.destroy_mode == 'season':
                    seasonal_tilde = torch.where(M, torch.zeros_like(seasonal), seasonal)  # 被遮盖处=0
                    x_tilde = trend + seasonal_tilde
                elif self.configs.destroy_mode == 'trend':
                    trend_tilde = torch.where(M, torch.zeros_like(trend), trend)  # 被遮盖处=0
                    x_tilde = trend_tilde + seasonal
                elif self.configs.destroy_mode == 'x':
                    x_tilde = torch.where(M, torch.zeros_like(x), x)  # 被遮盖处=0
            else:
                M = block_mask_torch(B, L, C, masking_ratio=self.configs.mask_ratio, block=self.configs.mask_block,
                                     shared_channels=True, variable_block=True, device=x.device)
                if self.configs.destroy_mode == 'season':

                    # c = M.mean()
                    seasonal_tilde = seasonal.masked_fill(M, 0.0)
                    # d = seasonal_tilde.mean()
                    x_tilde = trend + seasonal_tilde
                elif self.configs.destroy_mode == 'trend':
                    trend_tilde = trend.masked_fill(M, 0.0)
                    x_tilde = trend_tilde + seasonal
                elif self.configs.destroy_mode == 'x':
                    x_tilde = x.masked_fill(M, 0.0)
        elif self.configs.pretrain_mode == 'noise':
            # -------- 对季节项加扩散噪声（时间域）--------
            # 采样一个 t（也可 token 级），这里用样本级标量 t
            t = self.diffusion.sample_time_steps((B,))  # [B]
            # 广播到 [B,L,C]
            gamma_t = self.diffusion.gamma[t].view(B, 1, 1).to(device)
            if self.configs.destroy_mode == 'season':

                eps = torch.randn_like(seasonal)
                noisy_seasonal = torch.sqrt(gamma_t) * seasonal + torch.sqrt(1 - gamma_t) * eps
                x_tilde = trend + noisy_seasonal  # 只对 seasonal 加噪，趋势保真
            elif self.configs.destroy_mode == 'trend':
                eps = torch.randn_like(trend)
                noisy_trend = torch.sqrt(gamma_t) * trend + torch.sqrt(1 - gamma_t) * eps
                x_tilde = seasonal + noisy_trend  # 只对 seasonal 加噪，趋势保真
            elif self.configs.destroy_mode == 'x':
                eps = torch.randn_like(x)
                x_tilde = torch.sqrt(gamma_t) * trend + torch.sqrt(1 - gamma_t) * eps

        else:
            x_tilde = seasonal+trend
        # -------- 编码为 patch 表征 --------
        x_ci = self.channel_independence[0](x_tilde)  # [B*C, L, 1]
        x_patch = self.patch(x_ci)  # [B*C, S, P]
        x_emb = self.enc_embedding(x_patch)  # [B*C, S, D]



        if self.configs.use_sostoken == 1:
            x_embedding_bias = self.add_sos_token_and_drop_last(
                x_emb
            )  # [batch_size * num_features, seq_len, d_model]
        else:

            x_embedding_bias=x_emb
        if self.configs.use_positional_encoding == 1:

            x_embedding_bias = self.positional_encoding(x_embedding_bias)

        else:
            x_embedding_bias = x_embedding_bias
        # 分解  2
        # x_embedding_bias, _ = self.decomp_multi(x_embedding_bias)
        # x_embedding_bias, _ = x_embedding_bias,x_embedding_
        if self.configs.use_pretrain_encoder == 1:
            x_emb  = self.encoder(
                x_embedding_bias,
                is_mask=False,
                configs=self.configs, index=i
            )  # [batch_size * num_features, seq_len, d_model]
        else:
            x_emb=x_embedding_bias
        # 获取总去噪层数和扩散模型总时间步



        # -------- trend 条件编码（时间域 -> D，广播到序列）--------

        trend_ci = self.channel_independence[0](trend)  # [B*C, L, 1]
        trend_patch = self.patch(trend_ci)  # [B*C, S, P]
        trend_emb = self.enc_embedding_trend(trend_patch)

        cond_trend = trend_emb  # [B, 1, D]

        # -------- t-embedding（若启用）--------
        if self.configs.pretrain_mode == 'noise':

            if self.t_embed is not None:
                # 把样本级 t 展开到序列
                t_big = t.view(B, 1).repeat(1, x_emb.size(1))  # [B, S]
                t_big = t_big.repeat_interleave(C, dim=0)  # [B*C, S]
                t_emb = self.t_embed(t_big)  # [B*C, S, D]
            else:
                t_emb = torch.zeros_like(x_emb)
        else:
            t_emb = torch.zeros_like(x_emb)

        # -------- FiLM 条件 --------
        # h = torch.cat([t_emb, cond_trend], dim=-1)  # [B*C, S, 2D]
        # gamma = 1.0 + 0.1 * torch.tanh(self.cond_to_gamma(h))
        # beta = 0.1 * torch.tanh(self.cond_to_beta(h))

        # -------- FiLM 条件 --------



        # 多层残差去噪
        feat = x_emb
        # if getattr(self.configs, 'use_pretrain_in_ft', 0) == 1:
        if self.configs.pretrain_mode == 'noise':

            # 还原 B 和 C
            B_times_C, S, D = feat.shape
            B, C = B, C  # 已在上文得到
            # 1) 组装 cond（掩码重建：trend；扩散：trend + t）
            if self.configs.pretrain_mode == 'noise':
                cond_bc = trend_emb + t_emb  # [B*C, S, D]
            else:
                cond_bc = trend_emb  # [B*C, S, D]
            cond_bc = self.cond_proj(cond_bc)  # 轻量对齐到同尺度

            # 2) 选择 context（建议先用 trend_emb，当先验；可尝试 .detach() 更稳）
            context_bc = trend_emb.detach()  # [B*C, S, D] 先验，不回传梯度

            # 3) reshape 到 [B, S, D]，把通道当“batch 维中的子批次”
            feat_bsd = feat.view(B, C * S, D)  # 合并通道到序列会破坏时序，不建议
            # 正确方式：把通道“并回 batch”，对每个变量独立做注意力
            feat_bsd = feat.view(B, C, S, D).reshape(B * C, S, D)  # 仍是 [B*C, S, D]
            context_bsd = context_bc.view(B * C, S, D)
            cond_bsd = cond_bc.view(B * C, S, D)

            # 4) 稳定条件去噪（共享骨干）
            denoised_bsd = self.stable_denoiser(
                noisy=feat_bsd,  # 被掩或加噪后的表征
                context=context_bsd,  # 结构化先验（趋势或干净表征）
                cond=cond_bsd,  # 条件：trend (+ t)
                key_padding_mask=None,
                configs=self.configs,
                index=i
            )  # [B*C, S, D]

            feat = denoised_bsd  # 返回到 [B*C, S, D]

        # --- 投影：直接回到 seasonal_hat（推荐）---
        feat = feat.view(B, C, -1, self.d_model)  # [B, C, S, D]
        seasonal_hat = self.projection[0](feat)  # [B, L, C]

        # --- 只加一次 trend，得到 x_hat（归一化域）---
        x_hat_norm = seasonal_hat + trend  # [B, L, C]
        x_clean_norm = seasonal + trend  # 就是标准化后的 x

        # ---------- 预训练损失 ----------
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
        total_freq, total_orth, total_smooth, total_season_freq, total_recon_loss = freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss
        return loss_rec,total_freq,total_orth,total_smooth,total_season_freq,total_recon_loss

    def forecast(self, x, x_mark,i):
        # x = torch.fft.fft(x,dim=-2).real
        project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        path = project_path + os.sep + "vis" + os.sep + self.configs.data + os.sep + self.configs.task_name + os.sep
        os.makedirs(path, exist_ok=True)

        batch_size, _, num_features = x.size()
        means = torch.mean(x, dim=1, keepdim=True).detach()
        x = x - means
        stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / stdevs
        # x, trend = self.decomp_multi(x)
        # x = self.inverse_embedding(x.permute(0,2,1)).permute(0,2,1)
        list_ts = []
        list_ts_mov = []
        list_ts.append(x)
        list_ts_mov.append(x)
        seasonal, trend = self.decomp_multi(x)
        list_ts_mov.append(seasonal)
        list_ts_mov.append(trend)
        if self.configs.use_new_decomp == 1:
            seasonal, trend, freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = self.decomp_multi_learnable(x,i)
            list_ts.append(seasonal)
            list_ts.append(trend)
        else:
            seasonal, trend = self.decomp_multi(x)
            freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = torch.tensor(0.0), torch.tensor(
                0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        plot_line_charts(list_ts, ['source', 'season', 'trend'], path=path,
                         name=str(i) + '_draw_source_dynamic_fig_')
        plot_line_charts(list_ts_mov, ['source', 'season', 'trend'], path=path,
                         name=str(i) + '_draw_source_mov_fig_')

        # x, trend = x,x
        seasonal_ci = self.channel_independence[0](seasonal)  # [batch_size * num_features, input_len, 1]
        seasonal_ci = self.patch(seasonal_ci)  # [batch_size * num_features, seq_len, patch_len]

        seasonal_emb = self.enc_embedding(seasonal_ci)  # [batch_size * num_features, seq_len, d_model]

        if self.configs.use_positional_encoding == 1:
            seasonal_emb_pos = self.positional_encoding(seasonal_emb)  # [batch_size * num_features, seq_len, d_model]
        else:
            seasonal_emb_pos = seasonal_emb


        if self.configs.use_finetune_encoder == 1:
            emb = self.encoder(
                seasonal_emb_pos,
                is_mask=False,
                configs=self.configs, index=i
            )  # [batch_size * num_features, seq_len, d_model]
        else:
            emb=seasonal_emb_pos
        # 可选：FiLM
        if getattr(self.configs, 'use_film_in_ft', 0) == 1:
            ti = self.channel_independence[0](trend)
            tp = self.patch(ti)
            t_emb = self.enc_embedding_trend(tp)
            if self.configs.use_positional_encoding == 1:
                t_emb = self.positional_encoding(t_emb)
            zeros_t = torch.zeros_like(emb)



            # h = torch.cat([zeros_t, t_emb], -1)
            # gamma = 1.0 + 0.1 * torch.tanh(self.cond_to_gamma(h))
            # beta = 0.1 * torch.tanh(self.cond_to_beta(h))
            # emb = gamma * emb + beta
            # -------- FiLM 条件 --------
            h = torch.cat([zeros_t, t_emb], dim=-1)  # [B*C, S, 2D]

            if getattr(self.configs, 'film_mode', 'full') == 'full':
                # 原版：trend + t 都参与
                gamma = 1.0 + 0.1 * torch.tanh(self.cond_to_gamma(h))
                beta = 0.1 * torch.tanh(self.cond_to_beta(h))

            elif self.configs.film_mode == 'none':
                # 消融：关闭 FiLM 调制
                gamma = torch.ones_like(h[..., :self.d_model])
                beta = torch.zeros_like(h[..., :self.d_model])

            elif self.configs.film_mode == 'random':
                # 消融：随机调制
                gamma = 1.0 + 0.1 * torch.randn_like(h[..., :self.d_model])
                beta = 0.1 * torch.randn_like(h[..., :self.d_model])

            elif self.configs.film_mode == 'trend_only':
                # 消融：只用趋势条件
                gamma = 1.0 + 0.1 * torch.tanh(self.cond_to_gamma(t_emb))
                beta = 0.1 * torch.tanh(self.cond_to_beta(t_emb))

            elif self.configs.film_mode == 't_only':
                # 消融：只用时间步嵌入
                gamma = 1.0 + 0.1 * torch.tanh(self.cond_to_gamma(zeros_t))
                beta = 0.1 * torch.tanh(self.cond_to_beta(zeros_t))

            # 可选：复用去噪层做细化
            if getattr(self.configs, 'use_refine_in_ft', 0) == 1:
                feat = emb
                for layer in self.denoise_layers_cond:
                    feat = feat + layer(feat, gamma, beta)
                emb = feat




        seasonal_enc = emb.reshape(
            batch_size, num_features, -1, self.d_model
        )  # [batch_size, num_features, seq_len, d_model]
        # x = torch.fft.ifft(x,dim=-2).real
        # forecast
        seasonal_enc = self.head(seasonal_enc)  # [bs, pred_len, n_vars]
        y_enc = seasonal_enc + self.regression[0](trend.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # denormalization
        y_enc = y_enc * (stdevs[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
        y_enc = y_enc + (means[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)

        return y_enc, freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss
        # return x,0,0,0

    def forward(self, batch_x,x_mask,i=0):

        if self.task_name == "pretrain":
            return self.pretrain(batch_x,x_mask,i)
        elif self.task_name == "finetune":
            dec_out,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss = self.forecast(batch_x,x_mask,i)
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

    # loss
    parser.add_argument('--del_orth_loss', type=int, help='del_orth_loss', default=0)
    parser.add_argument('--del_season_freq_loss', type=int, help='del_season_freq_loss', default=0)
    parser.add_argument('--del_smoothness_loss', type=int, help='del_smoothness_loss', default=0)
    parser.add_argument('--del_freq_loss', type=int, help='del_freq_loss', default=0)
    parser.add_argument('--del_recon_loss', type=int, help='del_recon_loss', default=0)

    parser.add_argument('--log_var_freq', type=float, help='del_orth_loss', default=0.2)
    parser.add_argument('--log_var_orth', type=float, help='del_season_freq_loss', default=6.0)
    parser.add_argument('--log_var_smooth', type=float, help='del_smoothness_loss', default=1.0)
    parser.add_argument('--log_var_season_freq', type=float, help='del_freq_loss', default=4.0)

    parser.add_argument('--log_var_recon', type=float, help='del_freq_loss', default=0.1)
    parser.add_argument('--use_defire_noise', type=int, help='use_defire_noise', default=0)
    parser.add_argument('--use_trend_layer', type=int, help='use_trend_layer', default=1)
    parser.add_argument('--use_positional_encoding', type=int, help='use_positional_encoding', default=1)
    parser.add_argument('--use_sostoken', type=int, help='use_sostoken', default=1)
    parser.add_argument('--use_init_loss', type=int, help='del_freq_loss', default=0)
    parser.add_argument('--use_inner_encoder', type=int, help='use_inner_encoder', default=1)

    parser.add_argument('--use_inner_new_decomp', type=int, help='use_inner_new_decomp', default=1)
    parser.add_argument('--use_new_decomp', type=int, help='use_new_decomp', default=1)
    parser.add_argument('--use_denoise', type=int, help='use_denoise', default=1)
    parser.add_argument('--use_loss_compute', type=int, help='use_loss_compute', default=1)

    parser.add_argument('--distance', type=int, help='distance', default=10)

    # 2025-09-25 17:24:26 两种模式
    parser.add_argument('--pretrain_mode', type=str, help='mask or noise', default='mask')
    parser.add_argument('--freeze_decomp_in_pretrain', type=int, help='A or B', default=1)
    parser.add_argument('--mask_ratio', type=float, help='mask_ratio', default=0.5)
    parser.add_argument('--mask_block', type=int, help='mask_block', default=13)
    parser.add_argument('--use_t_embed', type=int, help='use_t_embed', default=1)
    parser.add_argument('--use_init_loss_pretrain', type=int, help='use_init_loss_pretrain', default=0)
    parser.add_argument('--use_init_loss_finetune', type=int, help='use_init_loss_finetune', default=1)
    parser.add_argument('--use_geo_mask', type=int, help='use_geo_mask', default=1)
    parser.add_argument('--use_film_in_ft', type=int, help='use_film_in_ft', default=1)
    parser.add_argument('--pretrained_backbone', type=int, help='pretrained_backbone', default=1)
    parser.add_argument('--pretrain_noise', type=str, help='mask or tembed', default='mask')
    parser.add_argument('--predict_eps', type=int, help='predict_eps', default=1)
    parser.add_argument('--use_refine_in_ft', type=int, help='use_refine_in_ft', default=0)
    parser.add_argument('--use_pretrain_in_ft', type=int, help='use_pretrain_in_ft', default=1)
    parser.add_argument('--film_mode', type=str, help='film_mode', default='full')
    parser.add_argument('--destroy_season', type=int, help='destroy_season', default=1)
    parser.add_argument('--use_film', type=int, help='use_film', default=1)
    parser.add_argument('--use_finetune_encoder', type=int, help='use_finetune_encoder', default=1)
    parser.add_argument('--use_pretrain_encoder', type=int, help='use_pretrain_encoder', default=1)
    parser.add_argument('--max_lag', type=int, help='max_lag', default=63)
    parser.add_argument('--num_scales', type=int, help='num_scales', default=4)
    parser.add_argument('--peak_threshold', type=float, help='peak_threshold', default=0.3)
    parser.add_argument('--destroy_mode', type=str, help='destroy_mode', default='season')

    configs = parser.parse_args()

    return configs


def get_checkpoint_path(base_dir, dataset_name, task_type):
    """
    根据数据集名称和任务类型自动选择对应路径

    参数:
        base_dir (str): 基础目录，例如 "F:\\模型的绘图等数据\\20250801MSCD改进版本的权重等\\outputs"
        dataset_name (str): 数据集名称，例如 "Weather"
        task_type (str): 任务类型，例如 "pretrain" 或 "finetune"

    返回:
        str: 匹配到的文件夹路径，如果未找到则返回 None
    """
    # 决定在哪个子目录查找
    sub_dir = "pretrain_checkpoints" if task_type.lower() == "pretrain" else "checkpoints"
    search_dir = os.path.join(base_dir, sub_dir)

    if not os.path.exists(search_dir):
        print(f"目录不存在: {search_dir}")
        return None

    # 遍历子目录寻找包含 dataset_name 的文件夹
    candidates = [
        os.path.join(search_dir, d) for d in os.listdir(search_dir)
        if os.path.isdir(os.path.join(search_dir, d)) and dataset_name.lower() in d.lower()
    ]

    if not candidates:
        print(f"未找到包含 '{dataset_name}' 的文件夹")
        return None

    # 如果有多个匹配，按修改时间排序（最新的放前面）
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]  # 取最新的



if __name__ == '__main__':
    # 示例

    # weights  = torch.randn(32,7,4,1)
    #
    # project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    #
    # path = project_path + os.sep + 'visiofigs' + os.sep + 'ETTh1' + os.sep
    #
    # # -------------------------------------------  绘制 heatmap for weight------------------
    #
    # i=1
    # # ======== 获取权重 ========
    # weights_mean = weights.squeeze(-1).mean(dim=0)
    # weights_mean = weights_mean.unsqueeze(0)  # [C,K]
    # plot_smooth_raw_heatmaps_smoothmore([weights_mean],sigma=3.0,           # 更大平滑
    # smooth_repeat=1,     # 多次滤波
    #                                      titles=["Scale Weights"], show_grid=False, path=path,
    #                                     name=str(i) + '_weight_heatmap.png')


    # pretrain
    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\ETTh1_dln_1"
    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\ETTm2_dln_3"

    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\Electricity_dln_2"
    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\ETTh2_dln_2"
    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\Weather_dln_1"
    #
    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\ETTm1_dln_1"

    # finetune
    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_ETTh1_M_il96_ll48_pl96_dm32_df64_nh16_el2_dl1_fc1_dp0.2_hdp0.1_ep10_bs16_lr0.0001_dln_1"
    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_ETTh2_M_il96_ll48_pl96_dm8_df32_nh8_el2_dl1_fc1_dp0.4_hdp0.1_ep10_bs16_lr0.0001_dln_2"
    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_ETTm1_M_il96_ll48_pl96_dm32_df64_nh8_el2_dl1_fc1_dp0.1_hdp0.0_ep10_bs64_lr0.0001_dln_1"
    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_ETTm2_M_il96_ll48_pl96_dm8_df16_nh8_el2_dl1_fc1_dp0.4_hdp0.1_ep10_bs64_lr0.0001_dln_3"
    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_Weather_M_il96_ll48_pl96_dm64_df64_nh8_el2_dl1_fc1_dp0.2_hdp0.1_ep10_bs16_lr0.0004_dln_1"
    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_Electricity_M_il96_ll48_pl96_dm128_df256_nh16_el2_dl1_fc1_dp0.2_hdp0.0_ep10_bs16_lr0.0004_dln_2"




    # folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_ETTh1_M_il96_ll48_pl96_dm32_df64_nh16_el2_dl1_fc1_dp0.2_hdp0.1_ep10_bs16_lr0.0001_dln_1"
    # folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\pretrain_checkpoints\Traffic"
    # folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\pretrain_checkpoints\ETTh1_dln_1"
    # folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\pretrain_checkpoints\ETTh1_dln_1"






    configs = get_config()



    use_defire_noise = 0
    use_inner_encoder = 0
    use_positional_encoding = 1
    use_sostoken = 1

    use_loss_compute = 1
    use_new_decomp = 1
    use_denoise = 1
    use_inner_new_decomp = 1

    use_init_loss = 0

    del_orth_loss = 0
    del_season_freq_loss = 1
    del_freq_loss = 0
    del_smoothness_loss = 0
    del_recon_loss = 0
    # del_smoothness_loss=1
    # del_recon_loss=1
    # h1m1 01000

    log_var_freq = 0.3
    log_var_orth = 0.01
    log_var_smooth = 0.0
    log_var_season_freq = 0.05
    log_var_recon = 0.0
    denoise_layers_num=3
    pretrain_mode="noise"


    d_model = 64
    n_heads = 8
    c_out=21
    d_ff=64
    patch_len=2
    stride =2
    batch_size=16
    configs.task_name = 'pretrain'
    configs.root_path = 'weather'
    configs.data_path = 'weather.csv'
    configs.model_id = 'weather'
    configs.model = 'TimeDART'
    configs.data = 'Weather'
    configs.features = 'M'
    configs.input_len = 96
    configs.e_layers = 2
    configs.d_layers = 1
    configs.enc_in = c_out
    configs.dec_in = c_out
    configs.c_out = c_out
    configs.n_heads = n_heads
    configs.d_model = d_model
    configs.d_ff = d_ff
    configs.denoise_layers_num = denoise_layers_num
    configs.patch_len = patch_len
    configs.stride = stride
    configs.head_dropout = 0.1
    configs.pretrain_mode = pretrain_mode




    configs.batch_size = batch_size
    configs.lr_decay = 0.5
    configs.lradj = 'step'
    configs.time_steps = 1000
    configs.scheduler = 'cosine'
    configs.patience = 3
    configs.learning_rate = 0.0001
    configs.pct_start = 0.3









    configs.del_orth_loss=del_orth_loss
    configs.del_season_freq_loss=del_season_freq_loss
    configs.del_smoothness_loss=del_smoothness_loss
    configs.del_freq_loss=del_freq_loss
    configs.del_recon_loss=del_recon_loss
    configs.log_var_recon=log_var_recon
    configs.log_var_freq=log_var_freq
    configs.log_var_orth=log_var_orth
    configs.log_var_smooth=log_var_smooth
    configs.log_var_season_freq=log_var_season_freq
    configs.use_init_loss=use_init_loss
    configs.use_new_decomp=use_new_decomp
    configs.use_loss_compute=use_loss_compute
    configs.use_denoise=use_denoise
    configs.use_inner_new_decomp=use_inner_new_decomp
    configs.use_defire_noise=use_defire_noise
    configs.use_positional_encoding=use_positional_encoding
    configs.use_sostoken=use_sostoken
    configs.use_inner_encoder=use_inner_encoder
    configs.down_sampling_window=2















    base_dir = r"F:\模型的绘图等数据\TimeDART的最终结果实验ckp-202510171056\outputs"
    dataset_name = configs.data
    task_type = configs.task_name
    path = get_checkpoint_path(base_dir, dataset_name, task_type)
    print("匹配到的路径:", path)
    folder_path = path
    if os.path.isdir(folder_path):
        # checkpoint_path = os.path.join(folder_path, 'checkpoint.pth')
        ckpt_best = os.path.join(folder_path, 'ckpt_best.pth')
        ckpt_default = os.path.join(folder_path, 'checkpoint.pth')
        checkpoint_path = ckpt_best if os.path.exists(ckpt_best) else ckpt_default
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path)
            # state_dictlist = state_dict['model_state_dict']
            # pretrain
            state_dictlist = state_dict





    train_data, train_loader = data_provider(configs, flag="train")
    vali_data, vali_loader = data_provider(configs, flag="val")

    # exp = Exp_TimeDART(configs)  # set experiments
    # train_data, train_loader = exp._get_data(flag="train")
    # vali_data, vali_loader = exp._get_data(flag="val")
    configs.device = 'cpu'

    model = Model(configs)

    # 处理多GPU训练保存的权重（如果有'module.'前缀）
    # state_dict = {k.replace('module.', ''): v for k, v in state_dictlist.items()}  # 去除前缀
    new_pth = model.state_dict()
    public_dict = {}

    for k, v in state_dictlist.items():
        for kk in new_pth.keys():
            if kk in k:
                public_dict[kk] = v
                break
    new_pth.update(public_dict)
    model.load_state_dict(new_pth)
    # 加载权重到模型
    # model.load_state_dict(state_dictlist)

    # 设置为评估模式（固定Dropout和BatchNorm）
    model.eval()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            train_loader
    ):
        batch_x = batch_x.float().to(model.device)
        batch_y = batch_y.float().to(model.device)
        batch_x_mark = batch_x_mark.float().to(model.device)

        # import torch
        # import torch.nn.functional as F
        # import matplotlib.pyplot as plt
        #
        # # ========== 1. 模拟数据 ==========
        # B, L, C = 8, 128, 4  # Batch, Length, Channels
        # torch.manual_seed(0)
        # x = batch_x  # [B, L, C]
        #
        #
        # # ========== 2. 自相关 + 峰值检测 ==========
        # def find_peaks_torch(acf, height, distance=10, max_num=6):
        #     peaks = torch.zeros_like(acf, dtype=torch.bool)
        #     peaks[1:-1] = (acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:])
        #     peaks &= (acf >= height)
        #     candidate_indices = torch.where(peaks)[0]
        #     if len(candidate_indices) == 0:
        #         return torch.tensor([], device=acf.device)
        #     peak_values = acf[candidate_indices]
        #     sorted_indices = torch.argsort(peak_values, descending=True)
        #     sorted_candidates = candidate_indices[sorted_indices]
        #     selected = []
        #     for idx in sorted_candidates:
        #         if all(abs(idx - s) > distance for s in selected):
        #             selected.append(idx)
        #             if len(selected) >= max_num:
        #                 break
        #     return torch.tensor(selected, device=acf.device, dtype=torch.long)
        #
        #
        # def calculate_acf_mean(data, max_lag=63):
        #     x = data.permute(0, 2, 1).contiguous()
        #     B, C, L = x.shape
        #     max_lag = min(max_lag, L - 1)
        #     x_mean = x.mean(dim=2, keepdim=True)
        #     x_centered = x - x_mean
        #     x_norm = x_centered / (x_centered.std(dim=2, keepdim=True) + 1e-8)
        #     pad_size = L - 1
        #     x_padded = F.pad(x_norm, (0, pad_size))
        #     fft_x = torch.fft.rfft(x_padded, dim=2)
        #     acf = torch.fft.irfft(fft_x * fft_x.conj(), dim=2)[..., :L]
        #     acf = acf / (acf[..., :1] + 1e-8)
        #     mean_acf = acf.mean(dim=(0, 1))
        #     return mean_acf[:max_lag]
        #
        #
        # mean_acf = calculate_acf_mean(x, max_lag=63)
        # peaks = find_peaks_torch(mean_acf, height=0.3, distance=5, max_num=6)
        #
        # # ========== 3. 绘制 ACF + 峰值 ==========
        # # ======== 绘图 ========
        # if len(peaks) == 0:
        #     continue
        # plt.figure(figsize=(6, 4))
        # plt.plot(mean_acf.numpy(), label="Mean ACF")
        # plt.scatter(peaks.numpy(), mean_acf[peaks].numpy(), color='red', zorder=5, label="Detected Peaks")
        # plt.xlabel("Lag")
        # plt.ylabel("ACF")
        # plt.title("Autocorrelation & Detected Peaks ")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('peaks.png')
        #
        # # ======== 绘制直方图 ========
        # plt.figure(figsize=(6, 4))
        # plt.hist(peaks.numpy(), bins=np.arange(0, 64, 2), color="gray", edgecolor="black")
        # plt.title("Peak Lag Distribution")
        # plt.xlabel("Lag (bin)")
        # plt.ylabel("Count")
        # plt.tight_layout()
        # plt.savefig('Lag.png')













        # batch_x_m = batch_x_m.float()

        # batch_x= torch.randn(1,336,7)
        # batch_y= torch.randn(16,336,4)
        # x_res= torch.randn(16,336,7)


        # configs.device = batch_x.device

        # mask = torch.ones_like(x)
        # # x_enc 64 336 7 ; x_mark_enc 16 336 4 ； batch_x 16 336 7  mask 64 336 7
        c = model(batch_x,batch_y,i)
        print(i)
        break
        d = 'end'
