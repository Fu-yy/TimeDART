import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from layers.Autoformer_EncDec import series_decomp
from datetime import datetime

# 获取当前时间
now = datetime.now()

# 方式1：标准格式化输出（示例：2023-10-25 15:30:45）
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
print("当前时间:", formatted_time)

class LearnableMultiScaleDecomp(nn.Module):
    def __init__(self, nvar, scales=[17, 45, 87]):
        super().__init__()
        self.nvar = nvar
        self.scales = [k if k % 2 == 1 else k + 1 for k in scales]
        self.num_scales = len(scales)

        # 多尺度卷积组
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(nvar, nvar, kernel_size=k, padding=(k-1)//2, groups=nvar, bias=False),
                nn.InstanceNorm1d(nvar)
            ) for k in self.scales
        ])

        # 动态权重生成器（修正分组维度）
        self.weight_gen = nn.Sequential(
            # 输入通道nvar，输出nvar*16，分组数nvar（确保可整除）
            nn.Conv1d(nvar, nvar*16, kernel_size=3, padding=1, groups=nvar),
            nn.GELU(),
            # 输出通道调整为nvar*num_scales，保持分组数nvar
            nn.Conv1d(nvar*16, nvar*self.num_scales, kernel_size=3, padding=1, groups=nvar)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for conv in self.conv_layers:
            kernel_size = conv[0].kernel_size[0]
            conv[0].weight.data = torch.ones_like(conv[0].weight) / kernel_size

        nn.init.normal_(self.weight_gen[0].weight, mean=0, std=0.01)
        nn.init.constant_(self.weight_gen[0].bias, 0.1)
        # 最后一层初始化调整为nvar*num_scales
        nn.init.normal_(self.weight_gen[-1].weight, mean=0, std=0.01/self.num_scales)

    # 频域约束（物理频率+归一化）
    def get_freq_loss(self,fused_trend, Fs=1.0, cutoff_freq=0.01):
        L = fused_trend.shape[2]
        k_cutoff = int(cutoff_freq * L / Fs)

        # 归一化
        fused_norm = fused_trend / (torch.std(fused_trend, dim=2, keepdim=True) + 1e-8)

        fft = torch.fft.rfft(fused_norm, dim=2)
        return torch.mean(torch.abs(fft[..., k_cutoff:]) ** 2)
    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]

        # 多尺度趋势提取（保持[B, C, L, K]结构）
        trends = []
        for conv in self.conv_layers:
            trend = conv(x)  # [B, C, L]
            trends.append(trend.unsqueeze(-1))  # [B, C, L, 1]
        trends = torch.cat(trends, dim=-1)  # [B, C, L, K=3]

        # 动态权重处理（维度对齐）
        weights = self.weight_gen(x)  # [B, C*K, L]
        weights = weights.view(B, self.nvar, self.num_scales, L)  # [B, C, K, L]
        weights = F.softmax(weights, dim=2)  # 沿K维度归一化

        # 维度对齐运算（关键修正）
        fused_trend = torch.einsum('bclk,bckl->bcl', trends, weights)



        seasonal = x - fused_trend
        # 频域约束
        trend_fft = torch.fft.rfft(fused_trend, dim=2)  # 时间维度为dim=2
        freq_loss = torch.mean(torch.abs(trend_fft[..., 3:]) ** 2)  # 取高频分量


        # freq_loss = self.get_freq_loss(fused_trend,Fs=1.0, cutoff_freq=0.01)
        # 正交约束
        orth_loss = torch.mean(torch.mean(seasonal * fused_trend, dim=2)) ** 2  # [B,C,L]逐点相乘后求和
        smoothness = torch.mean(torch.diff(fused_trend, n=2, dim=2) ** 2)
        return seasonal.permute(0, 2, 1), fused_trend.permute(0, 2, 1),freq_loss,orth_loss,smoothness
batch_size = 16
seq_len = 336
nvar = 7

# 生成趋势项（线性递增）
trend_base = torch.linspace(0, 10, seq_len)  # 基础趋势
trend_data = trend_base.repeat(batch_size, 1, 1).permute(0, 2, 1)  # 调整维度

# 生成季节项（正弦波）
t = torch.linspace(0, 4 * np.pi, seq_len)
season_base = torch.sin(t)  # 基础季节项
season_data = season_base.repeat(batch_size, 1, 1).permute(0, 2, 1)

# 合成数据 = 趋势 + 季节 + 噪声
x = trend_data + season_data + 0.1 * torch.randn(batch_size, seq_len, nvar)

model = LearnableMultiScaleDecomp(nvar=7)
seasonal, trend,freq_loss,orth_loss,smoothness = model(x)
decomp_multi = series_decomp(95)

season_static, trend_static= decomp_multi(x)
print(seasonal.shape)  # torch.Size([64, 96, 7])
print(trend.shape)     # torch.Size([64, 96, 7])

import matplotlib.pyplot as plt
import os


def plot_tensors(tensor_list, labels, file_name, index):
    project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    fig_path = os.path.join(project_path, 'test_t_s_figs')
    os.makedirs(fig_path, exist_ok=True)

    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '--', ':', '-.', '-']

    for i, (tensor, label) in enumerate(zip(tensor_list, labels)):
        data = tensor.cpu().detach().numpy()
        plt.plot(data[0, :, -1],
                 color=colors[i],
                 linestyle=linestyles[i],
                 linewidth=2 if i == 0 else 1.5,
                 alpha=0.8,
                 label=label)

    plt.title('Trend and Seasonal Decomposition Comparison', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'{file_name}_{index}_figure.png'))
    plt.close()


def plot_subplots(tensor_dict, file_name, index):
    project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    fig_path = os.path.join(project_path, 'test_t_s_figs')
    os.makedirs(fig_path, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 扩展颜色列表
    linestyles = ['-', '--', ':', '-.', '-.']  # 扩展线型列表

    # --- 绘制原始数据 ---
    data = tensor_dict['original'][0, :, -1].cpu().detach().numpy()
    axes[0].plot(data, color=colors[0], label='Original Data', linewidth=2)
    axes[0].set_title('Original Time Series')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- 绘制趋势对比 ---
    trend_keys = [key for key in tensor_dict if 'trend' in key]
    for j, key in enumerate(trend_keys):
        tensor = tensor_dict[key]
        data = tensor[0, :, -1].cpu().detach().numpy()
        axes[1].plot(data,
                     color=colors[j],
                     linestyle=linestyles[j],
                     label=key.replace('_', ' ').title())
    axes[1].set_title('Trend Components Comparison')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # --- 绘制季节对比 ---
    season_keys = [key for key in tensor_dict if 'season' in key]
    for j, key in enumerate(season_keys):
        tensor = tensor_dict[key]
        data = tensor[0, :, -1].cpu().detach().numpy()
        axes[2].plot(data,
                     color=colors[j],
                     linestyle=linestyles[j],
                     label=key.replace('_', ' ').title())
    axes[2].set_title('Seasonal Components Comparison')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'{file_name}_{index}_subplots.png'))
    plt.close()
# 分解后的结果
# model = LearnableMultiScaleDecomp(nvar=7)
# seasonal, trend = model(x)
decomp_multi = series_decomp(95)
season_static, trend_static = decomp_multi(x)

# 单图绘制
# plot_tensors(
#     tensor_list=[x, trend, trend_static],
#     labels=['Original', 'Model Trend', 'Static Trend'],
#     file_name='decomposition-trend',
#     index=0
# )

# plot_tensors(
#     tensor_list=[x,seasonal, season_static],
#     labels=['Original', 'Model Seasonal', 'Static Seasonal'],
#     file_name='decomposition——season',
#     index=0
# )

# 子图绘制
tensor_dict = {
    'original': x,
    'model_trend': trend,
    'static_trend': trend_static,
    'model_season': seasonal,
    'static_season': season_static
}
# plot_subplots(tensor_dict, 'decomposition', 0)
# plot_tensors([x,trend,trend_static,seasonal,season_static],'trend_season',0)
# plot_tensors([x,seasonal,season_static],'season',0)
# plot_tensors([x,trend,trend_static],'trend',0)

# 测试趋势显著度
def trend_significance(trend):
    return (trend.std(dim=1) / (trend.mean(dim=1).abs() + 1e-6)).mean()


class MultiScaleTrendExtractor(nn.Module):
    def __init__(self, input_dim, d_model, scales=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, d_model, k, padding=k // 2)
            for k in scales
        ])
        self.attn = nn.Sequential(
            nn.Linear(len(scales) * d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, len(scales)),
            nn.Softmax(dim=-1)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]

        # 多尺度特征
        features = [conv(x) for conv in self.convs]  # 各尺度特征形状 [B, d_model, L]

        # 自适应融合
        global_features = torch.cat([self.pool(f).squeeze(-1) for f in features], dim=-1)  # [B, scales*d_model]
        attn_weights = self.attn(global_features)  # [B, scales]

        # 拆分权重并按维度扩展
        fused = sum(
            w.unsqueeze(-1).unsqueeze(-1) * f  # w形状 [B,1,1], f形状 [B,d_model,L]
            for w, f in zip(attn_weights.unbind(dim=1), features)
        )

        pooled = self.pool(fused)  # [B, d_model, 1]
        return pooled.permute(0, 2, 1)  # [B, 1, d_model]
class ConditionalEncoding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.trend_extractor = MultiScaleTrendExtractor(input_dim, d_model)
        self.condition_proj = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        trend = self.trend_extractor(x)  # [B, 1, d_model]
        proj_trend = self.condition_proj(trend)
        gate = self.gate(proj_trend)
        return proj_trend * gate  # 自适应特征选择
d_model = 64
x = torch.randn(32,7,d_model)
model = ConditionalEncoding(64,64)
res = model(x)


class AdaptiveFusion_old(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # SE式通道门控
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 16),
            nn.GELU(),
            nn.Linear(d_model // 16, d_model),
            nn.Sigmoid()
        )
        # 深度可分离卷积 + GELU
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, noise_x, cond):
        # 通道融合
        channel_weights = self.channel_gate(cond.permute(0, 2, 1))  # 输入需调整为 [B,D,L]
        fused = noise_x * channel_weights.unsqueeze(1)  # [B,L,D] * [B,1,D]

        # 空间融合
        spatial_out = self.spatial_conv(fused.permute(0, 2, 1)).permute(0, 2, 1)
        fused = fused + spatial_out
        return self.norm(fused)


class DenoisingConditionDecoder_old(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.fusion = AdaptiveFusion_old(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Noise_x, X, cond):
        # 条件融合
        fused = self.fusion(Noise_x, cond)

        # 多头注意力
        attn_output, _ = self.attn(fused, X, X)
        attn_output = self.dropout(attn_output)
        attn_output = fused + attn_output
        attn_output = self.norm1(attn_output)

        # 前馈网络
        ff_output = self.ff(attn_output)
        output = attn_output + ff_output
        return self.norm2(output)




class AdaptiveFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 通道门控
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 16),
            nn.GELU(),
            nn.Linear(d_model // 16, d_model),
            nn.Sigmoid()
        )
        # 空间卷积+门控
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, noise_x, cond):
        B, L, D = noise_x.shape
        cond = cond.expand(-1, L, -1)  # 扩展条件维度 [B, L, D]

        # 通道融合
        channel_weights = self.channel_gate(cond.permute(0, 2, 1))  # [B, D]
        fused = noise_x * channel_weights.unsqueeze(1)  # [B, L, D] * [B, 1, D]

        # 空间融合
        spatial_weights = self.spatial_gate(fused.permute(0, 2, 1))  # [B, D, L]
        spatial_out = self.spatial_conv(fused.permute(0, 2, 1))      # [B, D, L]
        spatial_out = spatial_out * spatial_weights                  # 门控控制
        fused = fused + spatial_out.permute(0, 2, 1)                # [B, L, D]
        return self.norm(fused)

class DenoisingConditionDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim= embed_dim
        self.fusion = AdaptiveFusion(embed_dim)
        self.key_proj = nn.Linear(2 * embed_dim, embed_dim)  # 新增Key投影
        self.value_proj = nn.Linear(2 * embed_dim, embed_dim)  # 新增Value投影
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Noise_x, X, cond):
        # 条件融合
        fused = self.fusion(Noise_x, cond)
        B, L, D = fused.shape

        # 生成Key/Value时融合条件
        cond_expanded = cond.expand(-1, L, -1)  # [B, L, D]
        key_input = torch.cat([X, cond_expanded], dim=-1)  # [B, L, 2D]
        value_input = torch.cat([X, cond_expanded], dim=-1)
        key = self.key_proj(key_input)  # [B, L, D]
        value = self.value_proj(value_input)

        # 多头注意力
        attn_output, _ = self.attn(
            query=fused,
            key=key,
            value=value
        )
        attn_output = self.dropout(attn_output)
        attn_output = fused + attn_output
        attn_output = self.norm1(attn_output)

        # 前馈网络
        ff_output = self.ff(attn_output)
        output = attn_output + ff_output
        return self.norm2(output)

d_model = 64
x = torch.randn(32,7,d_model)
# model = ConditionalEncoding(64,64)
# res = model(x)

denoise_encoder = DenoisingConditionDecoder(d_model,dropout=0.1)
rs = denoise_encoder(x,x,x)

c = 'out'