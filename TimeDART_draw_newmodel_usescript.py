import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
import pandas as pd
from scipy.signal import find_peaks
import seaborn as sns
# from layers.SelfAttention_Family import DSAttention, AttentionLayer, FullAttention
from layers.TimeDART_EncDec import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    CausalTransformer,
    Diffusion,
    DenoisingPatchDecoder,
)
from layers.Embed import Patch, PatchEmbedding, PositionalEncoding
from models.TimeDART_draw import plot_line_charts, plot_smooth_raw_heatmaps, comprehensive_decomposition_eval, \
    create_comparison_table, plot_smooth_raw_heatmaps_smoothmore
from utils.augmentations import masked_data
import torch.nn.functional as F
import torch
import os
import torch.nn as nn

import matplotlib.pyplot as plt
# TimeDART_version2
# 只保留15分解，用新的分解策略
from data_provider.data_factory_draw import data_provider
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """

    def __init__(self, alpha,input_size):
        super(EMA, self).__init__()
        # self.alpha = nn.Parameter(torch.tensor(alpha))    # Learnable alpha
        self.alpha = alpha
        self.fusion = nn.Linear(input_size,input_size)  # 自适应权重融合

    # Optimized implementation with O(1) time complexity
    def forward(self, x):
        # x: [Batch, Input, Channel]
        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
        _, t, _ = x.shape
        x_fus = self.fusion(x.permute(0,2,1)).permute(0,2,1)
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,)).to('cuda')
        weights = torch.pow((1 - self.alpha), powers).to('cuda')
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x_fus = torch.cumsum(x_fus * weights, dim=1)
        x_fus = torch.div(x_fus, divisor)
        return x_fus.to(torch.float32)

    # # Naive implementation with O(n) time complexity
    # def forward(self, x):
    #     # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
    #     s = x[:, 0, :]
    #     res = [s.unsqueeze(1)]
    #     for t in range(1, x.shape[1]):
    #         xt = x[:, t, :]
    #         s = self.alpha * xt + (1 - self.alpha) * s
    #         res.append(s.unsqueeze(1))
    #     return torch.cat(res, dim=1)

class EMA_Learnable(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type,input_size, alpha, beta):
        super(EMA_Learnable, self).__init__()

        self.ma = EMA(alpha,input_size=input_size)
    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):

        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean



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

    return sorted(last_scales[:num_scales])#,peaks



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
            """
            import matplotlib.pyplot as plt
            import numpy as np

            # 转 numpy
            mean_acf = np.array(mean_acf.detach().cpu().numpy())
            peaks = np.array(peaks.detach().cpu().numpy())
            scales = np.array(scales)

            # 全局字体调整
            plt.rcParams.update({
                'font.size': 16,  # 基础字号
                'axes.titlesize': 18,  # 标题
                'axes.labelsize': 16,  # 坐标轴标签
                'xtick.labelsize': 14,  # x轴刻度
                'ytick.labelsize': 14,  # y轴刻度
                'legend.fontsize': 14  # 图例
            })

            plt.figure(figsize=(6, 6))
            # 画ACF
            plt.plot(mean_acf, color='blue', linewidth=2, label="Mean ACF")
            # 画所有峰值
            if len(peaks) > 0:
                plt.scatter(peaks, mean_acf[peaks], color='red', s=80, zorder=5, label="Detected Peaks")
            # 画最终选择的尺度
            if len(scales) > 0:
                plt.scatter(scales, mean_acf[scales], color='green', s=150, marker='*', zorder=6,
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
                plt.savefig(save_path, dpi=300)
            else:
                plt.show()

        out = calculate_scales_optimized_toech_for_fig(
            x,
            num_scales=self.num_scales,
            distance=self.distance,
            peak_threshold=self.peak_threshold,
            max_lag=self.max_lag,
            # peak_threshold=self.peak_threshold
        )
        project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        path = project_path + os.sep+'all_draw_figs'+os.sep+self.configs.task_name+os.sep + 'decomp_visio' + os.sep + self.configs.data + os.sep+str(self.configs.pred_len)+os.sep
        if not os.path.exists(path):
            os.makedirs(path)
        if len(out) == 3:
            mean_acf, scales, peaks = out


            plot_acf_with_scales(mean_acf,peaks,scales,save_path=path + str(i) + '_'+self.configs.task_name +  self.configs.data + str(self.configs.pred_len) +'_acf.png')

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
        plot_smooth_raw_heatmaps_smoothmore([weights_mean],titles=["Scale Weights"],show_grid=False, path=path, name=str(i) +'_'+ self.configs.task_name +  self.configs.data + str(self.configs.pred_len) +'_weight_heatmap.png')
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
        plt.figure(figsize=(5, 5))
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
        # 正交约束
        orth_loss = torch.mean(torch.mean(seasonal * fused_trend, dim=2)) ** 2  # [B,C,L]逐点相乘后求和
        smoothness = torch.mean(torch.diff(fused_trend, n=2, dim=2) ** 2)
        return seasonal.permute(0, 2, 1), fused_trend.permute(0, 2, 1),freq_loss,orth_loss,smoothness

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
        # 在 Model 类中替换 ConditionalEncoding
        # self.conditional_encoding_att = SeqAttention(
        #     target_dim=self.d_model,
        #     num_heads=self.num_heads,
        #     embed_dim=self.d_model,
        #     dropout=self.dropout,
        # )
        # Decoder
        if self.task_name == "pretrain":
            self.denoising_patch_decoder = DenoisingPatchDecoder(
                d_model=configs.d_model,
                num_layers=configs.d_layers,
                num_heads=configs.n_heads,
                feedforward_dim=configs.d_ff,
                dropout=configs.dropout,
            )


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
                nn.Linear(self.input_len // (configs.down_sampling_window ** i), self.input_len)
                for i in range(configs.down_sampling_layers + 1)
            ])
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

            self.head = nn.ModuleList(
                [FlattenHead(
                    seq_len=self.seq_len // (configs.down_sampling_window ** i),
                    d_model=self.d_model,
                    pred_len=configs.pred_len,
                    dropout=configs.head_dropout,
                )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

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

        # self.log_var_freq = nn.Parameter(torch.log(torch.tensor(0.1)))
        # self.log_var_orth = nn.Parameter(torch.log(torch.tensor(0.1)))
        # self.log_var_smooth = nn.Parameter(torch.log(torch.tensor(0.1)))
        # self.log_var_season_freq = nn.Parameter(torch.log(torch.tensor(0.1)))
        # self.log_var_recon = nn.Parameter(torch.log(torch.tensor(0.1)))


        self.decomp_multi = series_decomp(25)
        self.decomp_multi_learnable = StopLearnableMultiScaleDecomp(self.configs.c_out,max_lag=self.configs.max_lag,num_scales=self.configs.num_scales,peak_threshold=self.configs.peak_threshold,distance=self.configs.distance,configs=self.configs)
        # self.decomp_multi_learnable_second = StopLearnableMultiScaleDecomp(self.patch_len,num_scales=4)
        self.decomp_multi_learnable_third = StopLearnableMultiScaleDecomp(self.d_model, max_lag=self.configs.max_lag_inner,num_scales=self.configs.num_scales_inner,peak_threshold=self.configs.peak_threshold_inner,distance=self.configs.distance_inner,configs=self.configs)
        #

        # self.decomp_multi_learnable = StopLearnableMultiScaleDecomp(self.configs.c_out,max_lag=63,num_scales=1,peak_threshold=0.3,distance=10)
        # # self.decomp_multi_learnable_second = StopLearnableMultiScaleDecomp(self.patch_len,num_scales=4)
        # self.decomp_multi_learnable_third = StopLearnableMultiScaleDecomp(self.d_model, max_lag=31,num_scales=1,peak_threshold=0.1,distance=3)

        # self.decomp_multi_learnable = LearnableMultiScaleDecomp(self.configs.c_out)
        # self.decomp_multi_learnable_second = LearnableMultiScaleDecomp(self.patch_len,scales=[5, 13, 25])
        # self.decomp_multi_learnable_third = LearnableMultiScaleDecomp(self.d_model,scales=[5, 13, 25])
        self.denoise_layers_num = configs.denoise_layers_num
        self.denoise_layers = nn.ModuleList([
            DenoisingPatchDecoder(
                d_model=configs.d_model,
                num_layers=configs.d_layers,
                num_heads=configs.n_heads,
                feedforward_dim=configs.d_ff,
                dropout=configs.dropout,
            )
            for _ in range(self.denoise_layers_num)
        ])

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

    def pretrain(self, x,i=0):

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

        list_ts = []
        list_ts_mov = []
        list_ts.append(x)
        list_ts_mov.append(x)
        season_mov, trend_mov = self.decomp_multi(x)
        freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = torch.tensor(0.0), torch.tensor(
            0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        list_ts_mov.append(season_mov)
        list_ts_mov.append(trend_mov)
        x, trend,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss = self.decomp_multi_learnable(x,i)





        list_ts.append(x)
        list_ts.append(trend)

        project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        path = project_path + os.sep+'all_draw_figs'+os.sep+self.configs.task_name+ os.sep+  'visiofigs'  + os.sep + self.configs.data + os.sep + str(self.configs.pred_len) + os.sep

        plot_line_charts(list_ts,['source','season','trend'],path=path,name=str(i)+"_"+ self.configs.task_name +  self.configs.data + str(self.configs.pred_len) + '_draw_source_dynamic_fig_')
        plot_line_charts(list_ts_mov,['source','season','trend'],path=path,name=str(i)+'_'+ self.configs.task_name +  self.configs.data + str(self.configs.pred_len) + '_draw_source_mov_fig_')


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
            x_out_list = []
            x_out_list.append(x_out)
            x_out_list_mov = []
            x_out_list_mov.append(x_out)

            season_mov, trend_mov = self.decomp_multi(x_out)
            freq_loss_inner, orth_loss_inner, smoothness_inner, season_freq_loss_inner, recon_loss_inner = torch.tensor(
                0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
            x_out_list_mov.append(season_mov)
            x_out_list_mov.append(trend_mov)

            x_out, x_out_trend,freq_loss_inner,orth_loss_inner,smoothness_inner ,season_freq_loss_inner,recon_loss_inner= self.decomp_multi_learnable_third(x_out,i)

            x_out_list.append(x_out)
            x_out_list.append(x_out_trend)
            project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

            path  = project_path + os.sep+'all_draw_figs'+os.sep+self.configs.task_name + os.sep +'visiofigs'+os.sep+ self.configs.data + os.sep+str(self.configs.pred_len)+os.sep

            plot_smooth_raw_heatmaps(
                x_out_list,
                smooth_method='gaussian',
                sigma=1.5,
                titles=["Feature Map x", "Feature Map Season", "Feature Map Trend"],
                layer_idx=layer_idx,
                path=path,
                name=str(i) +'_'+ self.configs.task_name +  self.configs.data + str(self.configs.pred_len) +  '_heatmap_layer_idx='
            )
            plot_smooth_raw_heatmaps(
                x_out_list_mov,
                smooth_method='gaussian',
                sigma=1.5,
                titles=["Feature Map x", "Feature Map Season", "Feature Map Trend"],
                layer_idx=layer_idx,
                path=path,
                name=str(i) +'_'+ self.configs.task_name +  self.configs.data + str(self.configs.pred_len) + '_heatmap_mov_layer_idx='
            )

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

    def forecast(self, x,i):
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
        list_ts = []
        list_ts_mov = []

        season_mov, trend_mov = self.decomp_multi(x)
        freq_loss, orth_loss, smoothness, season_freq_loss, recon_loss = torch.tensor(0.0), torch.tensor(
            0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        list_ts.append(x)
        list_ts_mov.append(x)
        list_ts_mov.append(season_mov)
        list_ts_mov.append(trend_mov)


        x, trend,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss = self.decomp_multi_learnable(x,i)

        list_ts.append(x)
        list_ts.append(trend)



        # x, trend = x,x
        x = self.channel_independence[0](x)  # [batch_size * num_features, input_len, 1]
        x = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

        # x = torch.fft.fft(x,dim=-2).imag
        x = self.enc_embedding(x)  # [batch_size * num_features, seq_len, d_model]
        project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        path = project_path + os.sep + 'all_draw_figs' + os.sep + self.configs.task_name + os.sep + 'visiofigs' + os.sep + self.configs.data + os.sep + str(
            self.configs.pred_len) + os.sep
        plot_line_charts(list_ts, ['source', 'season', 'trend'], path=path,
                         name=str(i) + self.configs.task_name + self.configs.data + str(
                             self.configs.pred_len) + '_draw_source_dynamic_fig')
        plot_line_charts(list_ts_mov, ['source', 'season', 'trend'], path=path,
                         name=str(i) + self.configs.task_name + self.configs.data + str(
                             self.configs.pred_len) + '_draw_source_mov_fig')

        # -------------------------------
        # 获取评估结果
        metrics1 = comprehensive_decomposition_eval(list_ts[0][0, :, -1], list_ts[2][0, :, -1], list_ts[1][0, :, -1])

        # 可视化结果
        # pd.DataFrame(metrics1).T.style.bar(subset=['value'],
        #                                   align='mid',
        #                                   color=['#d65f5f', '#5fba7d'])  # 红/绿渐变色
        metrics2 = comprehensive_decomposition_eval(list_ts_mov[0][0, :, -1], list_ts_mov[2][0, :, -1],
                                                    list_ts_mov[1][0, :, -1])

        # 可视化结果
        # pd.DataFrame(metrics2).T.style.bar(subset=['value'],
        #                                   align='mid',
        #                                   color=['#d65f5f', '#5fba7d'])  # 红/绿渐变色
        # 使用示例
        table_md = create_comparison_table(metrics1, metrics2)
        # print(table_md)
        # 对比可视化选择
        # visualize_comparison(metrics1, metrics2, labels=['My Model', 'Baseline'])
        # radar_plot_comparison(metrics1, metrics2, labels=['My Model', 'Baseline'])
        # plot_metrics_comparison(metrics1, metrics2)
        # 或生成交互式图表
        # interactive_dashboard({
        #     'My Model': metrics1,
        #     'Baseline': metrics2
        # })
        # -------------------------------



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
            return self.pretrain(batch_x,i)
        elif self.task_name == "finetune":
            dec_out,freq_loss,orth_loss,smoothness,season_freq_loss,recon_loss = self.forecast(batch_x,i)
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
    parser.add_argument("--train_epochs", type=int, default=1, help="train epochs")
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

    parser.add_argument('--max_lag', type=int, help='max_lag', default=63)
    parser.add_argument('--num_scales', type=int, help='num_scales', default=4)
    parser.add_argument('--peak_threshold', type=float, help='peak_threshold', default=0.3)
    parser.add_argument('--distance', type=int, help='distance', default=10)

    parser.add_argument('--max_lag_inner', type=int, help='max_lag_inner', default=31)
    parser.add_argument('--num_scales_inner', type=int, help='num_scales_inner', default=4)
    parser.add_argument('--peak_threshold_inner', type=float, help='peak_threshold_inner', default=0.1)
    parser.add_argument('--distance_inner', type=int, help='distance_inner', default=3)

    configs = parser.parse_args()

    return configs



if __name__ == '__main__':
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
    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\ETTh1_dln_1"
    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\ETTm2_dln_3"

    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\Electricity_dln_2"
    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\ETTh2_dln_2"
    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\Weather_dln_1"

    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\pretrain_checkpoints\ETTm1_dln_1"

    # finetune
    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_ETTh1_M_il96_ll48_pl96_dm32_df64_nh16_el2_dl1_fc1_dp0.2_hdp0.1_ep10_bs16_lr0.0001_dln_1"
    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_ETTh2_M_il96_ll48_pl96_dm8_df32_nh8_el2_dl1_fc1_dp0.4_hdp0.1_ep10_bs16_lr0.0001_dln_2"
    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_ETTm1_M_il96_ll48_pl96_dm32_df64_nh8_el2_dl1_fc1_dp0.1_hdp0.0_ep10_bs64_lr0.0001_dln_1"
    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_ETTm2_M_il96_ll48_pl96_dm8_df16_nh8_el2_dl1_fc1_dp0.4_hdp0.1_ep10_bs64_lr0.0001_dln_3"
    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_Weather_M_il96_ll48_pl96_dm64_df64_nh8_el2_dl1_fc1_dp0.2_hdp0.1_ep10_bs16_lr0.0004_dln_1"
    folder_path = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs\checkpoints\finetune_TimeDART_Electricity_M_il96_ll48_pl96_dm128_df256_nh16_el2_dl1_fc1_dp0.2_hdp0.0_ep10_bs16_lr0.0004_dln_2"


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

    parser.add_argument('--max_lag', type=int, help='max_lag', default=63)
    parser.add_argument('--num_scales', type=int, help='num_scales', default=4)
    parser.add_argument('--peak_threshold', type=float, help='peak_threshold', default=0.3)
    parser.add_argument('--distance', type=int, help='distance', default=10)

    parser.add_argument('--max_lag_inner', type=int, help='max_lag_inner', default=31)
    parser.add_argument('--num_scales_inner', type=int, help='num_scales_inner', default=4)
    parser.add_argument('--peak_threshold_inner', type=float, help='peak_threshold_inner', default=0.1)
    parser.add_argument('--distance_inner', type=int, help='distance_inner', default=3)
    parser.add_argument('--folder_path', type=str, help='folder_path')

    configs = parser.parse_args()




    configs = get_config()


    base_dir = r"F:\模型的绘图等数据\20250801MSCD改进版本的权重等\outputs"
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


    configs.device = 'cuda:0'
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

        if i > 20:  # 如果i大于20，结束循环
            break
        batch_x = batch_x.float().to(model.device)
        batch_y = batch_y.float().to(model.device)
        batch_x_mark = batch_x_mark.float().to(model.device)


        c = model(batch_x,batch_y,i)
        print(i)
        d = 'end'
