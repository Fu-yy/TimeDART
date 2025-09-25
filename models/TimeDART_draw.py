import torch
import torch.nn as nn
from einops import rearrange, repeat
import pandas as pd
from scipy.signal import find_peaks

from data_provider.data_factory_draw import data_provider
from exp.exp_timedart_draw import Exp_TimeDART
from layers.Autoformer_EncDec import moving_avg, series_decomp, series_decomp_multi
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer, FullAttention
from layers.TimeDART_EncDec import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    CausalTransformer,
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
# TimeDART_version2
# 只保留15分解，用新的分解策略

# ----------------------

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import periodogram
def comprehensive_decomposition_eval(original, trend, seasonal):
    """
    综合分解效果评估方法
    传入参数：PyTorch Tensor格式的原始序列、趋势项、季节项
    返回：包含所有评估指标的字典，标注指标方向性
    """
    # 转换为numpy数组
    orig = original.detach().numpy()
    trd = trend.detach().numpy()
    seas = seasonal.detach().numpy()

    # 公共计算项
    residual = orig - trd - seas
    results = {}

    # ===== 基础指标 =====
    # 可逆性误差
    results['Reconstruction_Error'] = {
        'value': np.max(np.abs(orig - (trd + seas + residual))),
                        'criteria': '越小越好'
    }

    # # 残差白噪声检验
    # try:
    #     lb_test = acorr_ljungbox(residual, lags=[20], return_df=True)
    #     results['Residual_Whiteness'] = {
    #         'value': lb_test.iloc[0]['lb_pvalue'],
    #         'criteria': '越大越好'
    #     }
    # except:
    #     results['Residual_Whiteness'] = {'value': np.nan, 'criteria': '越大越好'}

    # ===== 实用指标 =====
    # 成分正交性
    cross_corr = np.correlate(trd - trd.mean(), seas - seas.mean(), mode='full')
    normalized_corr = cross_corr / (len(trd) * np.std(trd) * np.std(seas))
    results['Component_Orthogonality'] = {
        'value': np.max(np.abs(normalized_corr)),
        'criteria': '越小越好'
    }

    # 预测性能
    try:
        # 基准预测
        base_model = ARIMA(orig, order=(1, 1, 1)).fit()
        base_pred = base_model.forecast(10)

        # 分解预测
        trd_pred = ARIMA(trd, order=(1, 1, 1)).fit().forecast(10)
        seas_pred = ARIMA(seas, order=(1, 1, 1)).fit().forecast(10)

        mse_ratio = np.mean((orig[-10:] - (trd_pred + seas_pred)) ** 2) / np.mean((orig[-10:] - base_pred) ** 2)
        results['Forecast_Improvement'] = {
            'value': 1 - mse_ratio,  # 改进幅度
            'criteria': '越大越好'
        }
    except:
        results['Forecast_Improvement'] = {'value': np.nan, 'criteria': '越大越好'}

    # ===== 理论指标 =====
    # 信息准则
    def _safe_bic(series):
        try:
            return ARIMA(series, order=(1, 1, 1)).fit().bic
        except:
            return np.nan

    bic_total = _safe_bic(trd) + _safe_bic(seas)
    results['BIC_Optimization'] = {
        'value': bic_total / _safe_bic(orig) if _safe_bic(orig) else np.nan,
        'criteria': '越小越好'
    }

    # # 频域分析
    # try:
    #     f_orig, _ = periodogram(orig)
    #     f_seas, _ = periodogram(seas)
    #     results['Frequency_Match'] = {
    #         'value': np.abs(f_orig[0] - f_seas[0]),  # 主频匹配
    #         'criteria': '越小越好'
    #     }
    # except:
    #     results['Frequency_Match'] = {'value': np.nan, 'criteria': '越小越好'}

    return results


def visualize_comparison(*metrics_list, labels=None):
    """多模型指标对比可视化"""
    # 数据整合
    df = pd.concat(
        [pd.DataFrame(m).T.assign(Model=label)
         for m, label in zip(metrics_list, labels)],
        axis=0
    ).reset_index().rename(columns={'index': 'Metric'})

    # 创建对比热图
    plt.figure(figsize=(12, 6))
    pivot_table = df.pivot(index='Model', columns='Metric', values='value')
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=.5,
        annot_kws={"size": 12}
    )
    plt.title("Decomposition Quality Metrics Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()


import pandas as pd
import numpy as np


def create_comparison_table(my_metrics, baseline_metrics):
    """
    创建专业指标对比表格
    参数：
        my_metrics: 我的模型指标字典
        baseline_metrics: 基线模型指标字典
    返回：
        格式化后的对比表格（Markdown格式）
    """
    # 创建对比数据框架
    comparison = []

    for metric in my_metrics.keys():
        row = {
            'Metric': metric,
            'Direction': my_metrics[metric]['criteria'][:3],  # 显示"越大"或"越小"
            'My Model': _format_value(my_metrics[metric]['value']),
            'Baseline': _format_value(baseline_metrics[metric]['value']),
            'Comparison': _compare_values(
                my_metrics[metric]['value'],
                baseline_metrics[metric]['value'],
                my_metrics[metric]['criteria']
            )
        }
        comparison.append(row)

    df = pd.DataFrame(comparison)

    # 生成Markdown表格
    markdown_table = df.to_markdown(index=False, floatfmt=".2f")

    # 添加表格说明
    caption = ("\n\n**说明：**\n"
               "- ✅ 表示当前模型更优\n"
               "- ❌ 表示基线模型更优\n"
               "- ➖ 表示数据不可比\n"
               "- 数值格式：科学计数法用于绝对值<0.001的值")

    return markdown_table + caption


def _format_value(value):
    """专业数值格式化"""
    if pd.isna(value):
        return "N/A"
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _compare_values(my_val, base_val, criteria):
    """智能比较结果"""
    if pd.isna(my_val) or pd.isna(base_val):
        return "➖"

    if "小" in criteria:
        better = my_val < base_val
    else:
        better = my_val > base_val

    return "✅" if better else "❌"




def radar_plot_comparison(metrics1, metrics2, labels=('Model A', 'Model B')):
    """分解质量雷达图对比"""
    categories = list(metrics1.keys())
    values1 = [v['value'] for v in metrics1.values()]
    values2 = [v['value'] for v in metrics2.values()]

    # 修改后（统一归一化）
    combined_values = list(values1) + list(values2)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit([[v] for v in combined_values])
    scaled_values1 = [scaler.transform([[v]])[0][0] for v in values1]
    scaled_values2 = [scaler.transform([[v]])[0][0] for v in values2]
    # # 归一化处理（根据指标方向）
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_values1 = scaler.fit_transform([[v] for v in values1]).flatten()
    # scaled_values2 = scaler.fit_transform([[v] for v in values2]).flatten()

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # 绘图
    ax.plot(angles, scaled_values1, 'b-', label=labels[0])
    ax.fill(angles, scaled_values1, 'b', alpha=0.1)
    ax.plot(angles, scaled_values2, 'r-', label=labels[1])
    ax.fill(angles, scaled_values2, 'r', alpha=0.1)

    # 标注
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles, categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"], color="grey", size=7)
    plt.legend(loc='upper right')
    plt.savefig('TimeDART.png', dpi=300)

import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metrics_comparison(my_metrics, baseline_metrics, figsize=(12, 8)):
    """
    专业指标对比可视化
    参数：
        my_metrics: 我的模型指标字典
        baseline_metrics: 基线模型指标字典
        figsize: 图表尺寸
    """
    # 数据预处理
    metrics_order = [
        'Reconstruction_Error',
        'Residual_Whiteness',
        'Component_Orthogonality',
        'Forecast_Improvement',
        'BIC_Optimization',
        'Frequency_Match'
    ]

    # 创建对比DataFrame
    df = pd.DataFrame({
        'My Model': [my_metrics[m]['value'] for m in metrics_order],
        'Baseline': [baseline_metrics[m]['value'] for m in metrics_order]
    }, index=metrics_order)

    # 方向处理
    directions = {
        m: '↓' if '小' in my_metrics[m]['criteria'] else '↑'
        for m in metrics_order
    }

    # 创建画布
    fig, ax = plt.subplots(figsize=figsize)

    # 设置位置参数
    x = np.arange(len(metrics_order))
    width = 0.35

    # 定义颜色映射函数
    def get_colors(my_val, base_val, direction):
        if np.isnan(my_val) or np.isnan(base_val):
            return ('grey', 'grey')
        if direction == '↓':
            better = my_val < base_val
        else:
            better = my_val > base_val
        return ('#2ca02c' if better else '#d62728', '#7f7f7f')

    # 绘制柱状图
    for i, metric in enumerate(metrics_order):
        my_val = df.loc[metric, 'My Model']
        base_val = df.loc[metric, 'Baseline']
        my_color, base_color = get_colors(my_val, base_val, directions[metric])

        # 绘制我的模型
        ax.bar(x[i] - width / 2, my_val, width, color=my_color, edgecolor='black')
        # 绘制基线模型
        ax.bar(x[i] + width / 2, base_val, width, color=base_color, edgecolor='black', alpha=0.6)

        # 添加数值标注
        if not np.isnan(my_val):
            ax.text(x[i] - width / 2, my_val * 1.05, f'{my_val:.2e}' if abs(my_val) < 1e-3 else f'{my_val:.2f}',
                    ha='center', va='bottom', fontsize=9)
        if not np.isnan(base_val):
            ax.text(x[i] + width / 2, base_val * 1.05, f'{base_val:.2e}' if abs(base_val) < 1e-3 else f'{base_val:.2f}',
                    ha='center', va='bottom', fontsize=9)

    # 图表装饰
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n({dir})" for name, dir in directions.items()], rotation=45, ha='right')
    ax.set_ylabel('Metric Value')
    ax.set_title('Model Performance Comparison')

    # 添加图例
    ax.plot([], [], color='#2ca02c', label='My Model (Better)')
    ax.plot([], [], color='#d62728', label='My Model (Worse)')
    ax.plot([], [], color='#7f7f7f', alpha=0.6, label='Baseline')
    ax.legend(loc='upper right')

    # 添加网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('TimeDART.png',dpi=300)
    plt.show()
def interactive_dashboard(metrics_dict):
    """交互式指标分析仪表盘"""
    df = pd.DataFrame([
        {'Model': k, 'Metric': m, 'Value': v['value'], 'Direction': v['criteria']}
        for k, metrics in metrics_dict.items()
        for m, v in metrics.items()
    ])

    fig = px.parallel_coordinates(df,
                                  color="Model",
                                  dimensions=["Metric", "Value"],
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  labels={'Value': 'Normalized Value'},
                                  title="Decomposition Metrics Parallel Coordinates"
                                  )
    fig.show()

# ----------------------

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

        # plot_line_charts([x,seasonal,fused_trend],["x","season","trend"],name='decomp_line')
        # plot_smooth_raw_heatmaps([x,seasonal,fused_trend],["x","season","trend"],name="decomp_heat")

        # 频域约束（抑制高频）
        trend_fft = torch.fft.rfft(fused_trend, dim=-1)
        freq_loss = torch.mean(torch.abs(trend_fft[..., 2:]))  # 忽略前5个低频

        # 平滑性约束
        smooth_loss = torch.mean(torch.diff(fused_trend, n=2, dim=-1) ** 2)

        # 正交约束
        orth_loss = torch.mean((seasonal * fused_trend).sum(dim=-1) ** 2)

        # total_loss = freq_loss + 0.1 * smooth_loss + 0.1 * orth_loss
        # 季节项高频激励（可选）
        seasonal_fft = torch.fft.rfft(seasonal, dim=2)  # [B,C, L//2+1]
        season_freq_loss = -torch.mean(torch.abs(seasonal_fft[..., 5:]))  # 激励高频

        return seasonal.permute(0, 2, 1), fused_trend.permute(0, 2, 1), freq_loss,orth_loss,smooth_loss,season_freq_loss
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


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter  # 用于平滑处理

def plot_smooth_raw_heatmaps_old(tensors, smooth_method='gaussian', sigma=1.0, titles=None, layer_idx=0, name='name'):
    """
    绘制平滑版和原始版热图（分两个独立图像）

    参数：
        tensors       : 包含PyTorch张量的列表
        smooth_method : 平滑方法 ('gaussian' 或 'none')
        sigma         : 高斯滤波的标准差（仅对高斯平滑有效）
        titles       : 可选，每个子图的标题列表
    """
    # 设置学术论文风格参数
    sns.set_style("white")  # 白色背景
    plt.rcParams.update({
        'font.family': 'serif',  # 使用衬线字体（学术论文常用）
        'font.serif': ['Times New Roman'],  # 具体指定Times字体
        'font.size': 12,          # 基础字号
        'axes.labelsize': 12,    # 坐标轴标签字号
        'axes.titlesize': 14,     # 标题字号
        'xtick.labelsize': 10,    # x轴刻度字号
        'ytick.labelsize': 10,    # y轴刻度字号
        'figure.dpi': 300        # 输出分辨率
    })

    # 确保输入为2D张量
    for tensor in tensors:
        tensor = tensor[0,:,:]
        if tensor.dim() != 2:
            raise ValueError("仅支持2D张量，当前维度: {}".format(tensor.dim()))

    # ==================== 绘制平滑版本 ====================
    plt.figure(figsize=(5 * len(tensors), 5))
    for idx, tensor in enumerate(tensors):
        arr = tensor.detach().cpu().numpy()[0,:,:]

        # 应用高斯平滑
        smoothed = gaussian_filter(arr, sigma=sigma)

        plt.subplot(1, len(tensors), idx + 1)
        heatmap = sns.heatmap(
            smoothed,
            cmap='coolwarm',  # 改用对比度更好的冷暖色系
            annot=False,
            cbar=True,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Activation Value'},  # 添加颜色条标签
            linewidths=0.5,  # 添加细线分隔
            linecolor='whitesmoke',  # 浅灰色分隔线
            # vmin=-1.0,  # 固定颜色范围（根据实际数据调整）
            # vmax=1.0    # 固定颜色范围（根据实际数据调整）
        )
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)  # 颜色条刻度字号
        title = f"{titles[idx]} (Smoothed σ={sigma})" if titles else f"Tensor {idx + 1} (Smoothed)"
        plt.title(title, fontweight='bold', pad=20)  # 加粗标题，增加间距

    plt.tight_layout()
    plt.savefig(name+'smoothed_heatmaps'+str(layer_idx)+'.png', bbox_inches='tight')
    plt.close()

    # ==================== 绘制原始版本 ====================
    plt.figure(figsize=(5 * len(tensors), 5))
    for idx, tensor in enumerate(tensors):
        arr = tensor.detach().cpu().numpy()[0,:,:]

        plt.subplot(1, len(tensors), idx + 1)
        heatmap = sns.heatmap(
            arr,
            cmap='plasma',  # 改用高对比度的plasma色系
            annot=False,
            cbar=True,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Activation Value'},
            linewidths=0.5,
            linecolor='lightgray',
            # vmin=-1.0,  # 与平滑版本保持一致
            # vmax=1.0
        )

        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        title = f"{titles[idx]} (Raw)" if titles else f"Tensor {idx + 1} (Raw)"
        plt.title(title, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(name+'raw_heatmaps'+str(layer_idx)+'.png', bbox_inches='tight')
    plt.close()






import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter  # 用于平滑处理

def plot_smooth_raw_heatmaps(tensors, smooth_method='gaussian', sigma=1.0, titles=None, layer_idx=0,path='', name='name'):
    if not os.path.exists(path):
        os.makedirs(path)


    """
    绘制平滑版和原始版热图（整体正方形）
    参数：
        tensors       : 包含PyTorch张量的列表
        smooth_method : 平滑方法 ('gaussian' 或 'none')
        sigma         : 高斯滤波的标准差
        titles        : 可选，每个子图的标题列表
    """
    sns.set_style("white")  # 白色背景
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300
    })

    # ==================== 绘制平滑版本 ====================
    plt.figure(figsize=(5 * len(tensors), 5))
    for idx, tensor in enumerate(tensors):
        arr = tensor[0, :, :].detach().cpu().numpy()

        # 平滑处理
        smoothed = gaussian_filter(arr, sigma=sigma) if smooth_method == 'gaussian' else arr

        plt.subplot(1, len(tensors), idx + 1)
        ax = sns.heatmap(
            smoothed,
            cmap='coolwarm',
            annot=False,
            cbar=True,
            square=False,  # 不强制方格
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Activation Value'},
            linewidths=0.5,
            linecolor='whitesmoke'
        )
        # 设置整体为正方形
        h, w = arr.shape
        ax.set_aspect(w / h)  # 关键：让整个热图是正方形

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        title = f"{titles[idx]} (Smoothed σ={sigma})" if titles else f"Tensor {idx + 1} (Smoothed)"
        plt.title(title, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(path + name + 'smoothed_heatmaps' + str(layer_idx) + '.png', bbox_inches='tight')
    plt.close()

    # ==================== 绘制原始版本 ====================
    plt.figure(figsize=(5 * len(tensors), 5))
    for idx, tensor in enumerate(tensors):
        arr = tensor[0, :, :].detach().cpu().numpy()

        plt.subplot(1, len(tensors), idx + 1)
        ax = sns.heatmap(
            arr,
            cmap='plasma',
            annot=False,
            cbar=True,
            square=False,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Activation Value'},
            linewidths=0.5,
            linecolor='lightgray'
        )
        # 设置整体为正方形
        h, w = arr.shape
        ax.set_aspect(w / h)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        title = f"{titles[idx]} (Raw)" if titles else f"Tensor {idx + 1} (Raw)"
        plt.title(title, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(path + name + 'raw_heatmaps' + str(layer_idx) + '.png', bbox_inches='tight')
    plt.close()


def plot_smooth_raw_heatmaps_smoothmore(
        tensors,
        smooth_method='gaussian',
        sigma=1.0,
        smooth_repeat=1,       # 平滑迭代次数
        show_grid=True,        # 是否显示小格子
        titles=None,
        layer_idx=0,
        path='',
        name='name'):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if not os.path.exists(path):
        os.makedirs(path)

    sns.set_style("white")
    # 全局字体调大
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 18,           # 基础字号
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.dpi': 300
    })

    grid_linewidth = 0.5 if show_grid else 0
    grid_color = 'whitesmoke' if show_grid else None

    # ==================== 平滑版本 ====================
    plt.figure(figsize=(6 * len(tensors), 6))
    for idx, tensor in enumerate(tensors):
        arr = tensor[0, :, :].detach().cpu().numpy()

        smoothed = arr.copy()
        if smooth_method == 'gaussian':
            for _ in range(smooth_repeat):  # 多次滤波让更平滑
                smoothed = gaussian_filter(smoothed, sigma=sigma)

        plt.subplot(1, len(tensors), idx + 1)
        ax = sns.heatmap(
            smoothed,
            cmap='coolwarm',
            annot=False,
            cbar=True,
            square=False,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Activation Value'},
            linewidths=grid_linewidth,
            linecolor=grid_color
        )
        h, w = arr.shape
        ax.set_aspect(w / h)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)  # 色条刻度字体
        title = f"{titles[idx]} (Smoothed σ={sigma})" if titles else f"Tensor {idx + 1} (Smoothed)"
        plt.title(title, fontweight='bold', fontsize=20, pad=20)  # 标题更大

    plt.tight_layout()
    plt.savefig(path + name + 'smoothed_heatmaps' + str(layer_idx) + '.png', bbox_inches='tight')
    plt.close()

    # ==================== 原始版本 ====================
    plt.figure(figsize=(6 * len(tensors), 6))
    for idx, tensor in enumerate(tensors):
        arr = tensor[0, :, :].detach().cpu().numpy()

        plt.subplot(1, len(tensors), idx + 1)
        ax = sns.heatmap(
            arr,
            cmap='plasma',
            annot=False,
            cbar=True,
            square=False,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Activation Value'},
            linewidths=grid_linewidth,
            linecolor=grid_color
        )
        h, w = arr.shape
        ax.set_aspect(w / h)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        title = f"{titles[idx]} (Raw)" if titles else f"Tensor {idx + 1} (Raw)"
        plt.title(title, fontweight='bold', fontsize=20, pad=20)

    plt.tight_layout()
    plt.savefig(path + name + 'raw_heatmaps' + str(layer_idx) + '.png', bbox_inches='tight')
    plt.close()






# def plot_smooth_raw_heatmaps(tensors, smooth_method='gaussian', sigma=1.0, titles=None,layer_idx=0,name='name'):
#     """
#     绘制平滑版和原始版热图（分两个独立图像）
#
#     参数：
#         tensors       : 包含PyTorch张量的列表
#         smooth_method : 平滑方法 ('gaussian' 或 'none')
#         sigma         : 高斯滤波的标准差（仅对高斯平滑有效）
#         titles       : 可选，每个子图的标题列表
#     """
#     # 确保输入为2D张量
#     for tensor in tensors:
#         tensor = tensor[0,:,:]
#         if tensor.dim() != 2:
#             raise ValueError("仅支持2D张量，当前维度: {}".format(tensor.dim()))
#
#     # ==================== 绘制平滑版本 ====================
#     plt.figure(figsize=(5 * len(tensors), 5))
#     for idx, tensor in enumerate(tensors):
#         arr = tensor.detach().cpu().numpy()[0,:,:]
#
#         # 应用高斯平滑
#         smoothed = gaussian_filter(arr, sigma=sigma)
#
#         plt.subplot(1, len(tensors), idx + 1)
#         sns.heatmap(smoothed, cmap='viridis', annot=False,
#                     cbar=True, square=True, xticklabels=False)
#         plt.title(f"{titles[idx]}\n(Smoothed σ={sigma}" if titles else f"Tensor {idx + 1} (Smoothed)")
#
#     plt.tight_layout()
#     plt.savefig(name+'smoothed_heatmaps'+str(layer_idx)+'.png', dpi=300)
#     plt.close()
#
#     # ==================== 绘制原始版本 ====================
#     plt.figure(figsize=(5 * len(tensors), 5))
#     for idx, tensor in enumerate(tensors):
#         arr = tensor.detach().cpu().numpy()[0,:,:]
#
#         plt.subplot(1, len(tensors), idx + 1)
#         sns.heatmap(arr, cmap='viridis', annot=False,
#                     cbar=True, square=True, xticklabels=False)
#         plt.title(f"{titles[idx]}\n(Raw)" if titles else f"Tensor {idx + 1} (Raw)")
#
#     plt.tight_layout()
#     plt.savefig(name+'raw_heatmaps'+str(layer_idx)+'.png', dpi=300)
#     plt.close()
#


def plot_line_charts(tensors, titles=None,layer_idx=0,path='',name='0.jpg'):
    """
    绘制多个张量的折线图（横向排列）
    参数：
        tensors : 包含PyTorch张量的列表
        titles  : 可选，每个子图的标题列表
    """
    if not os.path.exists(path):
        os.makedirs(path)
    # 调整画布尺寸为宽幅横向布局
    plt.figure(figsize=(5 * len(tensors), 5))  # 宽度按子图数量扩展

    for idx, tensor in enumerate(tensors):
        # 转换张量到CPU并转为numpy
        arr = tensor.detach().cpu().numpy()  # 统一处理设备转移

        # 创建横向排列的子图 (1行N列)
        plt.subplot(1, len(tensors), idx + 1)

        # 绘制特定维度的数据（根据需求调整切片）
        plt.plot(arr[-1, :, -1], linewidth=2, color='steelblue')  # 示例取最后一个样本的最后一列特征

        # 优化可视化元素
        plt.grid(True, alpha=0.3)
        plt.title(titles[idx] if titles else f'Tensor {idx + 1}', fontsize=12)
        plt.xlabel('Time Step', fontsize=10)
        plt.ylabel('Feature Value', fontsize=10)
        plt.xticks(rotation=45)  # 横坐标标签旋转防重叠

    # 增强布局紧凑性
    plt.tight_layout(pad=2.0)
    plt.savefig(path + os.sep + name+str(layer_idx)+'.png', dpi=300, bbox_inches='tight')
    plt.close()  # 防止内存泄漏


def analyze_components(original, trend, seasonal,chunk=25):
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.stattools import adfuller
    original, trend, seasonal = original.detach().numpy(), trend.detach().numpy(), seasonal.detach().numpy()
    # 辅助函数：数据分块
    def chunk_data(series, window):
        """将数据分割为不重叠的chunks（长度不足的末尾部分会被舍弃）"""
        return [series[i:i + window] for i in range(0, len(series), window)
                if len(series[i:i + window]) == window]

    # ======================
    # 核心分析逻辑
    # ======================
    def perform_analysis(series_dict,chunk):
        """执行完整的平稳性分析流程"""
        L = len(trend) // chunk  # 与研究论文一致的窗口长度

        # 结果存储结构
        results = {
            'Component': [],
            'Full_ADF_pvalue': [],
            'Chunked_ADF_mean': [],
            'Stationary_Chunks': []
        }

        # 对每个组件进行分析
        for name, series in series_dict.items():
            # 全序列ADF检验
            full_adf = adfuller(series)

            # 分块分析
            chunks = chunk_data(series, L)
            chunk_pvalues = [adfuller(chunk)[1] for chunk in chunks]
            stationary_count = sum(p < 0.05 for p in chunk_pvalues)

            # 记录结果
            results['Component'].append(name)
            results['Full_ADF_pvalue'].append(full_adf[1])
            results['Chunked_ADF_mean'].append(np.mean(chunk_pvalues))
            results['Stationary_Chunks'].append(stationary_count)

        return pd.DataFrame(results)

    # ======================
    # 执行分析并格式化输出
    # ======================
    # 创建组件字典
    components = {
        'Original': original,
        'Trend': trend,
        'Seasonal': seasonal
    }

    # 执行分析
    analysis_df = perform_analysis(components,chunk)

    # 输出结果
    print("\n全序列ADF检验结果:")
    print(analysis_df[['Component', 'Full_ADF_pvalue']])

    print("\n分块分析结果（L=28）:")
    print(analysis_df[['Component', 'Chunked_ADF_mean', 'Stationary_Chunks']])

    # 返回分析结果便于后续使用
    return analysis_df


# 示例用法
# 假设已有分解好的三个组件：
# original_series, trend_series, seasonal_series = ...
# results = analyze_components(original_series, trend_series, seasonal_series)
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
        self.log_var_freq = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_var_orth = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_var_smooth = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_var_season_freq = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.decomp_multi = series_decomp(95)
        self.decomp_multi_learnable = StopLearnableMultiScaleDecomp(self.configs.c_out,max_lag=63,num_scales=4,peak_threshold=0.3,distance=10)
        # self.decomp_multi_learnable_second = StopLearnableMultiScaleDecomp(self.patch_len,num_scales=4)
        self.decomp_multi_learnable_third = StopLearnableMultiScaleDecomp(self.d_model, max_lag=31,num_scales=4,peak_threshold=0.1,distance=3)
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

        self.decomp_mov = series_decomp(25)

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
        list_ts = []
        list_ts_mov = []
        season_mov,trend_mov = self.decomp_mov(x)

        list_ts.append(x)
        list_ts_mov.append(x)
        list_ts_mov.append(season_mov)
        list_ts_mov.append(trend_mov)
        # 分解  1
        x, trend,freq_loss,orth_loss,smoothness,season_freq_loss = self.decomp_multi_learnable(x)
        # x, trend = self.decomp_multi(x)
        list_ts.append(x)
        list_ts.append(trend)

        plot_line_charts(list_ts,['source','season','trend'],name='draw_source_dynamic_fig')
        plot_line_charts(list_ts_mov,['source','season','trend'],name='draw_source_mov_fig')
        # results_ts = analyze_components(list_ts[0][0,:,-1],list_ts[2][0,:,-1],list_ts[1][0,:,-1])
        # results_ts_mov = analyze_components(list_ts_mov[0][0,:,-1],list_ts_mov[2][0,:,-1],list_ts_mov[1][0,:,-1])
        # plot_smooth_raw_heatmaps(
        #     list_ts,
        #     smooth_method='gaussian',
        #     sigma=1.5,
        #     titles=["Feature Map 1", "Feature Map 2", "Feature Map 3"]
        # )
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





        x_embedding_bias = self.add_sos_token_and_drop_last(
            x_embedding
        )  # [batch_size * num_features, seq_len, d_model]


        x_embedding_bias = self.positional_encoding(x_embedding_bias)


        # 分解  2
        # x_embedding_bias, _ = self.decomp_multi(x_embedding_bias)
        # x_embedding_bias, _ = x_embedding_bias,x_embedding_bias

        x_out = self.encoder(
            x_embedding_bias,
            is_mask=True,
        )  # [batch_size * num_features, seq_len, d_model]

        freq_loss_inner_list = []
        orth_loss_inner_list = []
        smoothness_inner_list = []
        season_freq_loss_inner_list = []
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
            # noise_x_patch, _, _ = self.diffusion(
            #     x_patch
            # )  # [batch_size * num_features, seq_len, patch_len]
            # 添加分层退火噪声
            noise_x_patch, _ = self.diffusion.noise_with_t(x_patch, t)  # 使用指定t的噪声生成

            noise_x_embedding = self.enc_embedding(
                noise_x_patch
            )  # [batch_size * num_features, seq_len, d_model]
            noise_x_embedding = self.positional_encoding(noise_x_embedding)
            noise_x_embedding_res = noise_x_embedding
            # denoise_input = noise_x_embedding
            denoise_input = self.encoder_noise(
                noise_x_embedding,
                is_mask=True,
            )  # [batch_size * num_features, seq_len, d_model]

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
            season_mov, trend_mov = self.decomp_mov(x_out)
            x_out_list_mov.append(season_mov)
            x_out_list_mov.append(trend_mov)

            x_out, x_out_trend,freq_loss_inner,orth_loss_inner,smoothness_inner ,season_freq_loss_inner= self.decomp_multi_learnable_third(x_out)

            x_out_list.append(x_out)
            x_out_list.append(x_out_trend)
            # results_ts = analyze_components(x_out_list[0][0, 0, :], x_out_list[2][0, 0, :], x_out_list[1][0, 0, :])
            # results_ts_mov = analyze_components(x_out_list_mov[0][0, 0, :], x_out_list_mov[2][0, 0, :],
            #                                     x_out_list_mov[1][0, 0, :])

            # plot_line_charts(x_out_list, ['source', 'season', 'trend'],layer_idx,"x_out_figure_line")
            plot_smooth_raw_heatmaps(
                x_out_list,
                smooth_method='gaussian',
                sigma=1.5,
                titles=["Feature Map 1", "Feature Map 2", "Feature Map 3"],
                layer_idx=layer_idx,
                name='x_out_figure_heatmap'
            )
            plot_smooth_raw_heatmaps(
                x_out_list_mov,
                smooth_method='gaussian',
                sigma=1.5,
                titles=["Feature Map 1", "Feature Map 2", "Feature Map 3"],
                layer_idx=layer_idx,
                name='x_out_figure_heatmap_mov'
            )
            # x_out, x_out_trend = self.decomp_multi(x_out)

            # x_out, x_out_trend = x_out,x_out

            # --------------------------- 添加条件 begin
            # 获取条件编码
            # cond_encoded = self.conditional_encoding_att(x_out_trend)  # [batch_size, 1, d_model]
            cond_encoded = self.conditional_encoding(x_out_trend)  # [batch_size, 1, d_model]
            # 扩展条件编码以匹配批次和特征维度
            # cond_encoded = cond_encoded.repeat_interleave(x_out_trend.size(0) // cond_encoded.size(0),
            #                                               dim=0)  # [batch_size * num_features, 1, d_model]
            # cond_encoded = cond_encoded.expand(-1, x_out_trend.size(1), -1)  # [batch_size * num_features, seq_len, d_model]

            # 将条件编码添加到嵌入中

            # --------------------------- 添加条件 end


            # For Denoising Patch Decoder
            denoise_out = layer(
                Noise_x=denoise_input,
                X=x_out,
                cond=cond_encoded
            )  # [batch_size * num_features, seq_len, d_model]

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
        # return predict_x,0,0,0
        # return predict_x,total_freq,total_orth,total_smooth,total_season_freq
        if self.configs.del_orth_loss == 1:
            total_orth = 0
        elif self.configs.del_smoothness_loss ==1:
            total_smooth = 0
        elif self.configs.del_season_freq_loss == 1:
            total_season_freq = 0
        elif self.configs.del_freq_loss ==1:
            total_freq = 0
        return predict_x,total_freq,total_orth,total_smooth,total_season_freq

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
        list_ts = []
        list_ts_mov = []
        season_mov, trend_mov = self.decomp_mov(x)

        list_ts.append(x)
        list_ts_mov.append(x)
        list_ts_mov.append(season_mov)
        list_ts_mov.append(trend_mov)


        x, trend,freq_loss,orth_loss,smoothness,season_freq_loss = self.decomp_multi_learnable(x)
        list_ts.append(x)
        list_ts.append(trend)

        plot_line_charts(list_ts, ['source', 'season', 'trend'], name='draw_source_dynamic_fig_finetune')
        plot_line_charts(list_ts_mov, ['source', 'season', 'trend'], name='draw_source_mov_fig_finetune')
        # results_ts = analyze_components(list_ts[0][0,:,-1],list_ts[2][0,:,-1],list_ts[1][0,:,-1])
        # results_ts_mov = analyze_components(list_ts_mov[0][0,:,-1],list_ts_mov[2][0,:,-1],list_ts_mov[1][0,:,-1])




        # -------------------------------
        # 获取评估结果
        metrics1 = comprehensive_decomposition_eval(list_ts[0][0,:,-1],list_ts[2][0,:,-1],list_ts[1][0,:,-1])

        # 可视化结果
        # pd.DataFrame(metrics1).T.style.bar(subset=['value'],
        #                                   align='mid',
        #                                   color=['#d65f5f', '#5fba7d'])  # 红/绿渐变色
        metrics2 = comprehensive_decomposition_eval(list_ts_mov[0][0,:,-1],list_ts_mov[2][0,:,-1],list_ts_mov[1][0,:,-1])

        # 可视化结果
        # pd.DataFrame(metrics2).T.style.bar(subset=['value'],
        #                                   align='mid',
        #                                   color=['#d65f5f', '#5fba7d'])  # 红/绿渐变色
        # 使用示例
        table_md = create_comparison_table(metrics1, metrics2)
        print(table_md)
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




        x = self.positional_encoding(x)  # [batch_size * num_features, seq_len, d_model]

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


        # return x,freq_loss,orth_loss,smoothness,season_freq_loss
        if self.configs.del_orth_loss == 1:
            orth_loss = 0
        elif self.configs.del_smoothness_loss ==1:
            smoothness = 0
        elif self.configs.del_season_freq_loss == 1:
            season_freq_loss = 0
        elif self.configs.del_freq_loss ==1:
            freq_loss = 0
        return x,freq_loss,orth_loss,smoothness,season_freq_loss
        # return x,0,0,0

    def forward(self, batch_x,x_mask,i=0):

        if self.task_name == "pretrain":
            return self.pretrain(batch_x,x_mask,i)
        elif self.task_name == "finetune":
            dec_out,freq_loss,orth_loss,smoothness,season_freq_loss = self.forecast(batch_x,x_mask)
            return dec_out[:, -self.pred_len: , :],freq_loss,orth_loss,smoothness,season_freq_loss
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
    # loss
    parser.add_argument('--del_orth_loss', type=int, help='del_orth_loss', default=0)
    parser.add_argument('--del_season_freq_loss', type=int, help='del_season_freq_loss', default=0)
    parser.add_argument('--del_smoothness_loss', type=int, help='del_smoothness_loss', default=0)
    parser.add_argument('--del_freq_loss', type=int, help='del_freq_loss', default=0)

    configs = parser.parse_args()

    return configs

def draw_alb_line():
    import matplotlib.pyplot as plt
    import numpy as np

    # 模块
    modules = ['TFOC', 'AASD', 'HCDM']

    # 数据：有/无模块 (取完整模型和对应去掉的组合)
    # ====== ETTh1 ======
    mse_etth1_with = [0.42175, 0.42175, 0.42175]
    mse_etth1_without = [0.4395, 0.44425, 0.433]
    mae_etth1_with = [0.431, 0.431, 0.431]
    mae_etth1_without = [0.435, 0.436, 0.432]

    # ====== Weather ======
    mse_weather_with = [0.24975, 0.24975, 0.24975]
    mse_weather_without = [0.25725, 0.26325, 0.2555]
    mae_weather_with = [0.2765, 0.2765, 0.2765]
    mae_weather_without = [0.27825, 0.28275, 0.27825]

    # 计算提升（Δ = 去掉 - 加上）
    mse_etth1_delta = [(w - a) / w * 100 for w, a in zip(mse_etth1_without, mse_etth1_with)]
    mae_etth1_delta = [(w - a) / w * 100 for w, a in zip(mae_etth1_without, mae_etth1_with)]
    mse_weather_delta = [(w - a) / w * 100 for w, a in zip(mse_weather_without, mse_weather_with)]
    mae_weather_delta = [(w - a) / w * 100 for w, a in zip(mae_weather_without, mae_weather_with)]
    x = np.arange(len(modules))
    width = 0.18

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2
    })

    # ------------------- 图1：彩色双指标 -------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    rects1 = ax.bar(x - 1.5 * width, mse_etth1_delta, width, label='ETTh1-MSE', color='#1f77b4')
    rects2 = ax.bar(x - 0.5 * width, mae_etth1_delta, width, label='ETTh1-MAE', color='#aec7e8')
    rects3 = ax.bar(x + 0.5 * width, mse_weather_delta, width, label='Weather-MSE', color='#ff7f0e')
    rects4 = ax.bar(x + 1.5 * width, mae_weather_delta, width, label='Weather-MAE', color='#ffbb78')

    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.0005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # ax.set_ylabel('Δ (Improvement)')
    ax.set_ylabel('Δ Improvement (%)')

    ax.set_title('Module Contribution on MSE & MAE')
    ax.set_xticks(x)
    ax.set_xticklabels(modules)
    ax.legend(ncol=2, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('MSCD_alb.png', dpi=300)


def draw_layer_line():
    import matplotlib.pyplot as plt

    # 层数
    layers = [1, 2, 3, 4]

    # 数据集及数值
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather', 'Electricity']
    mse_values = [
        [0.42175, 0.44025, 0.43925, 0.4265],  # ETTh1
        [0.387, 0.3825, 0.3835, 0.38275],  # ETTh2
        [0.395, 0.39675, 0.3975, 0.397],  # ETTm1
        [0.28475, 0.284, 0.284, 0.285],  # ETTm2
        [0.36275, 0.3835, 0.38325, 0.382],  # Exchange
        [0.24975, 0.25225, 0.2505, 0.25025],  # Weather
        [0.195, 0.19475, 0.19575, 0.19575]  # Electricity
    ]
    mae_values = [
        [0.43075, 0.437, 0.4405, 0.4355],  # ETTh1
        [0.40725, 0.404, 0.405, 0.4045],  # ETTh2
        [0.40125, 0.403, 0.40275, 0.403],  # ETTm1
        [0.334, 0.33225, 0.331, 0.3315],  # ETTm2
        [0.40625, 0.41675, 0.4175, 0.4165],  # Exchange
        [0.2765, 0.279, 0.27775, 0.27775],  # Weather
        [0.28, 0.28, 0.28025, 0.2805]  # Electricity
    ]

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2
    })

    for i, data in enumerate(datasets):
        plt.figure(figsize=(6, 4))
        plt.plot(layers, mse_values[i], marker='o', color='#1f77b4', label='MSE')
        plt.plot(layers, mae_values[i], marker='s', color='#ff7f0e', label='MAE')
        plt.xticks(layers)
        plt.xlabel('Decoder Layers')
        plt.ylabel('Error')
        plt.title(f'{data}: Effect of Decoder Layers on MSE & MAE')
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f'HCDM_layers_{data}.png', dpi=300)
        plt.close()

def draw_layer_line_new():
    import matplotlib.pyplot as plt
    import numpy as np

    layers = [1, 2, 3, 4]
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather', 'Electricity']
    mse_values = [
        [0.42175, 0.44025, 0.43925, 0.4265],
        [0.387, 0.3825, 0.3835, 0.38275],
        [0.395, 0.39675, 0.3975, 0.397],
        [0.28475, 0.284, 0.284, 0.285],
        [0.36275, 0.3835, 0.38325, 0.382],
        [0.24975, 0.25225, 0.2505, 0.25025],
        [0.195, 0.19475, 0.19575, 0.19575]
    ]
    mae_values = [
        [0.43075, 0.437, 0.4405, 0.4355],
        [0.40725, 0.404, 0.405, 0.4045],
        [0.40125, 0.403, 0.40275, 0.403],
        [0.334, 0.33225, 0.331, 0.3315],
        [0.40625, 0.41675, 0.4175, 0.4165],
        [0.2765, 0.279, 0.27775, 0.27775],
        [0.28, 0.28, 0.28025, 0.2805]
    ]

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2
    })

    for i, data in enumerate(datasets):
        plt.figure(figsize=(6, 4))
        plt.plot(layers, mse_values[i], marker='o', color='#1f77b4', label='MSE')
        plt.plot(layers, mae_values[i], marker='s', color='#ff7f0e', label='MAE')

        # ===== 标记 MSE & MAE 最低点 =====
        for values, color, label in zip([mse_values[i], mae_values[i]], ['#1f77b4', '#ff7f0e'], ['MSE', 'MAE']):
            min_idx = np.argmin(values)
            y = values[min_idx]
            plt.plot(layers[min_idx], y, 'p', color='red', markersize=10, zorder=5)
            # 动态调整文本位置：如果太靠近0，文字放上方
            offset = 0.005 if y > 0.3 else -0.005
            va = 'bottom' if y > 0.3 else 'top'
            # plt.text(layers[min_idx], y + offset, f'{y:.3f}',
            #          ha='center', va=va, fontsize=9, color='red')

        plt.xticks(layers)
        plt.xlabel('Decoder Layers')
        plt.title(f'{data}: Effect of Decoder Layers on MSE & MAE')
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f'HCDM_layers_{data}.png', dpi=300)
        plt.close()

def draw_alb_facet_bar():
    import matplotlib.pyplot as plt
    import numpy as np

    modules = ['TFOC', 'AASD', 'HCDM']
    mse_etth1_delta = [4.04, 5.08, 2.58]
    mae_etth1_delta = [0.92, 1.15, 0.23]
    mse_weather_delta = [2.92, 5.11, 2.25]
    mae_weather_delta = [0.63, 2.21, 0.63]

    x = np.arange(len(modules))
    width = 0.35
    datasets = ['ETTh1', 'Weather']
    mse_data = [mse_etth1_delta, mse_weather_delta]
    mae_data = [mae_etth1_delta, mae_weather_delta]
    colors = ['#1f77b4', '#ff7f0e']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, ax in enumerate(axes):
        ax.bar(x - width/2, mse_data[i], width, label='MSE', color=colors[0])
        ax.bar(x + width/2, mae_data[i], width, label='MAE', color=colors[1])
        ax.set_xticks(x)
        ax.set_xticklabels(modules)
        ax.set_title(datasets[i])
        if i == 0:
            ax.set_ylabel('Δ Improvement (%)')
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)

    plt.suptitle('Module Contribution on MSE & MAE')
    plt.tight_layout()
    plt.savefig('alb_facet_bar.png', dpi=300)
    plt.show()
def draw_alb_line_with_min():
    import matplotlib.pyplot as plt
    import numpy as np

    modules = ['TFOC', 'AASD', 'HCDM']
    x = np.arange(len(modules))
    mse_etth1_delta = [4.04, 5.08, 2.58]
    mae_etth1_delta = [0.92, 1.15, 0.23]
    mse_weather_delta = [2.92, 5.11, 2.25]
    mae_weather_delta = [0.63, 2.21, 0.63]

    plt.figure(figsize=(8, 5))
    for data, color, label in zip(
        [mse_etth1_delta, mae_etth1_delta, mse_weather_delta, mae_weather_delta],
        ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78'],
        ['ETTh1-MSE', 'ETTh1-MAE', 'Weather-MSE', 'Weather-MAE']
    ):
        plt.plot(x, data, marker='o', label=label, color=color)
        min_idx = np.argmin(data)
        plt.scatter(min_idx, data[min_idx], color='red', zorder=5)
        plt.text(min_idx, data[min_idx]+0.2, f'{data[min_idx]:.2f}', ha='center', color='red', fontsize=9)

    plt.xticks(x, modules)
    plt.ylabel('Δ Improvement (%)')
    plt.title('Module Contribution (Line)')
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('alb_line_min.png', dpi=300)
    plt.show()
def draw_alb_radar():
    import matplotlib.pyplot as plt
    import numpy as np

    modules = ['TFOC', 'AASD', 'HCDM']
    mse_etth1_delta = [4.04, 5.08, 2.58]
    mae_etth1_delta = [0.92, 1.15, 0.23]
    mse_weather_delta = [2.92, 5.11, 2.25]
    mae_weather_delta = [0.63, 2.21, 0.63]

    labels = np.array(modules)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    def extend(data):
        return data + data[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    for data, color, label in zip(
        [mse_etth1_delta, mae_etth1_delta, mse_weather_delta, mae_weather_delta],
        ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78'],
        ['ETTh1-MSE', 'ETTh1-MAE', 'Weather-MSE', 'Weather-MAE']
    ):
        ax.plot(angles, extend(data), color=color, linewidth=2, label=label)
        ax.fill(angles, extend(data), color=color, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('Module Contribution Radar')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig('alb_radar.png', dpi=300)
    plt.show()
def draw_alb_stacked_bar():
    import matplotlib.pyplot as plt
    import numpy as np

    modules = ['TFOC', 'AASD', 'HCDM']
    x = np.arange(len(modules))
    mse_etth1_delta = [4.04, 5.08, 2.58]
    mae_etth1_delta = [0.92, 1.15, 0.23]
    mse_weather_delta = [2.92, 5.11, 2.25]
    mae_weather_delta = [0.63, 2.21, 0.63]

    total_etth1 = [m + a for m, a in zip(mse_etth1_delta, mae_etth1_delta)]
    total_weather = [m + a for m, a in zip(mse_weather_delta, mae_weather_delta)]

    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, mse_etth1_delta, width, label='ETTh1-MSE', color='#1f77b4')
    ax.bar(x - width/2, mae_etth1_delta, width, bottom=mse_etth1_delta, label='ETTh1-MAE', color='#aec7e8')
    ax.bar(x + width/2, mse_weather_delta, width, label='Weather-MSE', color='#ff7f0e')
    ax.bar(x + width/2, mae_weather_delta, width, bottom=mse_weather_delta, label='Weather-MAE', color='#ffbb78')

    ax.set_xticks(x)
    ax.set_xticklabels(modules)
    ax.set_ylabel('Δ Improvement (%)')
    ax.set_title('Stacked Module Contribution')
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig('alb_stacked_bar.png', dpi=300)
    plt.show()
def draw_alb_line_annotated():
    import matplotlib.pyplot as plt
    import numpy as np

    modules = ['TFOC', 'AASD', 'HCDM']
    x = np.arange(len(modules))
    mse_etth1_delta = [4.04, 5.08, 2.58]
    mae_etth1_delta = [0.92, 1.15, 0.23]
    mse_weather_delta = [2.92, 5.11, 2.25]
    mae_weather_delta = [0.63, 2.21, 0.63]

    plt.figure(figsize=(8, 5))
    for data, color, label in zip(
        [mse_etth1_delta, mae_etth1_delta, mse_weather_delta, mae_weather_delta],
        ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78'],
        ['ETTh1-MSE', 'ETTh1-MAE', 'Weather-MSE', 'Weather-MAE']
    ):
        plt.plot(x, data, marker='o', label=label, color=color, linewidth=2)
        # ===== 标记最小值 + 箭头注释 =====
        min_idx = np.argmin(data)
        plt.scatter(min_idx, data[min_idx], color='red', s=50, zorder=5)
        plt.annotate(f'Min: {data[min_idx]:.2f}%',
                     xy=(min_idx, data[min_idx]), xycoords='data',
                     xytext=(min_idx, data[min_idx]+1.5),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1.2, headwidth=6),
                     ha='center', fontsize=9, color='red')

    plt.xticks(x, modules)
    plt.ylabel('Δ Improvement (%)')
    plt.title('Module Contribution (MSE & MAE)')
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('alb_line_annotated.png', dpi=300)
    plt.show()
def draw_alb_facet_bar_annotated():
    import matplotlib.pyplot as plt
    import numpy as np

    modules = ['TFOC', 'AASD', 'HCDM']
    mse_etth1_delta = [4.04, 5.08, 2.58]
    mae_etth1_delta = [0.92, 1.15, 0.23]
    mse_weather_delta = [2.92, 5.11, 2.25]
    mae_weather_delta = [0.63, 2.21, 0.63]

    x = np.arange(len(modules))
    width = 0.35
    datasets = ['ETTh1', 'Weather']
    mse_data = [mse_etth1_delta, mse_weather_delta]
    mae_data = [mae_etth1_delta, mae_weather_delta]
    colors = ['#1f77b4', '#ff7f0e']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, ax in enumerate(axes):
        bars1 = ax.bar(x - width/2, mse_data[i], width, label='MSE', color=colors[0])
        bars2 = ax.bar(x + width/2, mae_data[i], width, label='MAE', color=colors[1])
        for bars in [bars1, bars2]:
            for rect in bars:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(modules)
        ax.set_title(datasets[i])
        if i == 0:
            ax.set_ylabel('Δ Improvement (%)')
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)

    plt.suptitle('Module Contribution on MSE & MAE')
    plt.tight_layout()
    plt.savefig('alb_facet_bar_annotated.png', dpi=300)
    plt.show()
def draw_alb_radar_annotated():
    import matplotlib.pyplot as plt
    import numpy as np

    modules = ['TFOC', 'AASD', 'HCDM']
    mse_etth1_delta = [4.04, 5.08, 2.58]
    mae_etth1_delta = [0.92, 1.15, 0.23]
    mse_weather_delta = [2.92, 5.11, 2.25]
    mae_weather_delta = [0.63, 2.21, 0.63]

    labels = np.array(modules)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    def extend(data):
        return data + data[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    for data, color, label in zip(
        [mse_etth1_delta, mae_etth1_delta, mse_weather_delta, mae_weather_delta],
        ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78'],
        ['ETTh1-MSE', 'ETTh1-MAE', 'Weather-MSE', 'Weather-MAE']
    ):
        vals = extend(data)
        ax.plot(angles, vals, color=color, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.25)
        # ===== 标注每个点数值 =====
        for angle, val in zip(angles, vals):
            ax.text(angle, val + 0.3, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    # ===== 高亮最大值（加红色星号） =====
    combined = mse_etth1_delta + mae_etth1_delta + mse_weather_delta + mae_weather_delta
    max_val = max(combined)
    max_idx = combined.index(max_val) % len(modules)
    ax.plot(angles[max_idx], max_val, 'p', color='red', markersize=10, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('Module Contribution Radar', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig('alb_radar_annotated.png', dpi=300)
    plt.show()
def draw_alb_stacked_bar_annotated():
    import matplotlib.pyplot as plt
    import numpy as np

    modules = ['TFOC', 'AASD', 'HCDM']
    x = np.arange(len(modules))
    mse_etth1_delta = [4.04, 5.08, 2.58]
    mae_etth1_delta = [0.92, 1.15, 0.23]
    mse_weather_delta = [2.92, 5.11, 2.25]
    mae_weather_delta = [0.63, 2.21, 0.63]

    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, mse_etth1_delta, width, label='ETTh1-MSE', color='#1f77b4')
    bars2 = ax.bar(x - width/2, mae_etth1_delta, width, bottom=mse_etth1_delta, label='ETTh1-MAE', color='#aec7e8')
    bars3 = ax.bar(x + width/2, mse_weather_delta, width, label='Weather-MSE', color='#ff7f0e')
    bars4 = ax.bar(x + width/2, mae_weather_delta, width, bottom=mse_weather_delta, label='Weather-MAE', color='#ffbb78')

    # ===== 数值标注 =====
    for bars in [bars1, bars2, bars3, bars4]:
        for rect in bars:
            height = rect.get_height() + rect.get_y()
            ax.text(rect.get_x() + rect.get_width()/2., height + 0.1, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # ===== 突出总贡献最大的模块（加红框） =====
    total_contrib = [m + a for m, a in zip(mse_etth1_delta, mae_etth1_delta)]
    max_idx = np.argmax(total_contrib)
    ax.add_patch(plt.Rectangle((x[max_idx]-width, 0), width*2, max(total_contrib)+1,
                               fill=False, edgecolor='red', linewidth=2))

    ax.set_xticks(x)
    ax.set_xticklabels(modules, fontsize=14, fontweight='bold')
    ax.set_ylabel('Δ Improvement (%)', fontsize=14)
    ax.set_title('Stacked Module Contribution', fontsize=18, pad=15)
    ax.legend(frameon=False, ncol=2, fontsize=12)
    plt.tight_layout()
    plt.savefig('alb_stacked_bar_annotated_bigfont.png', dpi=300)
    plt.show()

def draw_alb_radar_annotated_v3():
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    import numpy as np

    modules = ['TFOC', 'AASD', 'HCDM']
    mse_etth1_delta = [4.04, 5.08, 2.58]
    mae_etth1_delta = [0.92, 1.15, 0.23]
    mse_weather_delta = [2.92, 5.11, 2.25]
    mae_weather_delta = [0.63, 2.21, 0.63]

    labels = np.array(modules)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    def extend(data):
        return data + data[:1]

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for j, (data, color, label) in enumerate(zip(
        [mse_etth1_delta, mae_etth1_delta, mse_weather_delta, mae_weather_delta],
        colors,
        ['ETTh1-MSE', 'ETTh1-MAE', 'Weather-MSE', 'Weather-MAE']
    )):
        vals = extend(data)
        ax.plot(angles, vals, color=color, linewidth=3, label=label)
        ax.fill(angles, vals, color=color, alpha=0.15)

        # 数值标签（加动态偏移防重叠）
        # for i, (angle, val) in enumerate(zip(angles, vals)):
        #     offset = 0.3 + j * 0.15
        #     ax.text(angle, val + offset, f'{val:.2f}',
        #             ha='center', va='center',
        #             fontsize=13, fontweight='bold', color=color,
        #             path_effects=[path_effects.withStroke(linewidth=1.5, foreground="white")])

    # 高亮最大值
    combined = mse_etth1_delta + mae_etth1_delta + mse_weather_delta + mae_weather_delta
    max_val = max(combined)
    max_idx = combined.index(max_val) % len(modules)
    ax.plot(angles[max_idx], max_val, 'p', color='red', markersize=14, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax.set_title('Module Contribution Radar', fontsize=18, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12)
    plt.tight_layout()
    plt.savefig('alb_radar_annotated_v3_bigfont.png', dpi=300)
    plt.show()


def draw_mse_different_dataset():
    import matplotlib.pyplot as plt
    import numpy as np

    # 定义数据
    datasets = {
        "h1": [[0.374, 0.418, 0.449, 0.446],
               [0.374, 0.424, 0.452, 0.493],
               [0.386, 0.441, 0.487, 0.503],
               [0.385, 0.439, 0.480, 0.462],
               [0.460, 0.512, 0.546, 0.544]],
        "h2": [[0.299, 0.381, 0.420, 0.430],
               [0.305, 0.388, 0.425, 0.434],
               [0.297, 0.380, 0.428, 0.427],
               [0.301, 0.378, 0.422, 0.427],
               [0.308, 0.393, 0.427, 0.436]],
        "m1": [[0.339, 0.374, 0.403, 0.464],
               [0.343, 0.380, 0.410, 0.468],
               [0.334, 0.377, 0.426, 0.491],
               [0.336, 0.378, 0.411, 0.469],
               [0.352, 0.390, 0.421, 0.462]],
        "m2": [[0.180, 0.245, 0.306, 0.405],
               [0.181, 0.245, 0.304, 0.405],
               [0.180, 0.250, 0.311, 0.412],
               [0.181, 0.247, 0.309, 0.406],
               [0.183, 0.255, 0.309, 0.412]],
        "wth": [[0.168, 0.215, 0.270, 0.346],
                [0.170, 0.217, 0.271, 0.348],
                [0.174, 0.221, 0.278, 0.358],
                [0.191, 0.236, 0.289, 0.362],
                [0.186, 0.234, 0.284, 0.356]],
        "elec": [[0.168, 0.178, 0.196, 0.237],
                 [0.177, 0.183, 0.199, 0.240],
                 [0.148, 0.162, 0.178, 0.225],
                 [0.198, 0.199, 0.212, 0.253],
                 [0.190, 0.199, 0.217, 0.258]],
    }

    models = ["MSCD", "TimeDART", "iTransformer", "Time-FFM", "PatchTST"]
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # 每个模型颜色
    markers = ['o', 's', '^', 'D', 'x']  # 每个模型的标记
    x_labels = [96, 192, 336, 720]

    # 绘制每个数据集一张图
    for dataset, values in datasets.items():
        plt.figure(figsize=(6, 4))
        for i, model in enumerate(models):
            linewidth = 2.5 if model == "MSCD" else 1.5  # MSCD加粗
            markersize = 8 if model == "MSCD" else 6  # MSCD加大点
            plt.plot(x_labels, values[i],
                     label=model,
                     color=colors[i],
                     marker=markers[i],
                     linewidth=linewidth,
                     markersize=markersize)
        plt.title(f'{dataset.upper()} - MSE vs Prediction Length', fontsize=14)
        plt.xlabel('Prediction Length', fontsize=12)
        plt.ylabel('MSE', fontsize=12)
        plt.xticks(x_labels)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=10, loc='upper left')  # 你也可以改成 'lower center' 放在下方
        plt.tight_layout()
        plt.savefig(f'{dataset}_mse_comparison.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    # draw_mse_different_dataset()
    draw_alb_radar_annotated_v3()
    # draw_alb_radar_annotated()

    draw_alb_stacked_bar_annotated()
    # draw_alb_line_annotated()

    # draw_alb_facet_bar_annotated()
    # draw_alb_facet_bar()
    # draw_alb_line_with_min()
    # draw_alb_radar()
    # draw_alb_stacked_bar()


    # data preparation
    # draw_layer_line_new()
    # draw_layer_line()
    # draw_alb_line()
    folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\checkpoints\finetune_TimeDART_ETTh1_M_il336_ll48_pl720_dm32_df64_nh16_el2_dl1_fc1_dp0.2_hdp0.1_ep10_bs16_lr0.0001_dln_1"
    folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\checkpoints\finetune_TimeDART_Traffic_M_il336_ll48_pl96_dm64_df128_nh16_el3_dl1_fc1_dp0.2_hdp0.1_ep10_bs8_lr0.003_dln_7"
    # folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\pretrain_checkpoints\Traffic"
    # folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\pretrain_checkpoints\ETTh1_dln_1"
    # folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\pretrain_checkpoints\ETTh1_dln_1"
    if os.path.isdir(folder_path):
        checkpoint_path = os.path.join(folder_path, 'checkpoint.pth')
        # checkpoint_path = os.path.join(folder_path, 'ckpt_best.pth')
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path)
            # state_dictlist = state_dict['model_state_dict']
            state_dictlist = state_dict


    configs = get_config()

    configs.task_name = 'finetune'

    configs.seq_len = 336
    configs.pred_len = 192
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
    configs.patch_len = 2
    configs.input_len = 336
    configs.root_path = 'traffic'
    # configs.root_path = 'ETT-small'
    configs.data = 'Traffic'
    configs.data_path = 'traffic.csv'
    configs.device = 'cpu'

    configs.down_sampling_layers = 2
    configs.down_sampling_window = 2

    configs.input_len = 336
    configs.label_len = 48
    configs.pred_len = 96
    configs.e_layers = 3
    configs.enc_in = 862
    configs.dec_in = 862
    configs.c_out = 862
    configs.n_heads = 16
    configs.d_model = 64
    configs.d_ff = 128
    configs.patch_len = 8
    configs.stride = 8
    configs.dropout = 0.2
    configs.head_dropout = 0.1
    configs.batch_size = 8
    configs.denoise_layers_num = 7
    configs.lr_decay = 0.5
    configs.time_steps = 1000
    configs.scheduler = 'cosine'
    configs.patience = 3
    configs.learning_rate = 0.003
    configs.pct_start = 0.2




    train_data, train_loader = data_provider(configs, flag="train")
    vali_data, vali_loader = data_provider(configs, flag="val")

    # exp = Exp_TimeDART(configs)  # set experiments
    # train_data, train_loader = exp._get_data(flag="train")
    # vali_data, vali_loader = exp._get_data(flag="val")

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
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        # batch_x_m = batch_x_m.float()

        # batch_x= torch.randn(1,336,7)
        # batch_y= torch.randn(16,336,4)
        # x_res= torch.randn(16,336,7)


        # configs.device = batch_x.device

        # mask = torch.ones_like(x)
        # # x_enc 64 336 7 ; x_mark_enc 16 336 4 ； batch_x 16 336 7  mask 64 336 7
        c = model(batch_x,batch_y)
        d = 'end'
