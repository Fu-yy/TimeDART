import torch
import numpy as np
from scipy.signal import find_peaks


def calculate_scales(data, num_scales=3, max_lag=63, peak_threshold=0.3):
    """向量化优化版卷积核尺寸自动计算"""
    # 转置数据为 [B, C, L]
    x = data.permute(0, 2, 1)
    B, C, L = x.shape

    # 批量标准化
    x_mean = x.mean(dim=2, keepdim=True)
    x_centered = x - x_mean
    x_norm = x_centered / (x_centered.std(dim=2, keepdim=True) + 1e-8)

                           # 向量化FFT计算自相关函数
    pad_size = L
    x_padded = torch.nn.functional.pad(x_norm, (0, pad_size))
    fft_x = torch.fft.rfft(x_padded, dim=2)

    # 批量计算自相关（向量化操作）
    acf = torch.fft.irfft(fft_x * fft_x.conj(), dim=2)[..., :L]
    acf = acf / (acf[..., :1] + 1e-8)  # 向量化归一化

    # 跨批次和通道聚合ACF
    mean_acf = acf.mean(dim=(0, 1)).cpu().numpy()  # [L]

    # 峰值检测
    peaks, _ = find_peaks(mean_acf[:max_lag],
                          height=peak_threshold,
                          distance=10)

    # 选择主要尺度
    if len(peaks) == 0:
        return [15, 31, 63][:num_scales]

    # 基于密度选择
    hist = np.histogram(peaks, bins=np.arange(max_lag + 1))[0]
    scales = []
    for _ in range(num_scales):
        max_bin = hist.argmax()
        if hist[max_bin] == 0:
            break
        scales.append(max_bin)
        hist[max(0, max_bin - 5):max_bin + 6] = 0

    # 确保奇数尺寸
    return sorted([s if s % 2 else s + 1 for s in scales[:num_scales]])


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

    peaks, _ = find_peaks(mean_acf[:max_lag],
                          height=peak_threshold,
                          distance=10)

    # 选择主要尺度
    if len(peaks) == 0:
        return [15, 31, 63][:num_scales]

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

    # 确保奇数尺寸
    return sorted([s if s % 2 else s + 1 for s in scales[:num_scales]])

# 使用示例
if __name__ == "__main__":
    sample_data = torch.randn(256, 100, 64)  # 大数据测试
    scales = calculate_scales(sample_data)
    scales_2 = calculate_scales_optimized(sample_data)
    print(f"Optimized scales: {scales}")
    print(f"Optimized scales: {scales_2}")