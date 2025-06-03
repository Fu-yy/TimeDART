import torch
import matplotlib.pyplot as plt
import numpy as np

from models.TimeDART import find_peaks_torch


def visualize_scale_calculation(data, init_conv_kernel=[15, 31, 63, 95], num_scales=3,
                                max_lag=63, peak_threshold=0.3, distance=10):
    # 生成模拟数据（如果未提供）
    if data is None:
        torch.manual_seed(42)
        time = torch.linspace(0, 10, 500)
        data = torch.stack([
            0.5 * torch.sin(2 * np.pi * 2 * time) + 0.3 * torch.randn(len(time)),
            0.7 * torch.sin(2 * np.pi * 1 * time) + 0.2 * torch.randn(len(time))
        ], dim=0).unsqueeze(0)  # [B, C, L]

    # 运行算法并捕获中间结果
    x = data.permute(0, 2, 1).contiguous()
    B, C, L = x.shape
    max_lag = min(max_lag, L - 1)

    # 预处理可视化
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 2, 1)
    plt.plot(data[0, 0].cpu().numpy(), lw=1, alpha=0.7)
    plt.title("(a) Raw Input Signal\n(Channel 1 Example)")
    plt.xlabel("Time Step")
    plt.grid(True, alpha=0.3)

    # 标准化过程
    x_mean = x.mean(dim=2, keepdim=True)
    x_centered = x - x_mean
    x_norm = x_centered / (x_centered.std(dim=2, keepdim=True) + 1e-8)

    plt.subplot(3, 2, 2)
    plt.plot(x_norm[0, 0].cpu().numpy(), lw=1, alpha=0.7)
    plt.title("(b) Normalized Signal\n(Zero Mean, Unit Variance)")
    plt.xlabel("Time Step")
    plt.grid(True, alpha=0.3)

    # 自相关计算
    pad_size = L - 1
    x_padded = torch.nn.functional.pad(x_norm, (0, pad_size))
    fft_x = torch.fft.rfft(x_padded, dim=2)
    acf = torch.fft.irfft(fft_x * fft_x.conj(), dim=2)[..., :L]
    acf = acf / (acf[..., :1] + 1e-8)
    mean_acf = acf.mean(dim=(0, 1)).cpu().numpy()  # [L]

    # 峰值检测（修改部分）
    peaks = find_peaks_torch(
        torch.from_numpy(mean_acf[1:max_lag]).to(data.device),
        height=peak_threshold,
        distance=distance,
        max_num=num_scales * 2
    )
    peaks = peaks.cpu().numpy() + 1  # 滞后修正
    peaks = peaks.astype(int)  # 新增强制类型转换

    # ACF可视化
    lags = np.arange(len(mean_acf))
    plt.subplot(3, 2, 3)
    plt.plot(lags[:max_lag], mean_acf[:max_lag], 'b-', lw=1.5, label='ACF')
    plt.axhline(peak_threshold, color='gray', linestyle='--', alpha=0.7)
    plt.title("(c) Autocorrelation Function\nwith Peak Threshold")
    plt.xlabel("Lag")
    plt.ylabel("ACF Value")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 峰值选择可视化
    plt.subplot(3, 2, 4)
    plt.plot(lags[:max_lag], mean_acf[:max_lag], 'b-', lw=1.5)
    # 峰值选择可视化（修改部分）
    candidate_peaks = peaks[peaks < max_lag].astype(int)  # 添加类型转换
    plt.plot(candidate_peaks, mean_acf[candidate_peaks], 'ro',
             markersize=8, label='Candidate Peaks')

    # 直方图统计过程
    hist = torch.histc(torch.from_numpy(candidate_peaks).float(),
                       bins=max_lag, min=0, max=max_lag - 1).cpu().numpy()

    # 最终选择的尺度
    scales = []
    hist_working = hist.copy()
    colors = ['darkorange', 'green', 'purple']
    for i in range(num_scales):
        if np.max(hist_working) == 0:
            break
        max_bin = np.argmax(hist_working)
        scales.append(max_bin)
        start = max(0, max_bin - 5)
        end = max_bin + 6
        hist_working[start:end] = 0
        plt.axvline(max_bin, color=colors[i], linestyle='--',
                    alpha=0.7, lw=2, label=f'Selected Scale {i + 1}')

    plt.title("(d) Peak Selection Process\n(Candidates vs Selected)")
    plt.xlabel("Lag")
    plt.ylabel("ACF Value")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 直方图可视化
    plt.subplot(3, 2, 5)
    plt.bar(np.arange(len(hist)), hist, width=1.0,
            color='skyblue', edgecolor='white')
    plt.title("(e) Scale Density Estimation\n(Histogram of Candidate Peaks)")
    plt.xlabel("Lag")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # 最终结果展示
    plt.subplot(3, 2, 6)
    final_scales = [s if s % 2 else s + 1 for s in scales[:num_scales]]
    if len(final_scales) < num_scales:
        final_scales += init_conv_kernel[len(final_scales):num_scales]
    plt.barh([f'Scale {i + 1}' for i in range(len(final_scales))],
             final_scales, color=colors[:len(final_scales)])
    plt.title("(f) Final Selected Scales")
    plt.xlabel("Kernel Size")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 生成示例数据（用户也可以传入自己的数据）
    sample_data = None  # 使用内置生成器
    visualize_scale_calculation(sample_data)