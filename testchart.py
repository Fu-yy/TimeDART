import torch
import torch.fft
import matplotlib.pyplot as plt
import numpy as np

def low_pass_filter_time_series(data, cutoff_ratio=0.1):
    """
    对输入的时间序列数据应用低通滤波器，去除高频噪声。

    参数:
    - data: 输入时间序列，形状为 (batchsize, seqlen, dim)
    - cutoff_ratio: 截断比例，决定保留多少低频成分 (0 < cutoff_ratio < 0.5)

    返回:
    - filtered_data: 经过低通滤波后的时间序列，形状与输入相同
    """
    batchsize, seqlen, dim = data.shape
    # 进行傅里叶变换，沿着时间序列长度维度
    fft = torch.fft.fft(data, dim=1)
    fft_shift = torch.fft.fftshift(fft, dim=1)

    # 创建低通滤波器掩码
    freq = torch.fft.fftfreq(seqlen).to(data.device)  # 频率范围 [-0.5, 0.5)
    freq_shifted = torch.fft.fftshift(freq)  # 移动到 [-0.5, 0.5) 中心对齐
    cutoff = cutoff_ratio * 0.5  # 因为频率范围是 [-0.5, 0.5)

    mask = (torch.abs(freq_shifted) <= cutoff).float().unsqueeze(0).unsqueeze(-1)  # 形状 (1, seqlen, 1)

    # 应用掩码
    fft_shift_filtered = fft_shift * mask

    # 反转移位
    fft_filtered = torch.fft.ifftshift(fft_shift_filtered, dim=1)

    # 反傅里叶变换
    filtered_data = torch.fft.ifft(fft_filtered, dim=1)
    filtered_data = torch.real(filtered_data)

    return filtered_data


def gaussian_low_pass_filter_time_series_rfft(data, cutoff_ratio=0.1, sigma=1.0):
    """
    对输入的时间序列数据应用高斯低通滤波器，去除高频噪声（使用 rFFT）。

    参数:
    - data: 输入时间序列，形状为 (batchsize, seqlen, dim)
    - cutoff_ratio: 控制高斯滤波器的中心频率 (0 < cutoff_ratio < 0.5)
    - sigma: 高斯滤波器的标准差，控制滤波器的平滑程度

    返回:
    - filtered_data: 经过高斯低通滤波后的时间序列，形状与输入相同
    """
    batchsize, seqlen, dim = data.shape
    # 使用 rFFT 进行傅里叶变换
    fft = torch.fft.rfft(data, dim=1)

    # 创建高斯低通滤波器掩码
    freqs = torch.fft.rfftfreq(seqlen).to(data.device)  # 频率范围 [0, 0.5]
    gaussian_mask = torch.exp(-0.5 * ((freqs - 0) / (cutoff_ratio * sigma)) ** 2).unsqueeze(0).unsqueeze(
        -1)  # 形状 (1, seqlen//2+1, 1)

    # 应用掩码
    fft_filtered = fft * gaussian_mask

    # 反傅里叶变换，使用 irfft 恢复时域信号
    filtered_data = torch.fft.irfft(fft_filtered, n=seqlen, dim=1)

    return filtered_data


# 示例用法
if __name__ == "__main__":

    # 设置随机种子以确保可重复性
    torch.manual_seed(0)
    np.random.seed(0)

    # 生成一个带有高频噪声的示例时间序列
    batchsize = 2
    seqlen = 256
    dim = 1

    t = torch.linspace(0, 4 * np.pi, seqlen)
    # 生成基本信号：正弦波
    signal = torch.sin(t).unsqueeze(0).unsqueeze(-1).repeat(batchsize, 1, dim)

    # 添加高频噪声
    noise = 0.5 * torch.randn_like(signal)
    noisy_signal = signal + noise

    # 应用高斯低通滤波器
    cutoff_ratio = 0.2  # 控制高斯滤波器的中心频率
    sigma = 1.0  # 高斯滤波器的标准差
    filtered_signal = gaussian_low_pass_filter_time_series_rfft(noisy_signal, cutoff_ratio=cutoff_ratio, sigma=sigma)

    # 可视化结果
    fig, axs = plt.subplots(batchsize, 3, figsize=(18, 6))
    if batchsize == 1:
        axs = axs.unsqueeze(0)  # 确保 axs 的维度正确

    for b in range(batchsize):
        axs[b, 0].plot(t.numpy(), signal[b, :, 0].numpy(), label='原始信号')
        axs[b, 0].set_title(f'Batch {b + 1} - 原始信号')
        axs[b, 0].legend()

        axs[b, 1].plot(t.numpy(), noisy_signal[b, :, 0].numpy(), label='带噪声的信号')
        axs[b, 1].set_title(f'Batch {b + 1} - 带噪声的信号')
        axs[b, 1].legend()

        axs[b, 2].plot(t.numpy(), filtered_signal[b, :, 0].numpy(), label='去噪后的信号')
        axs[b, 2].set_title(f'Batch {b + 1} - 去噪后的信号')
        axs[b, 2].legend()

    plt.tight_layout()
    plt.show()
























    # ----------------------------------
    # 设置随机种子以确保可重复性
    torch.manual_seed(0)
    np.random.seed(0)

    # 生成一个带有高频噪声的示例时间序列
    batchsize = 2
    seqlen = 256
    dim = 1

    t = torch.linspace(0, 4 * np.pi, seqlen)
    # 生成基本信号：正弦波
    signal = torch.sin(t).unsqueeze(0).unsqueeze(-1).repeat(batchsize, 1, dim)

    # 添加高频噪声
    noise = 0.5 * torch.randn_like(signal)
    noisy_signal = signal + noise

    # 应用低通滤波器
    cutoff_ratio = 0.2  # 保留前10%的低频成分
    filtered_signal = low_pass_filter_time_series(noisy_signal, cutoff_ratio=cutoff_ratio)

    # 可视化结果
    fig, axs = plt.subplots(batchsize, 3, figsize=(18, 6))
    if batchsize == 1:
        axs = axs.unsqueeze(0)  # 确保 axs 的维度正确

    for b in range(batchsize):
        axs[b, 0].plot(t.numpy(), signal[b, :, 0].numpy(), label='原始信号')
        axs[b, 0].set_title(f'Batch {b+1} - 原始信号')
        axs[b, 0].legend()

        axs[b, 1].plot(t.numpy(), noisy_signal[b, :, 0].numpy(), label='带噪声的信号')
        axs[b, 1].set_title(f'Batch {b+1} - 带噪声的信号')
        axs[b, 1].legend()

        axs[b, 2].plot(t.numpy(), filtered_signal[b, :, 0].numpy(), label='去噪后的信号')
        axs[b, 2].set_title(f'Batch {b+1} - 去噪后的信号')
        axs[b, 2].legend()

    plt.tight_layout()
    plt.show()
