import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.signal import find_peaks

# ======================
# 1. 生成模拟数据（可替换为真实数据）
# ======================
# np.random.seed(42)
n_samples = 336
time = np.linspace(0, 10, n_samples)
signal = (
    1.2 * np.sin(2 * np.pi * 1.0 * time) +
    0.8 * np.cos(2 * np.pi * 2.5 * time) +
    0.3 * np.random.randn(n_samples)
)

# ======================
# 2. 创建画布和子图（显式管理轴对象）
# ======================
fig, axs = plt.subplots(3, 1, figsize=(12, 10),
                        gridspec_kw={'hspace': 0.4})
fig.suptitle('Multi-scale Analysis Pipeline',
             y=1.02, fontsize=16, weight='bold')

# ======================
# 3. 绘制原始数据图
# ======================
axs[0].plot(time, signal, color='#2c7bb6', linewidth=1.5)
axs[0].set_title('Original Time Series Data', fontsize=14, pad=15)
axs[0].set_xlabel('Time (s)', fontsize=10)
axs[0].set_ylabel('Amplitude', fontsize=10)
axs[0].grid(True, linestyle='--', alpha=0.7)

# ======================
# 4. 绘制自相关函数图（显式指定轴对象）
# ======================
plot_acf(signal, lags=100, alpha=0.05, ax=axs[1],
         title='Autocorrelation Function (ACF) with 95% Confidence Intervals',
         color='#d7191c', vlines_kwargs={'colors':'#d7191c'})
axs[1].set_ylim(-1.1, 1.1)
axs[1].set_xlabel('Lag (samples)', fontsize=10)
axs[1].set_ylabel('Correlation', fontsize=10)
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].title.set_size(14)

# ======================
# 5. 峰值检测与标注（直接操作轴对象）
# ======================
# 获取ACF值
acf_line = axs[1].lines[1]  # 第1条线是ACF曲线
acf_values = acf_line.get_ydata()

# 执行峰值检测
peaks, properties = find_peaks(
    acf_values,
    height=0.3,       # 高度阈值
    distance=20,      # 最小峰间距
    prominence=0.15   # 最小突出度
)

# 标注峰值（在axs[1]上操作）
axs[1].scatter(peaks, acf_values[peaks],
             color='#2c7bb6', zorder=5,
             s=80, label='Detected Peaks',
             edgecolors='k', linewidths=0.8)
axs[1].legend(loc='upper right', fontsize=10)

# ======================
# 6. 尺度选择可视化（直方图）
# ======================
# 生成候选尺度直方图
bins = np.arange(0, 101, 5)
axs[2].hist(peaks, bins=bins, color='#abd9e9',
          edgecolor='#2c7bb6', linewidth=0.8,
          density=True, alpha=0.7)

# 标注最终选择尺度（示例：前3个最高峰）
selected_scales = sorted(peaks[np.argsort(acf_values[peaks])[-3:]])
for scale in selected_scales:
    axs[2].axvline(scale, color='#d7191c', linestyle='--',
                 linewidth=1.5, label=f'Selected Scale: {scale}')

axs[2].set_title('Scale Selection via Histogram Density', fontsize=14, pad=15)
axs[2].set_xlabel('Lag (samples)', fontsize=10)
axs[2].set_ylabel('Normalized Frequency', fontsize=10)
axs[2].set_xlim(0, 100)
axs[2].grid(True, linestyle='--', alpha=0.7)
axs[2].legend(fontsize=10)

# ======================
# 7. 调整布局
# ======================
plt.tight_layout()
plt.show()