import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller



# ======================
# 数据准备（示例）
# ======================
# 生成示例数据（带弱趋势和强季节性）
np.random.seed(42)
n = 720 * 1  # 25个chunks
time = pd.date_range('2020-01-01', periods=n, freq='H')
trend = np.cumsum(np.random.normal(0, 0.1, n))  # 弱非平稳趋势
seasonality = 15 * np.sin(2 * np.pi * np.arange(n) / 24)  # 强季节性
noise = np.random.normal(0, 5, n)

data = pd.Series(trend + seasonality + noise, index=time)

import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 假设我们有一个时间序列数据 series
result = adfuller(data)

# 输出检验结果
print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:", result[4])

# 判断平稳性
if result[1] < 0.05:
    print("序列是平稳的")
else:
    print("序列是非平稳的")
# AI写代码
# ======================
# 核心函数定义
# ======================
def chunk_data(series, window=720 // 25):
    """将数据分割为不重叠的chunks"""
    return [series[i:i + window] for i in range(0, len(series), window)
            if len(series[i:i + window]) == window]


def ema_trend(series, alpha=0.3):
    """EMA趋势提取"""
    return series.ewm(alpha=alpha, adjust=False).mean()


def sma_trend(series, window=24):
    """SMA趋势提取"""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def calculate_S(trend_components):
    """计算平稳chunk数量（S值）"""
    return sum(adfuller(chunk)[1] < 0.05 for chunk in trend_components)


# ======================
# 实验流程
# ======================
# 参数设置
L = 720 // 25
chunks = chunk_data(data, L)
alphas = [0.1, 0.3, 0.5]

# 存储结果
results = {
    'Dataset': [],
    'SMA': {'ADF': [], 'S': []},
    'EMA_0.1': {'ADF': [], 'S': []},
    'EMA_0.3': {'ADF': [], 'S': []},
    'EMA_0.5': {'ADF': [], 'S': []}
}

# 主循环处理每个chunk
for chunk in chunks:
    # 原始数据检验
    results['Dataset'].append(adfuller(chunk)[1])

    # SMA分解
    sma_t = sma_trend(chunk)
    results['SMA']['ADF'].append(adfuller(sma_t)[1])

    # EMA分解
    for alpha in alphas:
        ema_t = ema_trend(chunk, alpha)
        key = f'EMA_{alpha}'
        results[key]['ADF'].append(adfuller(ema_t)[1])

# 计算统计量
final_results = {
    'Method': ['Dataset', 'SMA', 'EMA (0.1)', 'EMA (0.3)', 'EMA (0.5)']
}

# 计算平均ADF p-value
final_results['ADF (L=720) ↑'] = [
    np.mean(results['Dataset']),
    np.mean(results['SMA']['ADF']),
    np.mean(results['EMA_0.1']['ADF']),
    np.mean(results['EMA_0.3']['ADF']),
    np.mean(results['EMA_0.5']['ADF'])
]

# 计算S值（平稳chunk数量）
final_results['S (max 25) ↓'] = [
    sum(np.array(results['Dataset']) < 0.05),
    calculate_S([sma_trend(c) for c in chunks]),
    calculate_S([ema_trend(c, 0.1) for c in chunks]),
    calculate_S([ema_trend(c, 0.3) for c in chunks]),
    calculate_S([ema_trend(c, 0.5) for c in chunks])
]

# ======================
# 结果输出
# ======================
result_df = pd.DataFrame(final_results)
print("Table 6 Reproduction:")
print(result_df)