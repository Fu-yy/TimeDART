import torch
import math

def calculate_loss(y, x):
    """
    计算正则化损失值

    参数:
        smoothness: 平滑项（标量或张量）
        log_var_smooth: 对数方差参数（标量或张量）

    返回:
        loss: 计算结果
    """
    x = torch.tensor(x)
    # 公式实现
    return 1 / (2 * torch.exp(x)) * y + 0.5 * x
if __name__ == '__main__':

    print(torch.exp(torch.tensor(10.0)))
    print(calculate_loss(1.23627877235412,0.2)) # freq_loss
    print(calculate_loss(517.4898071289062,6)) # orth_loss
    print(calculate_loss(0.002618541009,1)) # smoothness
    print(calculate_loss(47.8028106689453,4)) # season_freq_loss
    print(calculate_loss(1.3085879047025628e-08,0.1)) # recon_loss
