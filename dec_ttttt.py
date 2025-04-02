import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedMultiScaleConv(nn.Module):
    def __init__(self, nvar, scales=[15, 31, 63]):
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
    def __init__(self, nvar, scales=[15, 31, 63]):
        super().__init__()
        self.nvar = nvar
        self.scales = scales
        self.num_scales = len(scales)

        # 多尺度卷积组（固定参数）
        self.fixed_convs = FixedMultiScaleConv(nvar, scales)

        # 轻量权重生成器
        self.weight_gen = LightWeightGenerator(nvar, self.num_scales)

    def forward(self, x):
        # 输入形状: [Batch, Length, Channels]
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]

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

        # 频域约束（抑制高频）
        trend_fft = torch.fft.rfft(fused_trend, dim=-1)
        freq_loss = torch.mean(torch.abs(trend_fft[..., 2:]))  # 忽略前2个低频

        # 平滑性约束
        smooth_loss = torch.mean(torch.diff(fused_trend, n=2, dim=-1) ** 2)

        # 正交约束
        orth_loss = torch.mean((seasonal * fused_trend).sum(dim=-1) ** 2)

        total_loss = freq_loss + 0.1 * smooth_loss + 0.1 * orth_loss

        return seasonal.permute(0, 2, 1), fused_trend.permute(0, 2, 1), total_loss


# 测试用例
if __name__ == "__main__":
    # 参数设置
    B, L, C = 32, 100, 8
    x = torch.randn(B, L, C)

    # 初始化模型
    model = LearnableMultiScaleDecomp(nvar=C)

    # 前向传播
    seasonal, trend, loss = model(x)

    # 形状验证
    print(f"输入形状: {x.shape}")
    print(f"季节项形状: {seasonal.shape} (应与输入相同)")
    print(f"趋势项形状: {trend.shape} (应与输入相同)")
    print(f"正则损失值: {loss.item():.4f}")

    # 数值完整性检查
    assert torch.allclose(x, seasonal + trend, atol=1e-5), "季节项+趋势项应等于输入"
    print("\n所有测试通过！")