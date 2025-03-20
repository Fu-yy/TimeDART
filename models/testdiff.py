import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


# 自定义时间序列数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, seq_len=100, num_samples=10000):
        self.data = torch.stack([self._generate_synthetic_series(seq_len)
                                 for _ in range(num_samples)])

    def _generate_synthetic_series(self, n):
        """生成包含趋势、季节、残差的合成数据"""
        t = torch.linspace(0, 10, n)
        trend = 0.3 * t
        season = 2 * torch.sin(2 * np.pi * t)
        resid = 0.5 * torch.randn(n)
        return trend + season + resid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(-1)  # [seq_len, 1]


# 分解模型架构
class DecompModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        # 时间编码层
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256)
        )

        # 趋势提取网络
        self.trend_net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, 3, padding=1)
        )

        # 季节-残差分解
        self.season_resid = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim * 2, 5, padding=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim * 2, input_dim * 2, 1)
        )

        # 频域动态掩码生成器
        self.freq_mask_gen = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        """前向分解过程
        Args:
            x: 输入序列 [B, L, C]
            t: 时间步 [B]
        Returns:
            T, S, R: 趋势、季节、残差分量
        """
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]

        # --- 时间条件编码 ---
        t_embed = self.time_embed(t.float().view(-1, 1))  # [B, 256]

        # --- 趋势分量预测 ---
        trend = self.trend_net(x)  # [B, C, L]

        # --- 季节残差联合分解 ---
        s_r = self.season_resid(x)  # [B, 2C, 1]
        s_r = s_r.expand(-1, -1, L)  # [B, 2C, L]
        season, resid = torch.chunk(s_r, 2, dim=1)  # 各[B, C, L]

        # --- 动态频域掩码 ---
        freq_weights = torch.sigmoid(self.freq_mask_gen(t_embed))  # [B, 1]
        season = season * freq_weights.view(B, 1, 1)

        return (trend.permute(0, 2, 1),
                season.permute(0, 2, 1),
                resid.permute(0, 2, 1))


# 扩散过程核心实现
class DiffusionTS(nn.Module):
    def __init__(self, model, seq_len, timesteps=1000, beta_schedule='cosine'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.seq_len = seq_len

        # 设置噪声调度
        if beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule()
        else:
            self.betas = torch.linspace(1e-4, 0.02, timesteps)

        # 预计算扩散参数
        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)

    def _cosine_beta_schedule(self, s=0.008):
        """余弦调度（改进稳定性）"""
        t = torch.linspace(0, self.timesteps, self.timesteps + 1)
        alpha_bar = torch.cos((t / self.timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(betas, 0, 0.999)

    def q_sample(self, x0, t, noise=None):
        """闭式解前向扩散过程（向量化实现）
        Args:
            x0: 原始信号 [B, L, C]
            t: 时间步 [B]
        Returns:
            xt: 加噪后的信号
            noise: 使用的噪声
        """
        B, L, C = x0.shape
        device = x0.device

        if noise is None:
            noise = torch.randn_like(x0)

        # 预计算当前步的分解趋势（关键优化点）
        with torch.no_grad():
            T_t, _, _ = self.model(x0, t)

        # 闭式解公式（向量化实现）
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1).to(device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1).to(device)

        xt = sqrt_alpha * (x0 - T_t) + sqrt_one_minus * noise
        return xt, noise, T_t

    def p_loss(self, x0, t, noise=None):
        """计算训练损失（向量化实现）"""
        B = x0.size(0)
        device = x0.device

        # 前向扩散过程
        xt, true_noise, T_t = self.q_sample(x0, t, noise)

        # 模型预测分解
        pred_T, pred_S, pred_R = self.model(xt, t)

        # --- 多任务损失计算 ---
        loss_dict = {}

        # 1. 主重建损失（噪声预测）
        pred_noise = (xt - self.sqrt_alphas_cumprod[t].view(B, 1, 1).to(device) * (x0 - pred_T)) \
                     / self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1).to(device)
        loss_recon = F.mse_loss(pred_noise, true_noise)
        loss_dict['recon'] = loss_recon

        # 2. 趋势一致性损失
        loss_trend = 0.1 * F.mse_loss(pred_T, T_t)
        loss_dict['trend'] = loss_trend

        # 3. 季节项频域约束
        season_fft = torch.fft.rfft(pred_S, dim=1)
        loss_season = 0.01 * torch.mean(torch.abs(season_fft[:, 5:])  ** 2)  # 抑制高频
        loss_dict['season'] = loss_season

        # 4. 残差稀疏性
        loss_resid = 0.01 * torch.mean(torch.abs(pred_R))
        loss_dict['resid'] = loss_resid

        # 总损失加权求和
        total_loss = sum(loss_dict.values())
        return total_loss, loss_dict

    @torch.no_grad()
    def p_sample_step(self, xt, t):
        """单步采样过程"""
        B, L, C = xt.shape
        device = xt.device
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        # 预测当前步分解
        pred_T, pred_S, pred_R = self.model(xt, t_tensor)

        # 计算反向过程参数
        alpha_t = self.alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1. - alpha_t)

        if t == 0:
            z = 0
        else:
            z = torch.randn_like(xt)

        # 逆向过程核心公式
        x_prev = (xt - sqrt_one_minus_alpha_t * pred_T) / sqrt_alpha_t \
                 + sqrt_one_minus_alpha_t * (pred_S + pred_R) \
                 + self.betas[t] * z

        return x_prev.clamp(-5.0, 5.0)  # 数值稳定

    @torch.no_grad()
    def sample(self, num_samples, device):
        """完整采样过程"""
        # 初始噪声
        x_T = torch.randn(num_samples, self.seq_len, 1).to(device)

        # 逆向过程
        xt = x_T
        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling'):
            xt = self.p_sample_step(xt, t)

        # 最终分解
        final_T, final_S, final_R = self.model(xt, torch.zeros(num_samples, device=device, dtype=torch.long))
        return final_T, final_S, final_R


# 训练循环
def train_diffusion(model, dataloader, device, num_epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        progress = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
        for batch in progress:
            x0 = batch.to(device)
            B = x0.size(0)

            # 随机采样时间步
            t = torch.randint(0, model.timesteps, (B,), device=device)

            # 计算损失
            optimizer.zero_grad()
            loss, loss_dict = model.p_loss(x0, t)

            # 反向传播
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            progress.set_postfix({k: f"{v.item():.4f}" for k, v in loss_dict.items()})

        print(f'Epoch {epoch + 1} Average Loss: {total_loss / len(dataloader):.4f}')


if __name__ == "__main__":
    # 配置参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_len = 100
    batch_size = 64
    timesteps = 500

    # 初始化组件
    dataset = TimeSeriesDataset(seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    decomp_model = DecompModel().to(device)
    diffusion_model = DiffusionTS(decomp_model, seq_len=seq_len, timesteps=timesteps).to(device)

    # 训练
    train_diffusion(diffusion_model, dataloader, device, num_epochs=50)

    # 采样示例
    with torch.no_grad():
        T, S, R = diffusion_model.sample(num_samples=batch_size, device=device)
        generated_series = T + S + R
        print("Generated series shape:", generated_series.shape)