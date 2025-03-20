import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class TimeSeriesDataset(Dataset):
    def __init__(self, seq_len=100, pred_len=24, num_samples=10000):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = torch.stack([self._generate_synthetic_series(seq_len + pred_len)
                                 for _ in range(num_samples)])

    def _generate_synthetic_series(self, n):
        """生成包含历史序列和预测目标的合成数据"""
        t = torch.linspace(0, 10, n)
        trend = 0.3 * t
        season = 2 * torch.sin(2 * np.pi * t)
        resid = 0.5 * torch.randn(n)
        full_series = trend + season + resid
        return full_series  # [total_len]

    def __getitem__(self, idx):
        full_series = self.data[idx]
        # 前seq_len作为输入，后pred_len作为目标
        return full_series[:self.seq_len].unsqueeze(-1), \
            full_series[self.seq_len:].unsqueeze(-1)  # [seq_len,1], [pred_len,1]
    def __len__(self):
        return len(self.data)


class DecompModel(nn.Module):
    def __init__(self, input_dim=1, pred_len=24, hidden_dim=64):
        super().__init__()
        self.pred_len = pred_len

        # 时间编码层（新增预测长度条件）
        self.time_embed = nn.Sequential(
            nn.Linear(2, 128),  # 同时编码时间步和预测长度
            nn.SiLU(),
            nn.Linear(128, 256)
        )

        # 趋势预测网络（输出预测长度维度）
        self.trend_net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, 3, padding=1),
            nn.AdaptiveAvgPool1d(pred_len)  # 关键修改：输出预测长度
        )

        # 季节-残差分解（生成预测长度序列）
        self.season_resid = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim * 2, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, input_dim * 2, 3, padding=1),
            nn.AdaptiveAvgPool1d(pred_len)  # 关键修改：输出预测长度
        )

    def forward(self, x, t):
        """输入x: [B, seq_len, C] 输出: [B, pred_len, C]"""
        B, seq_len, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, seq_len]

        # 时间条件编码（新增预测长度信息）
        pred_len_tensor = torch.full((B,), self.pred_len, device=x.device).float()
        t_embed = self.time_embed(torch.stack([t.float(), pred_len_tensor], dim=1))  # [B, 256]

        # 趋势分量预测
        trend = self.trend_net(x)  # [B, C, pred_len]

        # 季节残差联合分解
        s_r = self.season_resid(x)  # [B, 2C, pred_len]
        season, resid = torch.chunk(s_r, 2, dim=1)  # 各[B, C, pred_len]

        return (trend.permute(0, 2, 1),  # [B, pred_len, C]
                season.permute(0, 2, 1),
                resid.permute(0, 2, 1))


class DiffusionTS(nn.Module):
    def __init__(self, model, seq_len, pred_len, n_vars=1, timesteps=500,devices='cpu'):
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars

        # 噪声调度（余弦调度）
        self.betas = self._cosine_beta_schedule(timesteps)
        self.timesteps = timesteps

        # 预计算扩散参数
        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(devices)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(devices)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(devices)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        t = torch.linspace(0, timesteps, timesteps + 1)
        alpha_bar = torch.cos((t / timesteps + s) / (1 + s) * torch.pi * 0.5)  ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(betas, 0, 0.999)

    def q_sample(self, x0, t, noise=None):
        """前向扩散过程（预测长度维度）"""
        B = x0.size(0)
        device = x0.device

        if noise is None:
            noise = torch.randn(B, self.pred_len, self.n_vars, device=device)

        # 动态趋势预测（关键修改：输入历史序列，输出预测趋势）
        with torch.no_grad():
            T_t, _, _ = self.model(x0, t)  # T_t: [B, pred_len, C]

        # 闭式解扩散公式
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)

        # 注意：x0是目标序列（预测长度维度）
        xt = sqrt_alpha * (x0 - T_t) + sqrt_one_minus * noise
        return xt, noise, T_t

    def p_loss(self, hist_series, target_series, t):
        """训练损失计算（历史序列->预测目标）"""
        B = hist_series.size(0)
        device = hist_series.device

        # 前向扩散（目标序列维度为pred_len）
        xt, true_noise, T_t = self.q_sample(target_series, t)

        # 模型预测（输入历史序列，输出预测分解）
        pred_T, pred_S, pred_R = self.model(hist_series, t)

        # 多任务损失计算
        loss_dict = {}

        # 1. 噪声预测损失（预测长度维度）
        pred_noise = (xt - self.sqrt_alphas_cumprod[t].view(B, 1, 1) * (target_series - pred_T)) \
                     / self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        loss_recon = F.mse_loss(pred_noise, true_noise)
        loss_dict['recon'] = loss_recon

        # 2. 趋势对齐损失
        loss_trend = 0.1 * F.mse_loss(pred_T, T_t)
        loss_dict['trend'] = loss_trend

        # 3. 季节项周期一致性
        season_fft = torch.fft.rfft(pred_S, dim=1)
        loss_season = 0.01 * torch.mean(torch.abs(season_fft[:, 5:])  ** 2)
        loss_dict['season'] = loss_season

        # 4. 残差稀疏性
        loss_resid = 0.01 * torch.mean(torch.abs(pred_R))
        loss_dict['resid'] = loss_resid

        return sum(loss_dict.values()), loss_dict

    @torch.no_grad()
    def p_sample_step(self, hist_series, xt, t):
        """单步采样（输入历史，输出预测）"""
        B = hist_series.size(0)
        device = hist_series.device

        # 预测分解项（关键：模型接收历史序列）
        pred_T, pred_S, pred_R = self.model(hist_series, t)

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

        return x_prev.clamp(-5.0, 5.0)

    @torch.no_grad()
    def sample(self, hist_series, num_samples=1):
        """完整采样过程（输入历史，生成预测）"""
        device = hist_series.device
        B, seq_len, C = hist_series.shape

        # 初始噪声（预测长度维度）
        x_T = torch.randn(B, self.pred_len, C).to(device)

        # 逆向过程
        xt = x_T
        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling'):
            xt = self.p_sample_step(hist_series, xt, t)

        # 最终分解结果
        final_T, final_S, final_R = self.model(hist_series,
                                               torch.zeros(B, device=device, dtype=torch.long))
        return final_T + final_S + final_R


def train_diffusion(model, dataloader, device, num_epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
        for hist, target in progress:
            hist, target = hist.to(device), target.to(device)
            B = hist.size(0)

            # 随机时间步
            t = torch.randint(0, model.timesteps, (B,), device=device)

            # 梯度更新
            optimizer.zero_grad()
            loss, loss_dict = model.p_loss(hist, target, t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            progress.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})

        scheduler.step()
        print(f'Epoch {epoch + 1} | Loss: {total_loss / len(dataloader):.4f} | LR: {scheduler.get_last_lr()[0]:.2e}')


if __name__ == "__main__":
    # 参数配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_len = 100
    pred_len = 24
    n_vars = 1
    batch_size = 64

    # 初始化组件
    dataset = TimeSeriesDataset(seq_len=seq_len, pred_len=pred_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    decomp_model = DecompModel(pred_len=pred_len).to(device)
    diffusion_model = DiffusionTS(decomp_model, seq_len=seq_len,
                                  pred_len=pred_len, n_vars=n_vars,devices=device).to(device)

    # 训练
    train_diffusion(diffusion_model, dataloader, device)

    # 测试采样
    test_hist = torch.randn(5, seq_len, n_vars).to(device)  # 示例输入
    with torch.no_grad():
        pred = diffusion_model.sample(test_hist)
        print(f"Generated prediction shape: {pred.shape}")  # 应输出 [5, 24, 1]