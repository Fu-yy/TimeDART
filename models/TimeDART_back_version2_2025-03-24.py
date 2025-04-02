import torch
import torch.nn as nn
from einops import rearrange, repeat
import pandas as pd

from layers.Autoformer_EncDec import moving_avg, series_decomp
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


class ConditionalEncoding_old(nn.Module):
    def __init__(self, input_dim, d_model):
        super(ConditionalEncoding_old, self).__init__()
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

class DenoisingConditionDecoder_old(nn.Module):
    def __init__(self, embed_dim, num_heads=1,dropout=0.1):
        super(DenoisingConditionDecoder_old, self).__init__()
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

class MultiScaleTrendExtractor(nn.Module):
    def __init__(self, input_dim, d_model, scales=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, d_model, k, padding=k // 2)
            for k in scales
        ])
        self.attn = nn.Sequential(
            nn.Linear(len(scales) * d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, len(scales)),
            nn.Softmax(dim=-1)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]

        # 多尺度特征
        features = [conv(x) for conv in self.convs]  # 各尺度特征形状 [B, d_model, L]

        # 自适应融合
        global_features = torch.cat([self.pool(f).squeeze(-1) for f in features], dim=-1)  # [B, scales*d_model]
        attn_weights = self.attn(global_features)  # [B, scales]

        # 拆分权重并按维度扩展
        fused = sum(
            w.unsqueeze(-1).unsqueeze(-1) * f  # w形状 [B,1,1], f形状 [B,d_model,L]
            for w, f in zip(attn_weights.unbind(dim=1), features)
        )

        pooled = self.pool(fused)  # [B, d_model, 1]
        return pooled.permute(0, 2, 1)  # [B, 1, d_model]
class ConditionalEncoding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.trend_extractor = MultiScaleTrendExtractor(input_dim, d_model)
        self.condition_proj = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        trend = self.trend_extractor(x)  # [B, 1, d_model]
        proj_trend = self.condition_proj(trend)
        gate = self.gate(proj_trend)
        return proj_trend * gate  # 自适应特征选择
class AdaptiveFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # SE式通道门控
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 16),
            nn.GELU(),
            nn.Linear(d_model // 16, d_model),
            nn.Sigmoid()
        )
        # 深度可分离卷积 + GELU
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, noise_x, cond):
        # 通道融合
        channel_weights = self.channel_gate(cond.permute(0, 2, 1))  # 输入需调整为 [B,D,L]
        fused = noise_x * channel_weights.unsqueeze(1)  # [B,L,D] * [B,1,D]

        # 空间融合
        spatial_out = self.spatial_conv(fused.permute(0, 2, 1)).permute(0, 2, 1)
        fused = fused + spatial_out
        return self.norm(fused)


class DenoisingConditionDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.fusion = AdaptiveFusion(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Noise_x, X, cond):
        # 条件融合
        fused = self.fusion(Noise_x, cond)

        # 多头注意力
        attn_output, _ = self.attn(fused, X, X)
        attn_output = self.dropout(attn_output)
        attn_output = fused + attn_output
        attn_output = self.norm1(attn_output)

        # 前馈网络
        ff_output = self.ff(attn_output)
        output = attn_output + ff_output
        return self.norm2(output)





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

        self.channel_independence = ChannelIndependence(
                input_len=self.input_len,
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

        # 条件编码模块
        # 假设条件信息维度为 configs.condition_dim
        self.conditional_encoding = ConditionalEncoding(
            input_dim=self.d_model,
            d_model=self.d_model
        )


        # Decoder
        if self.task_name == "pretrain":


            self.projection =FlattenHead(
                seq_len=self.seq_len,
                d_model=self.d_model,
                pred_len=configs.input_len,
                dropout=configs.head_dropout,
            )
            self.regression = nn.Linear(self.input_len, self.input_len)
            self.ff = nn.Sequential(
                nn.Linear(self.input_len, self.input_len),
                nn.ReLU(),
                nn.Linear(self.input_len, self.input_len),
            )


        elif self.task_name == "finetune":

            self.head = FlattenHead(
                    seq_len=self.seq_len,
                    d_model=self.d_model,
                    pred_len=configs.pred_len,
                    dropout=configs.head_dropout,
                )


            self.regression = nn.Linear(self.input_len, configs.pred_len)
            self.ff = nn.Sequential(
                nn.Linear(self.pred_len, self.pred_len),
                nn.ReLU(),
                nn.Linear(self.pred_len, self.pred_len),
            )


        self.decomp_multi = series_decomp(25)
        self.decomp_multi_learnable = LearnableMultiScaleDecomp(self.configs.c_out)
        self.decomp_multi_learnable_second = LearnableMultiScaleDecomp(self.patch_len,scales=[5, 13, 25])
        self.decomp_multi_learnable_third = LearnableMultiScaleDecomp(self.d_model,scales=[5, 13, 25])
        self.denoise_layers_num = configs.denoise_layers_num


        self.denoise_layers_cond = nn.ModuleList([
            DenoisingConditionDecoder(
                embed_dim=configs.d_model,
                num_heads=configs.n_heads,
                dropout=configs.dropout,
            )
            for _ in range(self.denoise_layers_num)
        ])

        # 自适应可学习权重参数
        self.log_var_freq = nn.Parameter(torch.zeros(1))
        self.log_var_orth = nn.Parameter(torch.zeros(1))
        self.log_var_smooth = nn.Parameter(torch.zeros(1))

    def pretrain(self, x,x_mask,i=0):

        # [batch_size, input_len, num_features]
        batch_size, input_len, num_features = x.size()
        means = torch.mean(
            x, dim=1, keepdim=True
        ).detach()  # [batch_size, 1, num_features], detach from gradient
        x = x - means  # [batch_size, input_len, num_features]
        stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # [batch_size, 1, num_features]
        x = x / stdevs  # [batch_size, input_len, num_features]

        # 分解  1
        x, trend,freq_loss,orth_loss,smoothness = self.decomp_multi_learnable(x)
        # Channel Independence
        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

        # For Casual Transformer
        x_embedding = self.enc_embedding(
            x_patch
        )  # [batch_size * num_features, seq_len, d_model]

        x_embedding_bias = self.add_sos_token_and_drop_last(
            x_embedding
        )  # [batch_size * num_features, seq_len, d_model]

        x_embedding_bias = self.positional_encoding(x_embedding_bias)

        x_out = self.encoder(
            x_embedding_bias,
            is_mask=True,
        )  # [batch_size * num_features, seq_len, d_model]


        freq_loss_inner_list = []
        orth_loss_inner_list = []
        smoothness_inner_list = []
        for layer in self.denoise_layers_cond:
            x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
            x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

            noise_x_patch, _, _ = self.diffusion(
                x_patch
            )  # [batch_size * num_features, seq_len, patch_len]

            noise_x_embedding = self.enc_embedding(
                noise_x_patch
            )  # [batch_size * num_features, seq_len, d_model]
            noise_x_embedding = self.positional_encoding(noise_x_embedding)
            denoise_input = noise_x_embedding

            # 分解  5
            x_out, x_out_trend,freq_loss_inner,orth_loss_inner,smoothness_inner = self.decomp_multi_learnable_third(x_out)
            cond_encoded = self.conditional_encoding(x_out_trend)  # [batch_size, 1, d_model]
            # For Denoising Patch Decoder
            denoise_out = layer(
                Noise_x=denoise_input,
                X=x_out,
                cond=cond_encoded
            )  # [batch_size * num_features, seq_len, d_model]
            freq_loss_inner_list.append(freq_loss_inner)
            orth_loss_inner_list.append(orth_loss_inner)
            smoothness_inner_list.append(smoothness_inner)
            # denoise_input = denoise_out+denoise_input  # 增加残差
            denoise_out = denoise_out.reshape(
                batch_size, num_features, -1, self.d_model
            )  # [batch_size, num_features, seq_len, d_model]
            denoise_out = self.projection(denoise_out)  # [batch_size, input_len, num_features]
            x = denoise_out
            predict_x = denoise_out
        predict_x = predict_x + self.regression(trend.permute(0,2,1)).permute(0,2,1).contiguous()
        # predict_x = self.ff(predict_x.permute(0, 2, 1)).permute(0, 2, 1)
        # Instance Denormalization
        predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
            1, input_len, 1
        )  # [batch_size, input_len, num_features]
        predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
            1, input_len, 1
        )  # [batch_size, input_len, num_features]

        total_freq = freq_loss + sum(freq_loss_inner_list)
        total_orth = orth_loss + sum(orth_loss_inner_list)
        total_smooth = smoothness + sum(smoothness_inner_list)
        return predict_x,total_freq,total_orth,total_smooth

    def forecast(self, x,x_mark):
        batch_size, _, num_features = x.size()
        means = torch.mean(x, dim=1, keepdim=True).detach()
        x = x - means
        stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / stdevs
        x, trend,freq_loss,orth_loss,smoothness = self.decomp_multi_learnable(x)
        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        x = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]
        x = self.enc_embedding(x)  # [batch_size * num_features, seq_len, d_model]

        x = self.positional_encoding(x)  # [batch_size * num_features, seq_len, d_model]

        x = self.encoder(
            x,
            is_mask=False,
        )  # [batch_size * num_features, seq_len, d_model]
        x = x.reshape(
            batch_size, num_features, -1, self.d_model
        )  # [batch_size, num_features, seq_len, d_model]
        # forecast
        x = self.head(x)  # [bs, pred_len, n_vars]
        x = x + self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        # x = self.ff(x.permute(0, 2, 1)).permute(0, 2, 1)
        # denormalization
        x = x * (stdevs[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
        x = x + (means[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)


        return x,freq_loss,orth_loss,smoothness

    def forward(self, batch_x,x_mask,i=0):

        if self.task_name == "pretrain":
            return self.pretrain(batch_x,x_mask,i)
        elif self.task_name == "finetune":
            dec_out,freq_loss,orth_loss,smoothness = self.forecast(batch_x,x_mask)
            return dec_out[:, -self.pred_len: , :],freq_loss,orth_loss,smoothness
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

    configs = parser.parse_args()

    return configs



if __name__ == '__main__':
    configs = get_config()

    configs.task_name = 'pretrain'

    configs.seq_len = 336
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
    configs.input_len = 336

    configs.down_sampling_layers = 2
    configs.down_sampling_window = 2

    x= torch.randn(1,336,7)
    x_mark_enc= torch.randn(16,336,4)
    x_res= torch.randn(16,336,7)


    configs.device = x.device
    model = Model(configs)
    mask = torch.ones_like(x)
    # # x_enc 64 336 7 ; x_mark_enc 16 336 4 ； batch_x 16 336 7  mask 64 336 7
    c = model(x,x_mark_enc)
    d = 'end'
