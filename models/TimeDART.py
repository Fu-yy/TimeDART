import math

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


import torch
from torch import nn


class STRNetwork(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super(STRNetwork, self).__init__()

        # Parameters
        self.pred_len = pred_len

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream
        # MLP
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # Streams Concatination
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend

        s = s.permute(0, 2, 1)  # to [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # to [Batch, Channel, Input]

        # Channel split for channel independence
        B = s.shape[0]  # Batch size
        C = s.shape[1]  # Channel size
        I = s.shape[2]  # Input size
        s = torch.reshape(s, (B * C, I))  # [Batch and Channel, Input]
        t = torch.reshape(t, (B * C, I))  # [Batch and Channel, Input]

        # Non-linear Stream
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)  # 96 -- 104
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # 224 12 16
        # s: [Batch and Channel, Patch_num, Patch_len]

        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # Linear Stream
        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # Streams Concatination
        # x = torch.cat((s, t), dim=1)
        # x = self.fc8(x)
        #
        # # Channel concatination
        # x = torch.reshape(x, (B, C, self.pred_len))  # [Batch, Channel, Output]
        #
        # x = x.permute(0, 2, 1)  # to [Batch, Output, Channel]

        return t,s



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
        self.forecast_head = nn.Linear(seq_len , pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        b,nvar,l = x.shape
        x = self.flatten(x)  # (batch_size, num_features, seq_len * d_model)
        x = self.forecast_head(x)  # (batch_size, num_features, pred_len)
        x = torch.reshape(x, [-1,l,nvar])
        x = self.dropout(x)  # (batch_size, num_features, pred_len)
        x = x.permute(0, 2, 1)  # (batch_size, pred_len, num_features)
        return x



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
        # x = self.pool(x)  # [batch_size, d_model, 1]
        # x = x.squeeze(-1)  # [batch_size, d_model]
        # trend = x.unsqueeze(1)  # [batch_size, 1, d_model]
        return x.permute(0, 2, 1).contiguous()


class ConditionalEncoding(nn.Module):
    def __init__(self, input_dim, d_model):
        super(ConditionalEncoding, self).__init__()
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


class SeqAttention(nn.Module):
    def __init__(self, target_dim, embed_dim, num_heads=1,dropout=0.1):
        super(SeqAttention, self).__init__()
        self.target_dim = target_dim
        # self.num_lags = num_lags
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 线性变换用于查询、键、值
        # self.query = nn.Linear(target_dim, embed_dim)
        # self.key = nn.Linear(target_dim, embed_dim)
        # self.value = nn.Linear(target_dim, embed_dim)
        #
        # # 多头注意力
        # self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # self.fc = nn.Linear(embed_dim, target_dim)




        # -----

        # self.self_attention = nn.MultiheadAttention(
        #     embed_dim=target_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        # )
        self.norm1 = nn.LayerNorm(target_dim)
        # self.encoder_attention = nn.MultiheadAttention(
        #     embed_dim=target_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        # )
        self.norm2 = nn.LayerNorm(target_dim)
        self.ff = nn.Sequential(
            nn.Linear(target_dim, embed_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*2, target_dim),
        )
        self.norm3 = nn.LayerNorm(target_dim)
        self.dropout = nn.Dropout(dropout)


    def compute_attention(self, q, k, v):
        scores = torch.matmul(q, k) / torch.sqrt(torch.tensor(q.size(-1)))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self,lags):
        # lags: (batch_size, sub_seq_len, target_dim, num_lags)
        batch_size, sub_seq_len, target_dim = lags.size()
        # 重塑为 (batch_size, sub_seq_len * num_lags, target_dim)
        # lags = lags.permute(0,1,3,2).contiguous()
        # lags = torch.reshape(lags, (batch_size, sub_seq_len , target_dim))
        query= lags
        key = lags.permute(0,2,1).contiguous()
        value = lags
        # Self-attention
        # attn_output, _ = self.self_attention(query, query, query, attn_mask=None)
        attn_output,_ = self.compute_attention(query, key, value)

        query = self.norm1(query + self.dropout(attn_output))

        # # Encoder attention
        # attn_output, _ = self.encoder_attention(query, key, value, attn_mask=None)
        # query = self.norm2(query + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(query)
        output = self.norm3(query + self.dropout(ff_output))



        # 重塑回 (batch_size, sub_seq_len, target_dim, num_lags)
        # output = torch.reshape(output, (batch_size, sub_seq_len , num_lags, target_dim)).permute(0, 1, 3, 2)

        return output


class DenoisingConditionDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads=1,dropout=0.1):
        super(DenoisingConditionDecoder, self).__init__()
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




class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )

    def forward(self, t):
        # 输入t的形状: (batch_size,)
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]  # (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (batch_size, embed_dim)
        return self.mlp(emb)  # 输出形状: (batch_size, embed_dim)

class TimeDART(nn.Module):
    """
    TimeDART
    """

    def __init__(self, configs):
        super(TimeDART, self).__init__()
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


        # Patch
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 1



        sos_token = torch.randn(1, 1, self.d_model, device=self.device)




        # 条件编码模块
        # 假设条件信息维度为 configs.condition_dim
        # self.conditional_encoding = ConditionalEncoding(
        #     input_dim=self.d_model,
        #     d_model=self.d_model
        # )
        # 在 Model 类中替换 ConditionalEncoding
        # self.conditional_encoding_att = SeqAttention(
        #     target_dim=self.d_model,
        #     num_heads=self.num_heads,
        #     embed_dim=self.d_model,
        #     dropout=self.dropout,
        # )
        # Decoder
        # if self.task_name == "pretrain":
        #     self.denoising_patch_decoder = DenoisingPatchDecoder(
        #         d_model=configs.d_model,
        #         num_layers=configs.d_layers,
        #         num_heads=configs.n_heads,
        #         feedforward_dim=configs.d_ff,
        #         dropout=configs.dropout,
        #     )
        #
        #
        #     self.projection = nn.ModuleList(
        #         [FlattenHead(
        #         seq_len=self.seq_len // (configs.down_sampling_window ** i),
        #         d_model=self.d_model,
        #         pred_len=configs.input_len,
        #         dropout=configs.head_dropout,
        #     )
        #             for i in range(configs.down_sampling_layers + 1)
        #         ]
        #     )
        #     self.regression = nn.ModuleList([
        #         nn.Linear(self.input_len // (configs.down_sampling_window ** i), self.input_len)
        #         for i in range(configs.down_sampling_layers + 1)
        #     ])
        #     self.regression = nn.ModuleList([
        #         nn.Linear(self.input_len, self.input_len)
        #         for i in range(configs.down_sampling_layers + 1)
        #     ])
        #
        #
        #
        # elif self.task_name == "finetune":
        #     # self.head = FlattenHead(
        #     #     seq_len=self.seq_len,
        #     #     d_model=configs.d_model,
        #     #     pred_len=configs.pred_len,
        #     #     dropout=configs.head_dropout,
        #     # )
        #
        #     self.head = nn.ModuleList(
        #         [FlattenHead(
        #             seq_len=self.seq_len // (configs.down_sampling_window ** i),
        #             d_model=self.d_model,
        #             pred_len=configs.pred_len,
        #             dropout=configs.head_dropout,
        #         )
        #             for i in range(configs.down_sampling_layers + 1)
        #         ]
        #     )
        #
        #     # self.regression = nn.Linear(self.input_len, configs.pred_len)
        #     self.regression = nn.ModuleList([
        #         nn.Linear(self.input_len // (configs.down_sampling_window ** i), configs.pred_len)
        #         for i in range(configs.down_sampling_layers + 1)
        #     ])
        #
        #     self.head = FlattenHead(
        #             seq_len=self.seq_len,
        #             d_model=self.d_model,
        #             pred_len=configs.pred_len,
        #             dropout=configs.head_dropout,
        #         )
        #
        #
        #     self.regression = nn.ModuleList([
        #         nn.Linear(self.input_len, configs.pred_len)
        #         for i in range(configs.down_sampling_layers + 1)
        #     ])
        #
        #
        # self.decomp_multi = series_decomp(95)
        # self.denoise_layers_num = configs.denoise_layers_num
        # self.denoise_layers = nn.ModuleList([
        #     DenoisingPatchDecoder(
        #         d_model=configs.d_model,
        #         num_layers=configs.d_layers,
        #         num_heads=configs.n_heads,
        #         feedforward_dim=configs.d_ff,
        #         dropout=configs.dropout,
        #     )
        #     for _ in range(self.denoise_layers_num)
        # ])
        #
        # self.denoise_layers_cond = nn.ModuleList([
        #     DenoisingConditionDecoder(
        #         embed_dim=configs.d_model,
        #         num_heads=configs.n_heads,
        #         dropout=configs.dropout,
        #     )
        #     for _ in range(self.denoise_layers_num)
        # ])




        # -----------------------------------------------------------------------------

        # 时间步嵌入
        self.time_embed = TimeEmbedding(configs.d_model)
        # self.time_embed = nn.Embedding(1, configs.d_model)
        self.input_proj = nn.Linear(1, configs.d_model)
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(self.seq_len, configs.d_model))

        # 共享特征提取（谱归一化保证Lipschitz连续性）
        self.encoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(1, 64, 5, padding=2)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv1d(64, configs.d_model, 5, padding=2))
        )

        # 趋势预测头
        self.trend_head = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(configs.d_model, 64, 3, padding=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv1d(64, 1, 3, padding=1))
        )

        # 季节-残差分离头
        self.season_resid_head = nn.Sequential(
            nn.Conv1d(configs.d_model, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 2, 3, padding=1)
        )

        # 动态频率预测器
        self.freq_predictor = nn.Linear(configs.d_model, 1)
        self.str_net = STRNetwork(seq_len=self.input_len,pred_len=self.input_len,patch_len=self.patch_len,stride=self.stride,padding_patch='emd')
        self.resid_linear = nn.Linear(self.input_len, self.input_len)

        self.flat_head_trend = FlattenHead(
            seq_len=self.input_len,
            d_model=self.d_model,
            pred_len=configs.input_len,
            dropout=configs.head_dropout,
        )
        self.flat_head_season = FlattenHead(
            seq_len=self.input_len,
            d_model=self.d_model,
            pred_len=configs.input_len,
            dropout=configs.head_dropout,
        )
        self.flat_head_residual = FlattenHead(
            seq_len=self.input_len,
            d_model=self.d_model,
            pred_len=configs.input_len,
            dropout=configs.head_dropout,
        )
        # -----------------------------------------------------------------------------
    def decomp_func(self,x,t):

        x = x.permute(0, 2, 1)
        batch_size, input_len, num_features = x.size()
        means = torch.mean(
            x, dim=1, keepdim=True
        ).detach()  # [batch_size, 1, num_features], detach from gradient
        x = x - means  # [batch_size, input_len, num_features]
        stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # [batch_size, 1, num_features]
        x = x / stdevs  # [batch_size, input_len, num_features]
        x = x.permute(0,2,1)

        B,_,seqlen=x.shape
        # 时间嵌入
        t_emb = self.time_embed(t)
        # 特征提取
        feat = self.encoder(x)

        feat = feat + t_emb.unsqueeze(-1)
        # 趋势预测
        trend = self.trend_head(feat)


        # 季节项计算：x - trend，并施加频域约束
        # 动态频率掩码
        freq_weights = torch.sigmoid(self.freq_predictor(feat.mean(dim=2)))
        season = x - trend
        season_fft = torch.fft.rfft(season, dim=-1)
        H_low = self._generate_lowpass_mask(freq_weights)
        season_filtered = torch.fft.irfft(season_fft * H_low, n=self.input_len)

        # 残差计算：剩余部分
        resid = x - trend - season_filtered


        trend,season_filtered,resid = self.inverse_transform(trend.permute(0,2,1),season_filtered.permute(0,2,1),resid.permute(0,2,1), means,
                                                                        stdevs)
        return trend.permute(0,2,1),season_filtered.permute(0,2,1),resid.permute(0,2,1)


    def _generate_lowpass_mask(self, freq_weights):
        B = freq_weights.size(0)
        F = self.input_len // 2 + 1

        # 生成频率索引矩阵 [B, F]
        freq_indices = torch.arange(F, device=freq_weights.device).view(1, F).expand(B, F)

        # 计算截止频率索引 [B, 1]
        cutoff = (freq_weights * F).long().clamp(0, F - 1)

        # 向量化比较生成掩码 [B, F]
        mask = (freq_indices < cutoff).float()

        # 添加通道维度 [B, 1, F]
        return mask.unsqueeze(1)

    def inverse_transform(self,trend_norm, season_norm, resid_norm, scaler_mean, scaler_std):
        trend_raw = trend_norm * (scaler_std[:, 0, :].unsqueeze(1)).repeat(
            1, self.input_len, 1
        ) + (scaler_mean[:, 0, :].unsqueeze(1)).repeat(
            1, self.input_len, 1
        )
        season_raw = season_norm * (scaler_std[:, 0, :].unsqueeze(1)).repeat(
            1, self.input_len, 1
        )
        residual_raw = resid_norm * (scaler_std[:, 0, :].unsqueeze(1)).repeat(
            1, self.input_len, 1
        )
        return trend_raw, season_raw, residual_raw

    def predit(self, trend,season,resid):
        batch_size,n, input_len  = trend.shape
        trend = trend.permute(0, 2, 1)
        season = season.permute(0, 2, 1)
        resid = resid.permute(0, 2, 1)

        res_t, res_s = self.str_net(trend, season)
        res_t = torch.reshape(res_t, (batch_size, input_len, 1))
        res_s = torch.reshape(res_s, (batch_size, input_len, 1))
        resid = self.resid_linear(resid.permute(0, 2, 1)).permute(0, 2, 1)

        flat_trend = self.flat_head_trend(res_t)
        flat_season = self.flat_head_season(res_s)
        flat_residual = self.flat_head_residual(resid)

        return flat_trend ,flat_season , flat_residual

    def pretrain(self, x,t):

        x = x.permute(0,2,1)
        batch_size, input_len, num_features = x.size()
        means = torch.mean(
            x, dim=1, keepdim=True
        ).detach()  # [batch_size, 1, num_features], detach from gradient
        x = x - means  # [batch_size, input_len, num_features]
        stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # [batch_size, 1, num_features]
        x = x / stdevs  # [batch_size, input_len, num_features]

        trend,season, resid = self.decomp_func(x.permute(0,2,1),t)
        flat_trend, flat_season, flat_residual = self.predit(trend,season,resid)
        flat_trend, flat_season, flat_residual = self.inverse_transform(flat_trend, flat_season, flat_residual,means,stdevs)
        predict_x = flat_trend.permute(0,2,1) + flat_season.permute(0,2,1) + flat_residual.permute(0,2,1)

        return predict_x,flat_trend ,flat_season , flat_residual

    def forecast(self, x,x_mark):
        # x = torch.fft.fft(x,dim=-2).real


        batch_size, _, num_features = x.size()
        means = torch.mean(x, dim=1, keepdim=True).detach()
        x = x - means
        stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / stdevs
        x, trend = self.decomp_multi(x)
        # x, trend = x,x
        x = self.channel_independence[0](x)  # [batch_size * num_features, input_len, 1]
        x = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

        # x = torch.fft.fft(x,dim=-2).imag
        x = self.enc_embedding(x)  # [batch_size * num_features, seq_len, d_model]


        # --------------------------- 添加条件 begin
        # 获取条件编码
        # cond_encoded = self.conditional_encoding(x)  # [batch_size, 1, d_model]
        # # 扩展条件编码以匹配批次和特征维度
        # cond_encoded = cond_encoded.repeat_interleave(x.size(0) // cond_encoded.size(0),
        #                                               dim=0)  # [batch_size * num_features, 1, d_model]
        # cond_encoded = cond_encoded.expand(-1, x.size(1), -1)  # [batch_size * num_features, seq_len, d_model]
        #
        # # 将条件编码添加到嵌入中
        # x = x + cond_encoded  # 结合条件编码

        # --------------------------- 添加条件 end




        x = self.positional_encoding(x)  # [batch_size * num_features, seq_len, d_model]

        x, _ = self.decomp_multi(x)
        # x, _ = x,x

        x = self.encoder(
            x,
            is_mask=False,
        )  # [batch_size * num_features, seq_len, d_model]
        x = x.reshape(
            batch_size, num_features, -1, self.d_model
        )  # [batch_size, num_features, seq_len, d_model]
        # x = torch.fft.ifft(x,dim=-2).real
        # forecast
        x = self.head(x)  # [bs, pred_len, n_vars]
        x = x + self.regression[0](trend.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # denormalization
        x = x * (stdevs[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
        x = x + (means[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)


        return x

    def forward(self, batch_x,t):

        if self.task_name == "pretrain":
            return self.pretrain(batch_x,t)
        elif self.task_name == "finetune":
            dec_out = self.forecast(batch_x,t)
            return dec_out[:, -self.pred_len: , :]
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
