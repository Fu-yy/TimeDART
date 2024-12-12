import torch
import torch.nn as nn
from einops import rearrange, repeat

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

class ComplexFrequencyCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparsity_threshold=0.01):
        """
        Args:
            embed_dim: 输入特征维度（频域嵌入维度）。
            num_heads: 注意力头数。
            sparsity_threshold: 稀疏化阈值，用于过滤无关频率分量。
        """
        super(ComplexFrequencyCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparsity_threshold = sparsity_threshold

        # Query, Key, Value 分别对两个输入进行计算
        self.real_q_proj = nn.Linear(embed_dim, embed_dim)
        self.real_k_proj = nn.Linear(embed_dim, embed_dim)
        self.real_v_proj = nn.Linear(embed_dim, embed_dim)

        self.imag_q_proj = nn.Linear(embed_dim, embed_dim)
        self.imag_k_proj = nn.Linear(embed_dim, embed_dim)
        self.imag_v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj_real = nn.Linear(embed_dim, embed_dim)
        self.out_proj_imag = nn.Linear(embed_dim, embed_dim)

        # 缩放因子
        self.scale = (embed_dim // num_heads) ** -0.5

    def forward(self, x1, x2):
        """
        Args:
            x1: [B, T, D] - 输入 1 的时域数据 (Batch, Time, Dimension)
            x2: [B, T, D] - 输入 2 的时域数据 (Batch, Time, Dimension)
        Returns:
            输出经过频域交叉注意力的张量 [B, T, D]
        """
        B, T, D = x1.shape

        # 转换到频域
        freq_x1 = torch.fft.rfft(x1, dim=1, norm='ortho')  # [B, F, D]
        freq_x2 = torch.fft.rfft(x2, dim=1, norm='ortho')  # [B, F, D]
        F = freq_x1.shape[1]  # 频域分量数

        # 分离实部和虚部
        real_x1, imag_x1 = freq_x1.real, freq_x1.imag  # [B, F, D]
        real_x2, imag_x2 = freq_x2.real, freq_x2.imag  # [B, F, D]

        # 分别计算 Query, Key, Value
        real_Q = self.real_q_proj(real_x1)  # [B, F, D]
        real_K = self.real_k_proj(real_x2)  # [B, F, D]
        real_V = self.real_v_proj(real_x2)  # [B, F, D]

        imag_Q = self.imag_q_proj(imag_x1)  # [B, F, D]
        imag_K = self.imag_k_proj(imag_x2)  # [B, F, D]
        imag_V = self.imag_v_proj(imag_x2)  # [B, F, D]

        # 实部和虚部的注意力权重计算
        attn_real = torch.matmul(real_Q, real_K.transpose(-2, -1)) * self.scale  # [B, F, F]
        attn_imag = torch.matmul(imag_Q, imag_K.transpose(-2, -1)) * self.scale  # [B, F, F]

        # 合并注意力权重并稀疏化
        attn_weights = attn_real + attn_imag  # [B, F, F]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = torch.where(attn_weights > self.sparsity_threshold, attn_weights, torch.zeros_like(attn_weights))

        # 实部和虚部的加权值计算
        out_real = torch.matmul(attn_weights, real_V)  # [B, F, D]
        out_imag = torch.matmul(attn_weights, imag_V)  # [B, F, D]

        # 输出投影
        out_real = self.out_proj_real(out_real)  # [B, F, D]
        out_imag = self.out_proj_imag(out_imag)  # [B, F, D]

        # 合并实部和虚部为复数
        freq_out = torch.complex(out_real, out_imag)

        # 转换回时域
        time_out = torch.fft.irfft(freq_out, n=T, dim=1, norm='ortho')  # [B, T, D]

        return time_out
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


        query= fused
        key = X.permute(0,2,1).contiguous()
        value = X
        attn_output,_ = self.compute_attention(query, key, value)

        query = self.norm1(query + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(query)
        output = self.norm2(query + self.dropout(ff_output))
        return output






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

        self.channel_independence = nn.ModuleList(
            [ChannelIndependence(
                input_len=self.input_len // (configs.down_sampling_window ** i),
            )
                for i in range(configs.down_sampling_layers + 1)
            ]
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
        # 在 Model 类中替换 ConditionalEncoding
        # self.conditional_encoding_att = SeqAttention(
        #     target_dim=self.d_model,
        #     num_heads=self.num_heads,
        #     embed_dim=self.d_model,
        #     dropout=self.dropout,
        # )
        # Decoder
        if self.task_name == "pretrain":
            self.denoising_patch_decoder = DenoisingPatchDecoder(
                d_model=configs.d_model,
                num_layers=configs.d_layers,
                num_heads=configs.n_heads,
                feedforward_dim=configs.d_ff,
                dropout=configs.dropout,
            )


            self.denoising_patch_decoder_frequency = ComplexFrequencyCrossAttention(
                embed_dim=configs.d_model, num_heads=configs.n_heads
            )


            self.projection = nn.ModuleList(
                [FlattenHead(
                seq_len=self.seq_len // (configs.down_sampling_window ** i),
                d_model=self.d_model,
                pred_len=configs.input_len,
                dropout=configs.head_dropout,
            )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            self.regression = nn.ModuleList([
                nn.Linear(self.input_len // (configs.down_sampling_window ** i), self.input_len)
                for i in range(configs.down_sampling_layers + 1)
            ])
            self.regression = nn.ModuleList([
                nn.Linear(self.input_len, self.input_len)
                for i in range(configs.down_sampling_layers + 1)
            ])



        elif self.task_name == "finetune":
            # self.head = FlattenHead(
            #     seq_len=self.seq_len,
            #     d_model=configs.d_model,
            #     pred_len=configs.pred_len,
            #     dropout=configs.head_dropout,
            # )

            self.head = nn.ModuleList(
                [FlattenHead(
                    seq_len=self.seq_len // (configs.down_sampling_window ** i),
                    d_model=self.d_model,
                    pred_len=configs.pred_len,
                    dropout=configs.head_dropout,
                )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            # self.regression = nn.Linear(self.input_len, configs.pred_len)
            self.regression = nn.ModuleList([
                nn.Linear(self.input_len // (configs.down_sampling_window ** i), configs.pred_len)
                for i in range(configs.down_sampling_layers + 1)
            ])

            self.head = FlattenHead(
                    seq_len=self.seq_len,
                    d_model=self.d_model,
                    pred_len=configs.pred_len,
                    dropout=configs.head_dropout,
                )


            self.regression = nn.ModuleList([
                nn.Linear(self.input_len, configs.pred_len)
                for i in range(configs.down_sampling_layers + 1)
            ])


        self.merge_linear = nn.Linear(self.d_model * 2, self.d_model)
        self.decomp_multi = series_decomp(95)
        self.denoise_layers_num = configs.denoise_layers_num
        self.denoise_layers = nn.ModuleList([
            DenoisingPatchDecoder(
                d_model=configs.d_model,
                num_layers=configs.d_layers,
                num_heads=configs.n_heads,
                feedforward_dim=configs.d_ff,
                dropout=configs.dropout,
            )
            for _ in range(self.denoise_layers_num)
        ])

        self.denoise_layers_cond = nn.ModuleList([
            DenoisingConditionDecoder(
                embed_dim=configs.d_model,
                num_heads=configs.n_heads,
                dropout=configs.dropout,
            )
            for _ in range(self.denoise_layers_num)
        ])

    def pretrain(self, x,x_mask):

        # [batch_size, input_len, num_features]
        # Instance Normalization

        # x = torch.fft.fft(x,dim=-2).real
        mask_rate = 0.5
        lm=3
        positive_nums=1
        batch_size, input_len, num_features = x.size()
        means = torch.mean(
            x, dim=1, keepdim=True
        ).detach()  # [batch_size, 1, num_features], detach from gradient
        x = x - means  # [batch_size, input_len, num_features]
        stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # [batch_size, 1, num_features]
        x = x / stdevs  # [batch_size, input_len, num_features]


        x, trend = self.decomp_multi(x)
        # Channel Independence
        x = self.channel_independence[0](x)  # [batch_size * num_features, input_len, 1]
        # Patch
        x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]
        # x_patch, trend = self.decomp_multi(x_patch)
        # x_patch_f = torch.fft.fft(x_patch,dim=-2).imag

        # For Casual Transformer
        x_embedding = self.enc_embedding(
            x_patch
        )  # [batch_size * num_features, seq_len, d_model]






        x_embedding_bias = self.add_sos_token_and_drop_last(
            x_embedding
        )  # [batch_size * num_features, seq_len, d_model]

        # --------------------------- 添加条件 begin
        # 获取条件编码
        # cond_encoded = self.conditional_encoding(x_embedding)  # [batch_size, 1, d_model]
        # # 扩展条件编码以匹配批次和特征维度
        # cond_encoded = cond_encoded.repeat_interleave(x_embedding.size(0) // cond_encoded.size(0),
        #                                               dim=0)  # [batch_size * num_features, 1, d_model]
        # cond_encoded = cond_encoded.expand(-1, x_embedding.size(1), -1)  # [batch_size * num_features, seq_len, d_model]
        #
        # # 将条件编码添加到嵌入中
        # x_embedding_bias = x_embedding + cond_encoded  # 结合条件编码

        # --------------------------- 添加条件 end

        x_embedding_bias = self.positional_encoding(x_embedding_bias)
        x_embedding_bias, _ = self.decomp_multi(x_embedding_bias)

        x_out = self.encoder(
            x_embedding_bias,
            is_mask=True,
        )  # [batch_size * num_features, seq_len, d_model]




        res_pred = []
        for layer in self.denoise_layers_cond:
            x = self.channel_independence[0](x)  # [batch_size * num_features, input_len, 1]
            # Patch
            x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

            # item = x_out
            x_patch, default_trend = self.decomp_multi(x_patch)

            noise_x_patch, _, _ = self.diffusion(
                x_patch
            )  # [batch_size * num_features, seq_len, patch_len]


            noise_x_embedding = self.enc_embedding(
                noise_x_patch
            )  # [batch_size * num_features, seq_len, d_model]
            noise_x_embedding = self.positional_encoding(noise_x_embedding)
            # noise end --------------------------
            noise_x_embedding, _ = self.decomp_multi(noise_x_embedding)
            x_out, x_out_trend = self.decomp_multi(x_out)

            # --------------------------- 添加条件 begin
            # 获取条件编码
            # cond_encoded = self.conditional_encoding_att(x_out_trend)  # [batch_size, 1, d_model]
            cond_encoded = self.conditional_encoding(x_out_trend)  # [batch_size, 1, d_model]
            # 扩展条件编码以匹配批次和特征维度
            # cond_encoded = cond_encoded.repeat_interleave(x_out_trend.size(0) // cond_encoded.size(0),
            #                                               dim=0)  # [batch_size * num_features, 1, d_model]
            # cond_encoded = cond_encoded.expand(-1, x_out_trend.size(1), -1)  # [batch_size * num_features, seq_len, d_model]

            # 将条件编码添加到嵌入中

            # --------------------------- 添加条件 end


            # For Denoising Patch Decoder
            denoise_out = layer(
                Noise_x=noise_x_embedding,
                X=x_out,
                cond=cond_encoded
            )  # [batch_size * num_features, seq_len, d_model]



            denoise_out = denoise_out.reshape(
                batch_size, num_features, -1, self.d_model
            )  # [batch_size, num_features, seq_len, d_model]
            denoise_out = self.projection[0](denoise_out)  # [batch_size, input_len, num_features]
            x = denoise_out


            predict_x = denoise_out + self.regression[0](trend.permute(0,2,1)).permute(0,2,1).contiguous()
            # res_pred.append(predict_x)

        # Instance Denormalization
        predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
            1, input_len, 1
        )  # [batch_size, input_len, num_features]
        predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
            1, input_len, 1
        )  # [batch_size, input_len, num_features]
        # predict_x = torch.fft.ifft(predict_x,dim=-2).real

        return predict_x

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

    def forward(self, batch_x,x_mask):

        if self.task_name == "pretrain":
            return self.pretrain(batch_x,x_mask)
        elif self.task_name == "finetune":
            dec_out = self.forecast(batch_x,x_mask)
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
