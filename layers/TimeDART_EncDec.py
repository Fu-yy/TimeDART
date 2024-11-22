from utils.masking import generate_causal_mask, generate_self_only_mask
import math
import torch
import torch.nn as nn
from torch.nn import init




class WeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4):
        super(WeightGenerator, self).__init__()
        #print('FACTORIZED {}'.format(factorized))
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim
        if self.factorized:
            self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cpu')
            # self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cuda:0')
            self.generator = self.generator = nn.Sequential(*[
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 100)
            ])

            self.mem_dim = 10
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.Q = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
        else:
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True) for _ in range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True) for _ in range(number_of_weights)])
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))
            bias = [torch.matmul(memory, self.B[i]).squeeze(1) for i in range(self.number_of_weights)]
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]
            return weights, bias
        else:
            return self.P, self.B


class ChannelIndependence(nn.Module):
    def __init__(
        self,
        input_len: int,
    ):
        super(ChannelIndependence, self).__init__()
        self.input_len = input_len

    def forward(self, x):
        """
        :param x: [batch_size, input_len, num_features]
        :return: [batch_size * num_features, input_len, 1]
        """
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, self.input_len, 1)
        return x


class AddSosTokenAndDropLast(nn.Module):
    def __init__(self, sos_token: torch.Tensor):
        super(AddSosTokenAndDropLast, self).__init__()
        assert sos_token.dim() == 3
        self.sos_token = sos_token

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        sos_token_expanded = self.sos_token.expand(
            x.size(0), -1, -1
        )  # [batch_size * num_features, 1, d_model]
        x = torch.cat(
            [sos_token_expanded, x], dim=1
        )  # [batch_size * num_features, seq_len + 1, d_model]
        x = x[:, :-1, :]  # [batch_size * num_features, seq_len, d_model]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(x)
        output = self.norm2(x + self.dropout(ff_output))

        return output


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super(CausalTransformer, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, is_mask=True):
        # x: [batch_size * num_features, seq_len, d_model]
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len).to(x.device) if is_mask else None
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x


class Diffusion(nn.Module):
    def __init__(
        self,
        time_steps: int,
        device: torch.device,
        scheduler: str = "cosine",
    ):
        super(Diffusion, self).__init__()
        self.device = device
        self.time_steps = time_steps

        if scheduler == "cosine":
            self.betas = self._cosine_beta_schedule().to(self.device)
        elif scheduler == "linear":
            self.betas = self._linear_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Invalid scheduler: {scheduler=}")

        self.alpha = 1 - self.betas
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, self.time_steps)
        return betas

    def sample_time_steps(self, shape):
        return torch.randint(0, self.time_steps, shape, device=self.device)

    def noise(self, x, t):
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1)  # [batch_size * num_features, seq_len, 1]
        # x_t = sqrt(gamma_t) * x + sqrt(1 - gamma_t) * noise
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x):
        # x: [batch_size * num_features, seq_len, patch_len]
        t = self.sample_time_steps(x.shape[:2])  # [batch_size * num_features, seq_len]
        noisy_x, noise = self.noise(x, t)
        return noisy_x, noise, t

class DiffusionForComp(nn.Module):
    def __init__(
        self,
        time_steps: int,
        device: torch.device,
        real_scheduler: str = "cosine",
        imag_scheduler: str = "quad",
    ):
        super(DiffusionForComp, self).__init__()
        self.device = device
        self.time_steps = time_steps

        if real_scheduler == "cosine":
            self.real_betas = self._cosine_beta_schedule().to(self.device)
        elif real_scheduler == "linear":
            self.betas = self._linear_beta_schedule().to(self.device)
        elif real_scheduler == "quad":
            self.betas = self._quad_beta_schedule().to(self.device)
        elif real_scheduler == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            self.betas = self._jsd_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Invalid scheduler: {real_scheduler=}")
        self.real_alpha = 1 - self.real_betas
        self.real_gamma = torch.cumprod(self.real_alpha, dim=0).to(self.device)


        if imag_scheduler == "cosine":
            self.imag_betas = self._cosine_beta_schedule().to(self.device)
        elif imag_scheduler == "linear":
            self.imag_betas = self._linear_beta_schedule().to(self.device)
        elif imag_scheduler == "quad":
            self.imag_betas = self._quad_beta_schedule().to(self.device)
        elif imag_scheduler == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            self.imag_betas = self._jsd_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Invalid scheduler: {imag_scheduler=}")
        self.imag_alpha = 1 - self.imag_betas
        self.imag_gamma = torch.cumprod(self.imag_alpha, dim=0).to(self.device)





        # self.betas_real = self._quad_beta_schedule().to(self.device)
        # self.betas_imag = self._jsd_beta_schedule().to(self.device)
        #
        # self.alpha = 1 - self.betas
        # self.alpha_real = 1 - self.betas_real
        # self.alpha_imag = 1 - self.betas_imag
        # self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)
        # # Calculate alpha and gamma for real and imaginary parts separately
        #
        # self.gamma_real = torch.cumprod(self.alpha_real, dim=0).to(self.device)
        # self.gamma_imag = torch.cumprod(self.alpha_imag, dim=0).to(self.device)

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, self.time_steps)
        return betas

    def _quad_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, self.time_steps) ** 2
        return betas
    def _jsd_beta_schedule(self):
        betas = 1.0 /torch.linspace(self.time_steps, 1, self.time_steps)
        return betas

    def sample_time_steps(self, shape):
        return torch.randint(0, self.time_steps, shape, device=self.device)

    def noise(self, real,imag, t):
        real_noise = torch.randn_like(real)

        real_gamma_t = self.real_gamma[t].unsqueeze(-1)  # [batch_size * num_features, seq_len, 1]
        # x_t = sqrt(gamma_t) * x + sqrt(1 - gamma_t) * noise
        real_noisy_x = torch.sqrt(real_gamma_t) * real + torch.sqrt(1 - real_gamma_t) * real_noise



        imag_noise = torch.randn_like(imag)

        imag_gamma_t = self.imag_gamma[t].unsqueeze(-1)  # [batch_size * num_features, seq_len, 1]
        # x_t = sqrt(gamma_t) * x + sqrt(1 - gamma_t) * noise
        imag_noisy_x = torch.sqrt(imag_gamma_t) * imag + torch.sqrt(1 - imag_gamma_t) * imag_noise

        return real_noisy_x, real_noise,imag_noisy_x,imag_noise




    def forward(self, real,imag):
        # x: [batch_size * num_features, seq_len, patch_len]
        t = self.sample_time_steps(real.shape[:2])  # [batch_size * num_features, seq_len]
        real_noisy_x, real_noise,imag_noisy_x, imag_noise = self.noise(real=real,imag=imag, t=t)
        return real_noisy_x, real_noise,imag_noisy_x, imag_noise, t


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerDecoderBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, tgt_mask, src_mask):
        """
        :param query: [batch_size * num_features, seq_len, d_model]
        :param key: [batch_size * num_features, seq_len, d_model]
        :param value: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.self_attention(query, query, query, attn_mask=tgt_mask)
        query = self.norm1(query + self.dropout(attn_output))

        # Encoder attention
        attn_output, _ = self.encoder_attention(query, key, value, attn_mask=src_mask)
        query = self.norm2(query + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(query)
        x = self.norm3(query + self.dropout(ff_output))

        return x


class DenoisingPatchDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super(DenoisingPatchDecoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, is_tgt_mask=True, is_src_mask=True):
        seq_len = query.size(1)
        tgt_mask = (
            generate_self_only_mask(seq_len).to(query.device) if is_tgt_mask else None
        )
        src_mask = (
            generate_self_only_mask(seq_len).to(query.device) if is_src_mask else None
        )
        for layer in self.layers:
            query = layer(query, key, value, tgt_mask, src_mask)
        x = self.norm(query)
        return x


