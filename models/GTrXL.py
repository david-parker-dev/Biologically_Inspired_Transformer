import math

import torch
import torch.nn.functional as F
from entmax import entmax_bisect
from models.RoPE import RotaryPositionalEmbeddings
from torch import nn


class GTrXL_Block(nn.Module):
    # Structure of a GTrXL Block
    # Features two stages of processing, and add gating to the residiual connections

    def __init__(self, config):
        super().__init__()

        self.config = config
        embed_dim = config.MODEL_EMBED_SIZE
        enable_sparsity = config.ENABLE_SPARSITY

        self.an1 = MultiheadAttention(config, enable_mask=True, enable_sparsity= enable_sparsity)

        self.activation = F.gelu

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

        self.Residual_Gate_1 = Gated_Residual_Layer(embed_dim, config.GRU_GATE_INITIAL_BIAS)
        self.Residual_Gate_2 = Gated_Residual_Layer(embed_dim, config.GRU_GATE_INITIAL_BIAS)

        self.fc1 = nn.Linear(embed_dim, config.MODEL_HIDDEN_DIM)
        self.fc2 = nn.Linear(config.MODEL_HIDDEN_DIM, embed_dim)

    def forward(self, x, past_input=None):

        # Stage 1 - Multi Head Self Attention
        x_output = self.layer_norm_1(x)
        x_output, new_memory = self.an1(x_output, past_input=past_input)
        x = self.Residual_Gate_1(x, x_output)

        # Stage 2 - Multi Layer Perceptron
        x_output = self.layer_norm_2(x)
        x_output = self.activation(self.fc1(x_output))
        x_output = self.fc2(x_output)
        x = self.Residual_Gate_2(x, x_output)
        return x, new_memory

class Gated_Residual_Layer(nn.Module):

    def __init__(self, embed_dim, initial_bias):
        super().__init__()

        # Reset Weights -
        self.Wr = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Ur = nn.Linear(embed_dim, embed_dim, bias=False)

        # Update Weights
        self.Wz = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Uz = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.Wz.weight)
        nn.init.zeros_(self.Uz.weight)

        # Update Bias
        nn.init.constant_(self.Uz.bias, initial_bias)

        # Candidate State
        self.Wg = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Ug = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, Residual_Connection, Layer_Output):
        # Residual Connection (x)
        # Layer Output (y)

        # Reset Gate (r) = σ(Wr y + Ur x)
        r = torch.sigmoid(self.Wr(Layer_Output) + self.Ur(Residual_Connection))

        # Update Gate (z) = σ(Wz y + Uz x − b(l)g)
        z = torch.sigmoid(self.Wz(Layer_Output) + self.Uz(Residual_Connection))

        # Candidate State (h) = tanh(Wg y + Ug (r⊙x))
        h = torch.tanh(self.Wg(Layer_Output) + self.Ug(r * Residual_Connection))

        # GRU(x, y) = (1 − z) x + z⊙h
        return ((1 - z) * Residual_Connection) + (z * h)

class MultiheadAttention(nn.Module):
    def __init__(self,config, enable_sparsity = False):
        super().__init__()

        self.model_dim_size = config.MODEL_EMBED_SIZE
        self.num_heads = config.MODEL_NUM_HEADS
        self.head_dim = self.model_dim_size // self.num_heads
        self.enable_sparsity = enable_sparsity
        self.max_memory_length = config.MAX_MEMORY_LENGTH

        # Query, Key, Value, Final
        self.q_layer = nn.Linear(self.model_dim_size, self.model_dim_size)
        self.k_layer = nn.Linear(self.model_dim_size, self.model_dim_size)
        self.v_layer = nn.Linear(self.model_dim_size, self.model_dim_size)
        self.linear_layer = nn.Linear(self.model_dim_size, self.model_dim_size)

        self.positional_encoder = RotaryPositionalEmbeddings(self.head_dim)

        if self.enable_sparsity == True:
            self.alpha = nn.Parameter(torch.full((self.num_heads,), -0.4363))

    def build_causal_mask(self, query_length, cache_length, device):
        cache_mask = torch.zeros(query_length, cache_length, device=device)
        new_chunk_mask = torch.triu(torch.full((query_length, query_length), float('-inf'), device=device), diagonal=1)
        return torch.cat([cache_mask, new_chunk_mask], dim=-1)

    def split_heads(self, x, batch_size, sequence_length):
        # Convert (batch, seq, d_model) to (batch, heads, seq, head_dim)
        x = x.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x

    def scaled_dot_product(self, q, k, v):
        # Scaling Factor
        d_k = q.size()[-1]
        query_length = q.size(-2)
        key_length = k.size(-2)
        cache_length = key_length - query_length

        # Combine / Scale
        dot_product = torch.matmul(q, k.transpose(-1, -2))
        scaled = dot_product / math.sqrt(d_k)

        mask = self.build_causal_mask(query_length, cache_length, device=q.device)
        scaled = scaled + mask

        if self.enable_sparsity:
            alpha = 1.0 + F.softplus(self.alpha)
            alpha = alpha.view(1, self.num_heads, 1, 1)
            attention_weights = entmax_bisect(scaled, alpha, dim=-1)
        else:
            attention_weights = F.softmax(scaled, dim=-1)

        values = torch.matmul(attention_weights, v)

        return values, attention_weights

    def forward(self, x, past_input=None):

        batch_size, sequence_length, _ = x.size()

        # Merge Context & Input
        if past_input is not None:
            full_input = torch.cat([past_input, x], dim=-2)
        else:
            full_input = x

        # Trim Full Sequence to max size if required
        max_context_length = self.max_memory_length + sequence_length
        if full_input.size(-2) > max_context_length:
            full_input = full_input[:, -max_context_length:, :]

        context_length = full_input.size(-2)

        # Compute Values
        q = self.q_layer(x)
        k = self.k_layer(full_input)
        v = self.v_layer(full_input)

        # Split Each into Head Shapes
        q = self.split_heads(q, batch_size, sequence_length)
        k = self.split_heads(k, batch_size, context_length)
        v = self.split_heads(v, batch_size, context_length)

        context_length = k.size(-2)
        query_start = context_length - sequence_length

        q_positions = torch.arange(query_start, query_start + sequence_length, device=x.device)
        k_positions = torch.arange(0, context_length, device=x.device)

        q = self.positional_encoder(q, q_positions)
        k = self.positional_encoder(k, k_positions)

        # Scaled Dot Product Attention
        values, _ = self.scaled_dot_product(q, k, v)

        values = values.permute(0, 2, 1, 3)  # (batch, seq_len, heads, head_dim)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)

        out = self.linear_layer(values)
        new_memory = full_input[:, -self.max_memory_length:, :].detach()

        return out, new_memory
