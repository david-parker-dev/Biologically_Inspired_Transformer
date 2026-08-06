import math

import torch
import torch.nn.functional as F
from torch import nn


class GTrXL_Block(nn.Module):
    # Structure of a GTrXL Block
    # Features two stages of processing, and add gating to the residiual connections

    def __init__(self, embed_dim, num_heads, hidden_dim, max_sequence_length, max_memory_length):
        super().__init__()

        self.an1 = MultiheadAttention(embed_dim, num_heads, max_sequence_length=max_sequence_length, max_memory_length=max_memory_length, enable_mask=True)

        self.activation = F.gelu

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

        self.Residual_Gate_1 = Gated_Residual_Layer(embed_dim)
        self.Residual_Gate_2 = Gated_Residual_Layer(embed_dim)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

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

    def __init__(self, embed_dim, initial_bias=-2.0):
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
    def __init__(self, d_model, num_heads, enable_mask, max_sequence_length, max_memory_length):
        super().__init__()

        self.d_model = d_model  # Model Embedding Size
        self.num_heads = num_heads  # Number of Heads
        self.head_dim = d_model // num_heads  # Dimensionality of each head
        self.enable_mask = enable_mask  # Toggles Causal Mask
        self.max_sequence_length = max_sequence_length
        self.max_memory_length = max_memory_length

        # Query, Key, and Value Layers
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_E_layer = nn.Linear(d_model, d_model) # W_k,E
        self.v_layer = nn.Linear(d_model, d_model)
        self.k_R_layer = nn.Linear(d_model, d_model, bias=False)

        # Positional Encoding Parameters
        self.u = nn.Parameter(torch.empty(self.num_heads, 1, self.head_dim))
        self.v = nn.Parameter(torch.empty(self.num_heads, 1, self.head_dim))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

        # Combines heads into final hidden layer
        self.linear_layer = nn.Linear(d_model, d_model)

    def _rel_shift(self, x, zero_triu=False):
        batch_size, n_head, query_len, key_len = x.size()

        # Pad x with a column of zeros
        zero_pad = torch.zeros((batch_size, n_head, query_len, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        # Then reshape to offset the matrix
        x_padded = x_padded.view(batch_size, n_head, key_len + 1, query_len)

        # Slice off the top row and restore original shape
        return x_padded[:, :, 1:].view_as(x)

    def relative_positional_embeddings(self, seq_len, d_model, device):
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0, device=device) / d_model))
        pos_seq = torch.arange(seq_len - 1, -1, -1.0, device=device)
        sinusoid_inp = torch.ger(pos_seq, inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb.unsqueeze(1) # [seq_len, 1, d_model]

    def build_causal_mask(self, query_length, cache_length, device):
        cache_mask = torch.zeros(query_length, cache_length, device=device)
        new_chunk_mask = torch.triu(torch.full((query_length, query_length), float('-inf'), device=device), diagonal=1)
        return torch.cat([cache_mask, new_chunk_mask], dim=-1)

    def split_heads(self, x, batch_size, sequence_length):
        # Convert (batch, seq, d_model) to (batch, heads, seq, head_dim)
        x = x.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x

    def scaled_dot_product(self, q, k_E, k_R, v):
        # Scaling Factor
        d_k = q.size()[-1]
        query_length = q.size(-2)
        key_length = k_E.size(-2)
        cache_length = key_length - query_length

        # Terms A & C - Content_based Addressing (Query + u)
        word_query = q + self.u
        AC = torch.matmul(word_query, k_E.transpose(-1, -2))

        # Term B & D - Position-based addressing (Query + v)
        position_query = q + self.v
        BD = torch.matmul(position_query, k_R.transpose(-1, -2))
        BD = self._rel_shift(BD)

        # Combine / Scale
        scaled = (AC + BD) / math.sqrt(d_k)

        if self.enable_mask is True:
            mask = self.build_causal_mask(query_length, cache_length, device=q.device)
            scaled = scaled + mask

        attention_weights = F.softmax(scaled, dim=-1)
        values = torch.matmul(attention_weights, v)

        return values, attention_weights

    def forward(self, x, past_input=None):

        batch_size, sequence_length, _ = x.size()

        # Concatenate past input with current input
        if past_input is not None:
            extended_input = torch.cat([past_input, x], dim=-2)
        else:
            extended_input = x

        # Trim input size to max memory length
        if extended_input.size(-2) > self.max_memory_length + sequence_length:
            extended_input = extended_input[:, -(self.max_memory_length + sequence_length):, :]

        context_length = extended_input.size(-2)

        # Compute Values
        q = self.q_layer(x)
        k_E = self.k_E_layer(extended_input)
        v = self.v_layer(extended_input)

        # Relative Positional Key (k_R)
        positional_embedding = self.relative_positional_embeddings(context_length, self.d_model, x.device)
        k_R = self.k_R_layer(positional_embedding) # shape: [context_length, 1, d_model]

        # Split Each into Head Shapes
        q = self.split_heads(q, batch_size, sequence_length)
        k_E = self.split_heads(k_E, batch_size, context_length)
        k_R = self.split_heads(k_R, batch_size=1, sequence_length=context_length)
        v = self.split_heads(v, batch_size, context_length)

        # Scaled Dot Product Attention
        values, _ = self.scaled_dot_product(q, k_E, k_R, v)

        values = values.permute(0, 2, 1, 3)  # (batch, seq_len, heads, head_dim)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)

        #out = self.linear_layer(values)
        out = self.linear_layer(values)
        new_memory = extended_input.detach()

        return out, new_memory
