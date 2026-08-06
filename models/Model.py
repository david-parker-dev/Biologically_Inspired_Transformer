from torch import nn

from models.GTrXL import GTrXL_Block


class Actor(nn.Module):
    def __init__(self, embedd_dim, num_actions):
        super().__init__()
        self.actor = nn.Linear(embedd_dim, num_actions)

    def forward(self, x):
        return self.actor(x)

class Critic(nn.Module):
    def __init__(self, embedd_dim):
        super().__init__()
        self.critic = nn.Linear(embedd_dim, 1)

    def forward(self, x):
        return self.critic(x)

class Model(nn.Module):
    def __init__(self, num_states, embed_dim, num_actions, num_layers=2, num_heads=2, hidden_dim=2048, max_sequence_length=128, max_memory_length=128):
        super().__init__()

        self.input_projection = nn.Linear(num_states, embed_dim)
        self.blocks = nn.ModuleList([
            GTrXL_Block(embed_dim, num_heads, hidden_dim, max_sequence_length, max_memory_length) for _ in range(num_layers)])
        self.Actor = Actor(embed_dim, num_actions)
        self.Critic = Critic(embed_dim)

    def forward(self, x, memory=None):

        assert x.dim() == 3, "Model needs (batch, seq, features)"

        x = self.input_projection(x)

        if memory is None:
                memory = [None] * len(self.blocks)

        new_memory = []
        for block, past_input in zip(self.blocks, memory):
            x, block_memory = block(x, past_input=past_input)
            new_memory.append(block_memory)

        return self.Critic(x), self.Actor(x), new_memory
