import torch
import torch.nn.functional as F
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

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.object_embedding = nn.Embedding(config.INPUT_NUM_OBJECTS, config.INPUT_OBJECT_EMBED_DIM)
        self.colour_embedding = nn.Embedding(config.INPUT_NUM_COLOURS, config.INPUT_COLOUR_EMBED_DIM)
        self.state_embedding = nn.Embedding(config.INPUT_NUM_STATES, config.INPUT_STATE_EMBED_DIM)
        cell_embed_dim = (config.INPUT_OBJECT_EMBED_DIM + config.INPUT_COLOUR_EMBED_DIM + config.INPUT_STATE_EMBED_DIM)

        # Image
        self.conv = nn.Conv2d(in_channels=cell_embed_dim, out_channels=config.INPUT_CONV_CHANNELS, kernel_size=2)
        conv_output_size = config.INPUT_CONV_CHANNELS * (config.INPUT_GRID_SIZE - 1) ** 2
        self.image_projection = nn.Linear(conv_output_size, config.MODEL_EMBED_SIZE)

        # Direction
        self.direction_embedding = nn.Embedding(config.INPUT_NUM_DIRECTIONS, config.INPUT_DIRECTION_EMBED_DIM)

        # Combined
        self.combined_projection = nn.Linear(config.MODEL_EMBED_SIZE + config.INPUT_DIRECTION_EMBED_DIM, config.MODEL_EMBED_SIZE)

    def forward(self, image, direction):
        batch_size, sequence_length = image.size(0), image.size(1)

        # Image
        images = image.reshape(batch_size * sequence_length, *image.shape[2:]).long()
        objects = self.object_embedding(images[..., 0])
        colours = self.colour_embedding(images[..., 1])
        states = self.state_embedding(images[..., 2])
        cells = torch.cat([objects, colours, states], dim=-1)   # (N, grid, grid, cell_embed_dim)
        cells = cells.permute(0, 3, 1, 2)                       # (N, cell_embed_dim, grid, grid)
        image_features = self.conv(cells)
        image_features = F.relu(image_features)
        image_features = image_features.flatten(start_dim=1)
        image_features = self.image_projection(image_features)
        image_features = image_features.reshape(batch_size, sequence_length, -1)

        # Direction
        direction_features = self.direction_embedding(direction.long())

        # Combination
        combined = torch.cat([image_features, direction_features], dim=-1)
        return self.combined_projection(combined)

class Model(nn.Module):
    def __init__(self, num_actions, config):
        super().__init__()

        self.config = config
        self.Encoder = Encoder(config)
        self.blocks = nn.ModuleList([ GTrXL_Block(config) for _ in range(config.MODEL_LAYERS)])
        self.Actor = Actor(config.MODEL_EMBED_SIZE, num_actions)
        self.Critic = Critic(config.MODEL_EMBED_SIZE)

    def forward(self, image, direction, memory=None, dones=None):

        x = self.Encoder(image, direction)

        if memory is None:
            memory = [None] * len(self.blocks)

        # Non Masked Path
        input = x
        new_memory = []
        for block, past_input in zip(self.blocks, memory):
            input, block_memory = block(input, past_input=past_input, dones=dones)
            new_memory.append(block_memory)

        return self.Critic(input), self.Actor(input), new_memory
