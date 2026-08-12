import torch
import torch.nn.functional as F
from models.GTrXL import GTrXL_Block
from torch import nn


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

class MLR_Objective(nn.Module):
    def __init__(self, embedd_dim):
        super().__init__()
        self.projection = nn.Linear(embedd_dim, 1)

    def forward(self, x):
        return self.projection(x)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Image
        self.conv = nn.Conv2d(in_channels=3, out_channels=config.INPUT_CONV_CHANNELS, kernel_size=2)
        conv_output_size = config.INPUT_CONV_CHANNELS * (config.INPUT_GRID_SIZE - 1) ** 2
        self.image_projection = nn.Linear(conv_output_size, config.MODEL_EMBED_SIZE)

        # Direction
        self.direction_embedding = nn.Embedding(config.INPUT_NUM_DIRECTIONS, config.INPUT_DIRECTION_EMBED_DIM)

        # Combined
        self.combined_projection = nn.Linear(config.MODEL_EMBED_SIZE + config.INPUT_DIRECTION_EMBED_DIM, config.MODEL_EMBED_SIZE)

    def forward(self, image, direction):
        batch_size, sequence_length = image.size(0), image.size(1)

        # Image
        images = image.reshape(batch_size * sequence_length, *image.shape[2:])
        images = images.permute(0, 3, 1, 2)
        images = images.float()
        image_features = self.conv(images)
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
        self.MLR_Objective = MLR_Objective(config.MODEL_EMBED_SIZE)

    def forward(self, image, direction, memory=None):

        x = self.Encoder(image, direction)

        if memory is None:
                memory = [None] * len(self.blocks)

        new_memory = []
        for block, past_input in zip(self.blocks, memory):
            x, block_memory = block(x, past_input=past_input)
            new_memory.append(block_memory)

        return self.Critic(x), self.Actor(x), self.MLR_Objective(x), new_memory
