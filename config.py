from dataclasses import dataclass, field


@dataclass
class config:

    # TensorBoard
    RUN_NAME: str = "13/08 - Sparse"

    # Ablation
    ENABLE_SPARSITY: bool = True

    # Environment
    NUM_ENVS: int = 8
    ENV_NAME: str = "MiniGrid-MemoryS13-v0"
    EPISODE_MAX_LENGTH: int = 845
    SEED: int = 7
    EVAL_SEEDS: list = field(default_factory=lambda: [212 + i for i in range(10)])
    ROLLOUT_SIZE: int = 500

    # Encoder
    INPUT_CONV_CHANNELS: int = 16
    INPUT_DIRECTION_EMBED_DIM: int = 4
    INPUT_GRID_SIZE: int = 7
    INPUT_NUM_DIRECTIONS: int = 4

    # Trunk / Model Architecture
    MODEL_EMBED_SIZE: int = 160
    MODEL_HIDDEN_DIM: int = 640
    MODEL_LAYERS: int = 4
    MODEL_NUM_HEADS: int = 4
    MAX_MEMORY_LENGTH: int = 126
    GRU_GATE_INITIAL_BIAS: float = -2.0

    # Evaluation
    EVAL_FREQUENCY: int = 5
    EVAL_EPISODES: int = 10

    # PPO
    LEARNING_RATE: float = 10 ** -5
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    SEQUENCE_LENGTH: int = 100
    NUMBER_ITERATIONS: int = 1000
    EPOCHS: int = 3
    CLIP_EPS: float = 0.2
    VALUE_LOSS_COEFFICIENT: float = 0.5
    ENTROPY_COEFFICIENT: float = 0.01
    MAX_GRAD_NORM: float = 0.5
