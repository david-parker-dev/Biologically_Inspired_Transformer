from dataclasses import dataclass


@dataclass
class config:

    # TensorBoard
    RUN_NAME: str = "Sparse - RPE"
    EVAL_FREQUENCY: int = 5

    # Trunk / Model Architecture
    MODEL_INPUT_SIZE: int = 4
    MODEL_NUM_ACTIONS: int = 2
    MODEL_EMBED_SIZE: int = 512
    MODEL_LAYERS: int = 6
    MAX_MEMORY_LENGTH: int = 128

    # PPO
    LEARNING_RATE: float = 10 ** -5
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    ROLLOUT_SIZE: int = 2000
    SEQUENCE_LENGTH: int = 100
    NUMBER_ITERATIONS: int = 1000
    EPOCHS: int = 3
    CLIP_EPS: float = 0.2
    VALUE_LOSS_COEFFICIENT: float = 0.5
    ENTROPY_COEFFICIENT: float = 0.01
    MAX_GRAD_NORM: float = 0.5
