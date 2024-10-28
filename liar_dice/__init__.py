from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

from torch import Tensor
from torch.optim.optimizer import StateDict

Roll: TypeAlias = tuple[int, ...]
PlayerId: TypeAlias = Literal[0, 1]
MatchResult: TypeAlias = Literal[-1, 1]

State = Tensor  # Public state. Includes previous actions. Initialized as `zeros(self.D_PUB)`.
Priv = Tensor
# Private state of a player. Initialized as `zeros(self.D_PRI)`. including the perspective for the scores

Variant = Literal["normal", "joker", "stairs"]
PruneType = Literal["zero", "upper", "lower", "us", "them", "avg"]


@dataclass
class TrainArg:
    d1: int
    d2: int
    sides: int = 6
    variant: Variant = "normal"
    eps: float = 1e-2
    layers: int = 4
    layer_size: int = 100
    lr: float = 1e-3
    w: float = 1e-2
    path: str = "model.pt"
    device: Literal["cpu", "cuda"] = "cpu"


class CheckpointFile(TypedDict):
    epoch: int
    model_state_dict: StateDict
    optimizer_state_dict: StateDict
    args: TrainArg
