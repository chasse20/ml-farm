from dataclasses import dataclass
from typing import Any
from Common.TrainerType import TrainerType

@dataclass
class TrainedModel():
	Model: Any
	InitialTypes: Any
	TrainerType: TrainerType