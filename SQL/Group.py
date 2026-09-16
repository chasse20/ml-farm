from dataclasses import dataclass
from Common.TrainerType import TrainerType

@dataclass
class Group:
	GroupId: int
	MasterId: int
	Key: int
	Type: TrainerType
	HyperParams: str