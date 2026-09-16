from dataclasses import dataclass

@dataclass
class JobTrainSet:
	JobId: int
	TableId: int
	Data: bytes