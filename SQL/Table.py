from dataclasses import dataclass

@dataclass
class Table:
	TableId: int
	MasterId: int
	Data: bytes