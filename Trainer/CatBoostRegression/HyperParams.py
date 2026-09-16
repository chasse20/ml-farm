from dataclasses import dataclass, field

@dataclass
class HyperParams():
	RandomSeed: int = field( default = 0 )
	SequenceLength: int = field( default = 1 )
	Iterations: int = field( default = 1000 )
	Depth: int = field( default = 6 )
	LearningRate: float = field( default = 0.03 )
	L2: float = field( default = 3.0 )
	LossFunction: str = field( default = "RMSE" )