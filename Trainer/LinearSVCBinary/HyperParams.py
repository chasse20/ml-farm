from dataclasses import dataclass, field

@dataclass
class HyperParams():
	RandomSeed: int = field( default = 0 )
	MaxIterations: int = field( default = 1000 )
	SequenceLength: int = field( default = 1 )
	C: float = field( default = 1.0 )