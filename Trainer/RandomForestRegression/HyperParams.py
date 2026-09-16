from dataclasses import dataclass, field

@dataclass
class HyperParams():
	RandomSeed: int = field( default = 0 )
	NumberOfTrees: int = field( default = 300 )
	SequenceLength: int = field( default = 1 )
	NumberOfLeaves: int = field( default = 31 )
	MinimumExampleCountPerLeaf: int = field( default = 20 )
	FeatureFraction: float = field( default = 1.0 )