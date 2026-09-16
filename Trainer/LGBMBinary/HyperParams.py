from dataclasses import dataclass, field

@dataclass
class HyperParams():
	RandomSeed: int = field( default = 0 )
	MaxBin: int = field( default = 255 )
	NumberOfIterations: int = field( default = 100 )
	SequenceLength: int = field( default = 1 )
	MaximumTreeDepth: int = field( default = -1 )
	NumberOfLeaves: int = field( default = 31 )
	MinimumExampleCountPerLeaf: int = field( default = 20 )
	SubsampleFrequency: int = field( default = 0 )
	LearningRate: float = field( default = 0.1 )
	MinimumSplitGain: float = field( default = 0.0 )
	SubsampleFraction: float = field( default = 1.0 )
	FeatureFraction: float = field( default = 1.0 )
	L1: float = field( default = 0.0 )
	L2: float = field( default = 0.0 )
	PositiveWeight: float = field( default = 0.5 )