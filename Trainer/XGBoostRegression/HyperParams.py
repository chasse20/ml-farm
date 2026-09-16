from dataclasses import dataclass, field

@dataclass
class HyperParams():
	RandomSeed: int = field( default = 0 )
	SequenceLength: int = field( default = 1 )
	MaxBin: int = field( default = 256 )
	Estimators: int = field( default = 1000 )
	MaxDepth: int = field( default = 6 )
	LearningRate: float = field( default = 0.03 )
	MinChildWeight: float = field( default = 1.0 )
	SubsampleFraction: float = field( default = 1.0 )
	FeatureFraction: float = field( default = 1.0 )
	RegAlpha: float = field( default = 0.0 )
	RegLambda: float = field( default = 1.0 )
	TreeMethod: str = field( default = "auto" )
	Objective: str = field( default = "reg:squarederror" )