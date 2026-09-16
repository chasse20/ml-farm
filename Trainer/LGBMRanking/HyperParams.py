from dataclasses import dataclass, field
from typing import List, Optional

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
	TruncationLevel: int = field( default = 30 )
	LabelGain: Optional[ List[ float ] ] = field( default = None )
	LearningRate: float = field( default = 0.1 )
	MinimumSplitGain: float = field( default = 0.0 )
	SubsampleFraction: float = field( default = 1.0 )
	FeatureFraction: float = field( default = 1.0 )
	L1: float = field( default = 0.0 )
	L2: float = field( default = 0.0 )