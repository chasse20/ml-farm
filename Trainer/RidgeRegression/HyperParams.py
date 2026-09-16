from dataclasses import dataclass, field

@dataclass
class HyperParams():
	SequenceLength: int = field( default = 1 )
	PcaComponents: int = field( default = 15 )
	Alpha: float = field( default = 1.0 )