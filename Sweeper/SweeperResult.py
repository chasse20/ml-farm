from dataclasses import dataclass
from typing import List

@dataclass
class SweeperResult():
	Means: List[ float ]
	StandardDeviations: List[ float ]