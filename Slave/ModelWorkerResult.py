from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

@dataclass
class ModelWorkerResult():
	JobId: int
	Path: Optional[ str ]
	Elapsed: timedelta
	Error: Optional[ str ]