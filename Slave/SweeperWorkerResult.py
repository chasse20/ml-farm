from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

@dataclass
class SweeperWorkerResult():
	JobId: int
	Data: Optional[ bytes ]
	Elapsed: timedelta
	Error: Optional[ str ]