from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

@dataclass
class Job:
	JobId: int
	SlaveId: Optional[ int ]
	GroupId: int
	Index: int
	Elapsed: Optional[ timedelta ]
	IsReady: bool