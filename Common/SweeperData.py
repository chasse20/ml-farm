from dataclasses import dataclass
from typing import List
from Common.Table import Table

@dataclass
class SweeperData():
	Table: Table
	CandidateFeatures: List[ List[ float ] ]