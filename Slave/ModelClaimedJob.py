from dataclasses import dataclass
from typing import Optional
from SQL.Group import Group
from SQL.Job import Job
from SQL.Table import Table
from SQL.JobTrainSet import JobTrainSet

@dataclass
class ModelClaimedJob():
	Group: Group
	Job: Job
	Table: Optional[ Table ]
	JobTrainSet: Optional[ JobTrainSet ]