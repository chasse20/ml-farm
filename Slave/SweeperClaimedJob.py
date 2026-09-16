from dataclasses import dataclass
from SQL.Group import Group
from SQL.Job import Job

@dataclass
class SweeperClaimedJob():
	Group: Group
	Job: Job
	Data: bytes
	Results: bytes