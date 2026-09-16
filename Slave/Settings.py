from dataclasses import dataclass

@dataclass
class Settings():
	Id: int
	WorkerProcessCount: int
	WorkerThreadCount: int
	PollFrequency: int
	Name: str
	SQLPort: int
	SQLHost: str
	SQLDB: str
	SQLUser: str
	SQLPassword: str

	def __post_init__( self ):
		self.Id = int( self.Id )
		self.WorkerProcessCount = int( self.WorkerProcessCount )
		self.WorkerThreadCount = int( self.WorkerThreadCount )
		self.PollFrequency = int( self.PollFrequency )
		self.SQLPort = int( self.SQLPort )

		if ( self.WorkerProcessCount <= 0 ):
			raise ValueError( "WorkerProcessCount must be greater than zero" )

		if ( self.WorkerThreadCount <= 0 ):
			raise ValueError( "WorkerThreadCount must be greater than zero" )

		if ( self.PollFrequency <= 0 ):
			raise ValueError( "PollFrequency must be greater than zero" )