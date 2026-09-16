from dataclasses import dataclass

@dataclass
class Settings():
	RetryAmount: int
	Port: int
	Host: str
	User: str
	Password: str
	RootPath: str
	ModelFile: str

	def __post_init__( self ):
		self.RetryAmount = int( self.RetryAmount )
		self.Port = int( self.Port )

		if ( self.RetryAmount < 0 ):
			raise ValueError( "RetryAmount must be zero or greater" )