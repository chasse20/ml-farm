from dataclasses import dataclass

@dataclass
class Settings():
	CandidateChunkSize: int

	def __post_init__( self ):
		self.CandidateChunkSize = int( self.CandidateChunkSize )

		if ( self.CandidateChunkSize <= 0 ):
			raise ValueError( "CandidateChunkSize must be greater than zero" )