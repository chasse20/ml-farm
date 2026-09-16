from dataclasses import dataclass
from io import BytesIO
import struct

@dataclass
class JobTrainRow():
	RowIndex: int
	TrainRowIndex: int
	
	def Deserialize( self, tStream: BytesIO ):
		self.RowIndex = struct.unpack( "<i", tStream.read( 4 ) )[ 0 ]
		self.TrainRowIndex = struct.unpack( "<i", tStream.read( 4 ) )[ 0 ]