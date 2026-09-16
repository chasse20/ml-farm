from dataclasses import dataclass
from io import BytesIO
import struct
from typing import List
from Common.TrainRow import TrainRow


@dataclass
class Row():
	TrainRows: List[ TrainRow ]
	Features: List[ float ]
	
	def Deserialize( self, tStream: BytesIO ):
		# Train Rows
		tempListLength = struct.unpack( "<i", tStream.read( 4 ) )[ 0 ]
		self.TrainRows = []
		
		for _ in range( tempListLength ):
			tempTrainRow = TrainRow( 0, 0, [] )
			tempTrainRow.Deserialize( tStream )
			self.TrainRows.append( tempTrainRow )
			
		# Features
		tempListLength = struct.unpack( "<i", tStream.read( 4 ) )[ 0 ]
		self.Features = []
		
		for _ in range( tempListLength ):
			self.Features.append( struct.unpack( "<f", tStream.read( 4 ) )[ 0 ] )