from dataclasses import dataclass
from io import BytesIO
import struct
from typing import List

@dataclass
class TrainRow():
	Label: float
	Weight: float
	Features: List[ float ]
	
	def Deserialize( self, tStream: BytesIO ):
		# Label and Weight
		self.Label = struct.unpack( "<f", tStream.read( 4 ) )[ 0 ]
		self.Weight = struct.unpack( "<f", tStream.read( 4 ) )[ 0 ]
		
		# Features
		tempListLength = struct.unpack( "<i", tStream.read( 4 ) )[ 0 ]
		self.Features = []
		
		for _ in range( tempListLength ):
			self.Features.append( struct.unpack( "<f", tStream.read( 4 ) )[ 0 ] )