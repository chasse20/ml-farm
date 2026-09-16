from dataclasses import dataclass
from io import BytesIO
import struct
from typing import List
from Slave.JobTrainRow import JobTrainRow

@dataclass
class JobTrainSet():
	TrainRows: List[ JobTrainRow ]
	
	def Deserialize( self, tStream: BytesIO ):
		# Train Rows
		tempListLength = struct.unpack( "<i", tStream.read( 4 ) )[ 0 ]
		self.TrainRows = []
		
		for _ in range( tempListLength ):
			tempTrainRow = JobTrainRow( 0, 0 )
			tempTrainRow.Deserialize( tStream )
			self.TrainRows.append( tempTrainRow )