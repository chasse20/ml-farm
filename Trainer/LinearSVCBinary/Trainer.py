import numpy as np
from skl2onnx.common.data_types import FloatTensorType
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from Common.Table import Table
from Common.TrainedModel import TrainedModel
from Common.TrainerType import TrainerType
from Trainer.LinearSVCBinary.HyperParams import HyperParams

class Trainer:
	_hyperParams: HyperParams

	def __init__( self, tHyperParams: HyperParams ):
		self._hyperParams = tHyperParams

	def Train( self, tTable: Table ) -> TrainedModel:
		if ( tTable is not None ):
			# Create Labels and Features
			tempLabels = []
			tempWeights = []
			tempInputs = []
									
			for i in range( self._hyperParams.SequenceLength - 1, len( tTable.Rows ) ):
				# Generate Temporal Features
				tempTemporalFeatures = []
					
				for j in range( i - self._hyperParams.SequenceLength + 1, i + 1 ):
					tempTemporalFeatures.extend( tTable.Rows[ j ].Features )

				# Train Rows
				for tempTrainRow in tTable.Rows[ i ].TrainRows:
					tempLabels.append( int( tempTrainRow.Label ) )
					tempWeights.append( tempTrainRow.Weight )
					tempInputs.append( tempTrainRow.Features + tempTemporalFeatures )

			tempLabels = np.asarray( tempLabels, dtype = np.int64 )
			tempWeights = np.asarray( tempWeights, dtype = np.float32 )
			tempInputs = np.asarray( tempInputs, dtype = np.float32 )
			
			# Train
			tempModel = self._GetModel()
			tempModel.fit( tempInputs, tempLabels, svc__sample_weight = tempWeights )
			
			tempInitialTypes = [ ( "inputs", FloatTensorType( [ None, tempInputs.shape[ 1 ] ] ) ) ]

			return TrainedModel( tempModel, tempInitialTypes, TrainerType.LinearSVCBinary )
	
		return None
	
	def _GetModel( self ):
		return Pipeline( [
			( "svc", LinearSVC( random_state = self._hyperParams.RandomSeed, C = self._hyperParams.C, max_iter = self._hyperParams.MaxIterations ) )
		] )