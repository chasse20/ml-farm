import numpy as np
from skl2onnx.common.data_types import FloatTensorType
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from Common.Table import Table
from Common.TrainedModel import TrainedModel
from Common.TrainerType import TrainerType
from Trainer.RidgeRegression.HyperParams import HyperParams

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
					tempLabels.append( tempTrainRow.Label )
					tempWeights.append( tempTrainRow.Weight )
					tempInputs.append( tempTrainRow.Features + tempTemporalFeatures )

			tempLabels = np.asarray( tempLabels, dtype = np.float32 )
			tempWeights = np.asarray( tempWeights, dtype = np.float32 )
			tempInputs = np.asarray( tempInputs, dtype = np.float32 )
			
			# Train
			tempModel = self._GetModel()
			tempModel.fit( tempInputs, tempLabels, ridge__sample_weight = tempWeights )
			
			tempInitialTypes = [ ( "inputs", FloatTensorType( [ None, tempInputs.shape[ 1 ] ] ) ) ]

			return TrainedModel( tempModel, tempInitialTypes, TrainerType.RidgeRegression )
	
		return None
	
	def _GetModel( self ):
		if ( self._hyperParams.PcaComponents > 0 ):
			return Pipeline( [
				( "pca", PCA( n_components = self._hyperParams.PcaComponents ) ),
				( "ridge", Ridge( alpha = self._hyperParams.Alpha ) ),
			] )

		return Pipeline( [
				( "ridge", Ridge( alpha = self._hyperParams.Alpha ) )
			] )