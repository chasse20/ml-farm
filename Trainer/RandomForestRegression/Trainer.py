import numpy as np
from joblib import parallel_config
from sklearn.ensemble import RandomForestRegressor
from skl2onnx.common.data_types import FloatTensorType
from Common.Table import Table
from Common.TrainedModel import TrainedModel
from Common.TrainerType import TrainerType
from Trainer.RandomForestRegression.HyperParams import HyperParams

class Trainer:
	_hyperParams: HyperParams

	def __init__( self, tHyperParams: HyperParams ):
		self._hyperParams = tHyperParams

	def Train( self, tThreadCount: int, tTable: Table ) -> TrainedModel:
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
			tempModel = RandomForestRegressor(
				n_estimators = self._hyperParams.NumberOfTrees,
				max_leaf_nodes = self._hyperParams.NumberOfLeaves,
				min_samples_leaf = self._hyperParams.MinimumExampleCountPerLeaf,
				max_features = self._hyperParams.FeatureFraction,
				random_state = self._hyperParams.RandomSeed & 0xFFFFFFFF,
				n_jobs = tThreadCount,
				bootstrap = True
			)

			with parallel_config( backend = "threading", n_jobs = tThreadCount ):
				tempModel.fit( tempInputs, tempLabels, sample_weight = tempWeights )

			tempInitialTypes = [ ( "inputs", FloatTensorType( [ None, tempInputs.shape[ 1 ] ] ) ) ]

			return TrainedModel( tempModel, tempInitialTypes, TrainerType.RandomForestRegression )
	
		return None