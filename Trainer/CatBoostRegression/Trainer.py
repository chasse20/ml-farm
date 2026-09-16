import numpy as np
from catboost import CatBoostRegressor
from Common.Table import Table
from Common.TrainedModel import TrainedModel
from Common.TrainerType import TrainerType
from Trainer.CatBoostRegression.HyperParams import HyperParams

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
			tempModel = self._GetModel( tThreadCount )
			tempModel.fit( tempInputs, tempLabels, sample_weight = tempWeights )
			
			return TrainedModel( tempModel, None, TrainerType.CatBoostRegression )
	
		return None
	
	def _GetModel( self, tThreadCount: int ):
		return CatBoostRegressor(
			iterations = self._hyperParams.Iterations,
			learning_rate = self._hyperParams.LearningRate,
			depth = self._hyperParams.Depth,
			l2_leaf_reg = self._hyperParams.L2,
			loss_function = self._hyperParams.LossFunction,
			random_seed = self._hyperParams.RandomSeed,
			thread_count = tThreadCount,
			verbose = False,
			allow_writing_files = False
		)