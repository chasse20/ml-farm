import numpy as np
from onnxmltools.convert.common.data_types import FloatTensorType
from xgboost import XGBRegressor
from Common.Table import Table
from Common.TrainedModel import TrainedModel
from Common.TrainerType import TrainerType
from Trainer.XGBoostRegression.HyperParams import HyperParams

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
			
			tempInitialTypes = [ ( "inputs", FloatTensorType( [ None, tempInputs.shape[ 1 ] ] ) ) ]

			return TrainedModel( tempModel, tempInitialTypes, TrainerType.XGBoostRegression )
	
		return None
	
	def _GetModel( self, tThreadCount: int ):
		return XGBRegressor(
			n_estimators = self._hyperParams.Estimators,
			learning_rate = self._hyperParams.LearningRate,
			max_depth = self._hyperParams.MaxDepth,
			min_child_weight = self._hyperParams.MinChildWeight,
			subsample = self._hyperParams.SubsampleFraction,
			colsample_bytree = self._hyperParams.FeatureFraction,
			reg_alpha = self._hyperParams.RegAlpha,
			reg_lambda = self._hyperParams.RegLambda,
			max_bin = self._hyperParams.MaxBin,
			tree_method = self._hyperParams.TreeMethod,
			objective = self._hyperParams.Objective,
			random_state = self._hyperParams.RandomSeed,
			n_jobs = tThreadCount,
			verbosity = 0
		)