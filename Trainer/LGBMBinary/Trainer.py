import numpy as np
from lightgbm import LGBMClassifier
from skl2onnx.common.data_types import FloatTensorType
from Common.Table import Table
from Common.TrainedModel import TrainedModel
from Common.TrainerType import TrainerType
from Trainer.LGBMBinary.HyperParams import HyperParams

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
			
			# Process
			for i in range( self._hyperParams.SequenceLength - 1, len( tTable.Rows ) ):
				# Generate Temporal Features
				tempTemporalFeatures = []
					
				for j in range( i - self._hyperParams.SequenceLength + 1, i + 1 ):
					tempTemporalFeatures.extend( tTable.Rows[ j ].Features )

				# Train Rows
				for tempTrainRow in tTable.Rows[ i ].TrainRows:
					tempLabels.append( tempTrainRow.Label )
					tempInputs.append( tempTrainRow.Features + tempTemporalFeatures )
					
					# Weights
					tempWeight = 1.0
					
					if ( tempTrainRow.Label == True ):
						tempWeight = self._hyperParams.PositiveWeight
					else:
						tempWeight = 1.0 - self._hyperParams.PositiveWeight
					
					tempWeights.append( tempWeight )

			tempLabels = np.asarray( tempLabels, dtype = np.float32 )
			tempWeights = np.asarray( tempWeights, dtype = np.float32 )
			tempInputs = np.asarray( tempInputs, dtype = np.float32 )
			
			# Hyper Parameters
			tempLeaves = self._hyperParams.NumberOfLeaves
			
			if ( self._hyperParams.MaximumTreeDepth > 0 ):
				tempLeaves = min( self._hyperParams.NumberOfLeaves, 2 ** self._hyperParams.MaximumTreeDepth )

			# Train
			tempModel = LGBMClassifier(
				random_state = self._hyperParams.RandomSeed,
				objective = "binary",
				boosting_type = "dart",
				max_depth = self._hyperParams.MaximumTreeDepth,
				num_leaves = tempLeaves,
				min_child_samples = self._hyperParams.MinimumExampleCountPerLeaf,
				bagging_freq = self._hyperParams.SubsampleFrequency,
				learning_rate = self._hyperParams.LearningRate,
				min_split_gain = self._hyperParams.MinimumSplitGain,
				subsample = self._hyperParams.SubsampleFraction,
				colsample_bytree = self._hyperParams.FeatureFraction,
				reg_alpha = self._hyperParams.L1,
				reg_lambda = self._hyperParams.L2,
				verbose = -1,
				max_bin = self._hyperParams.MaxBin,
				n_estimators = self._hyperParams.NumberOfIterations,
				device = "cpu",
				n_jobs = tThreadCount
			)
			
			tempModel.fit( tempInputs, tempLabels, tempWeights )
			
			# Convert from Huber (latest RELEASED version of onnxmltools requires this)
			#tempModelString = tempModel.model_to_string()
			#tempModelString = tempModelString.replace( "huber", "regression" )
			#tempRegressionModel = lgb.Booster( model_str = tempModelString )

			tempInitialTypes = [ ( "inputs", FloatTensorType( [ None, tempInputs.shape[ 1 ] ] ) ) ]

			return TrainedModel( tempModel, tempInitialTypes, TrainerType.LGBMBinary )
	
		return None