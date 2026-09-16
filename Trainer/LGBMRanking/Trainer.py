import numpy as np
from lightgbm import LGBMRanker
from skl2onnx.common.data_types import FloatTensorType
from Common.Table import Table
from Common.TrainedModel import TrainedModel
from Common.TrainerType import TrainerType
from Trainer.LGBMRanking.HyperParams import HyperParams

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
			tempGroups = []
									
			for i in range( self._hyperParams.SequenceLength - 1, len( tTable.Rows ) ):
				# Generate Temporal Features
				tempTemporalFeatures = []
					
				for j in range( i - self._hyperParams.SequenceLength + 1, i + 1 ):
					tempTemporalFeatures.extend( tTable.Rows[ j ].Features )

				# Train Rows
				tempTrainRows = tTable.Rows[ i ].TrainRows

				if ( len( tempTrainRows ) <= 0 ):
					continue

				tempGroups.append( len( tempTrainRows ) )

				for tempTrainRow in tempTrainRows:
					tempLabels.append( tempTrainRow.Label )
					tempWeights.append( tempTrainRow.Weight )
					tempInputs.append( tempTrainRow.Features + tempTemporalFeatures )

			tempLabels = np.asarray( tempLabels, dtype = np.int32 )
			tempWeights = np.asarray( tempWeights, dtype = np.float32 )
			tempInputs = np.asarray( tempInputs, dtype = np.float32 )
			
			# Hyper Parameters
			tempLeaves = self._hyperParams.NumberOfLeaves

			if ( self._hyperParams.MaximumTreeDepth > 0 ):
				tempLeaves = min( self._hyperParams.NumberOfLeaves, 2 ** self._hyperParams.MaximumTreeDepth )
			
			# Train
			tempParams = dict(
				random_state = self._hyperParams.RandomSeed,
				objective = "lambdarank",
				boosting_type = "gbdt",
				max_depth = self._hyperParams.MaximumTreeDepth,
				num_leaves = tempLeaves,
				min_child_samples = self._hyperParams.MinimumExampleCountPerLeaf,
				bagging_freq = self._hyperParams.SubsampleFrequency,
				learning_rate = self._hyperParams.LearningRate,
				min_split_gain = self._hyperParams.MinimumSplitGain,
				subsample = self._hyperParams.SubsampleFraction,
				colsample_bytree = self._hyperParams.FeatureFraction,
				lambdarank_truncation_level = self._hyperParams.TruncationLevel,
				reg_alpha = self._hyperParams.L1,
				reg_lambda = self._hyperParams.L2,
				verbosity = -1,
				max_bin = self._hyperParams.MaxBin,
				n_estimators = self._hyperParams.NumberOfIterations,
				device = "cpu",
				n_jobs = tThreadCount
			)

			if self._hyperParams.LabelGain is not None:
				tempParams[ "label_gain" ] = self._hyperParams.LabelGain

			tempModel = LGBMRanker( **tempParams )
						
			tempModel.fit( tempInputs, tempLabels, sample_weight = tempWeights, group = tempGroups )
			
			# Convert from Huber (latest RELEASED version of onnxmltools requires this)
			#tempModelString = tempModel.model_to_string()
			#tempModelString = tempModelString.replace( "huber", "regression" )
			#tempRegressionModel = lgb.Booster( model_str = tempModelString )

			tempInitialTypes = [ ( "inputs", FloatTensorType( [ None, tempInputs.shape[ 1 ] ] ) ) ]

			return TrainedModel( tempModel, tempInitialTypes, TrainerType.LGBMRanking )
	
		return None