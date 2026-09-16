from concurrent.futures import ThreadPoolExecutor
import numpy as np
from Sweeper.SweeperResult import SweeperResult
from Common.TrainerType import TrainerType
from Common.SweeperData import SweeperData
from Sweeper.Settings import Settings
from Trainer.Service import Service as TrainerService

class Service:
	_settings: Settings

	def __init__( self, tSettings: Settings ):
		self._settings = tSettings

	def Sweep( self, tTrainer: TrainerService, tThreadCount: int, tTrainerType: TrainerType, tHyperParams: str, tData: SweeperData ) -> SweeperResult:
		if ( tData is None ):
			return None

		# Train
		if ( tTrainerType != TrainerType.RandomForestRegression ):
			raise ValueError( f"Unsupported Sweeper Trainer Type: {tTrainerType}" )

		tempTrainedModel = tTrainer.Train( tThreadCount, tTrainerType, tHyperParams, tData.Table )

		if ( tempTrainedModel is None ):
			raise RuntimeError( "Sweeper Trainer returned no model" )

		tempTrees = tempTrainedModel.Model.estimators_
		tempTreeCount = len( tempTrees )

		if ( tempTreeCount == 0 ):
			raise RuntimeError( "Sweeper model contains no trees" )

		# Score Candidates
		tempCandidateFeatures = np.asarray( tData.CandidateFeatures, dtype = np.float32 )
		tempCandidateCount = len( tempCandidateFeatures )
		tempMeans = np.empty( tempCandidateCount, dtype = np.float64 )
		tempStandardDeviations = np.empty( tempCandidateCount, dtype = np.float64 )
		tempTreeGroups = [ tempTrees[ i::tThreadCount ] for i in range( 0, tThreadCount ) ]

		with ThreadPoolExecutor( max_workers = tThreadCount ) as tempExecutor:
			for tempStartIndex in range( 0, tempCandidateCount, self._settings.CandidateChunkSize ):
				tempEndIndex = min( tempStartIndex + self._settings.CandidateChunkSize, tempCandidateCount )
				tempCandidates = tempCandidateFeatures[ tempStartIndex:tempEndIndex ]
				tempFutures = [ tempExecutor.submit( self._GetTreeSums, tempTreeGroup, tempCandidates ) for tempTreeGroup in tempTreeGroups ]
				tempSum = np.zeros( tempEndIndex - tempStartIndex, dtype = np.float64 )
				tempSquareSum = np.zeros( tempEndIndex - tempStartIndex, dtype = np.float64 )

				for tempFuture in tempFutures:
					tempPartialSum, tempPartialSquareSum = tempFuture.result()
					tempSum += tempPartialSum
					tempSquareSum += tempPartialSquareSum

				tempMean = tempSum / tempTreeCount
				tempVariance = np.maximum( tempSquareSum / tempTreeCount - tempMean * tempMean, 1e-12 )
				tempMeans[ tempStartIndex:tempEndIndex ] = tempMean
				tempStandardDeviations[ tempStartIndex:tempEndIndex ] = np.sqrt( tempVariance )

		return SweeperResult( tempMeans.tolist(), tempStandardDeviations.tolist() )

	@staticmethod
	def _GetTreeSums( tTrees: list, tCandidateFeatures: np.ndarray ) -> tuple[ np.ndarray, np.ndarray ]:
		tempSum = np.zeros( len( tCandidateFeatures ), dtype = np.float64 )
		tempSquareSum = np.zeros( len( tCandidateFeatures ), dtype = np.float64 )

		for tempTree in tTrees:
			tempPredictions = tempTree.predict( tCandidateFeatures )
			tempSum += tempPredictions
			tempSquareSum += tempPredictions * tempPredictions

		return tempSum, tempSquareSum