import json
import logging
from Common.TrainedModel import TrainedModel
from Common.TrainerType import TrainerType
from Common.Table import Table
from Utility.PascalCaseDecoder import PascalCaseDecoder

logger = logging.getLogger( __name__ )

class Service:
	def Train( self, tThreadCount: int, tTrainerType: TrainerType, tHyperParams: str, tTable: Table ) -> TrainedModel:
		if ( tHyperParams is None ):
			raise ValueError( "Missing HyperParams" )
		elif ( tTable is None ):
				raise ValueError( "Missing Table" )

		try:
			tempHyperParams = json.loads( tHyperParams, cls = PascalCaseDecoder )
			return self.TrainModel( tThreadCount, tTrainerType, tempHyperParams, tTable )
		except Exception as tException:
			logger.error( f"Failed to Train: {tException}" )
			raise

	def TrainModel( self, tThreadCount: int, tTrainerType: TrainerType, tHyperParams: dict, tTable: Table ) -> TrainedModel:
		if ( tTrainerType == TrainerType.LGBMBinary ):
			from Trainer.LGBMBinary.HyperParams import HyperParams
			from Trainer.LGBMBinary.Trainer import Trainer
			return Trainer( HyperParams( **tHyperParams ) ).Train( tThreadCount, tTable )
		elif ( tTrainerType == TrainerType.LGBMRegression ):
			from Trainer.LGBMRegression.HyperParams import HyperParams
			from Trainer.LGBMRegression.Trainer import Trainer
			return Trainer( HyperParams( **tHyperParams ) ).Train( tThreadCount, tTable )
		elif ( tTrainerType == TrainerType.LGBMRanking ):
			from Trainer.LGBMRanking.HyperParams import HyperParams
			from Trainer.LGBMRanking.Trainer import Trainer
			return Trainer( HyperParams( **tHyperParams ) ).Train( tThreadCount, tTable )
		elif ( tTrainerType == TrainerType.RidgeRegression ):
			from Trainer.RidgeRegression.HyperParams import HyperParams
			from Trainer.RidgeRegression.Trainer import Trainer
			return Trainer( HyperParams( **tHyperParams ) ).Train( tTable )
		elif ( tTrainerType == TrainerType.RandomForestRegression ):
			from Trainer.RandomForestRegression.HyperParams import HyperParams
			from Trainer.RandomForestRegression.Trainer import Trainer
			return Trainer( HyperParams( **tHyperParams ) ).Train( tThreadCount, tTable )
		elif ( tTrainerType == TrainerType.LinearSVCBinary ):
			from Trainer.LinearSVCBinary.HyperParams import HyperParams
			from Trainer.LinearSVCBinary.Trainer import Trainer
			return Trainer( HyperParams( **tHyperParams ) ).Train( tTable )
		elif ( tTrainerType == TrainerType.CatBoostRegression ):
			from Trainer.CatBoostRegression.HyperParams import HyperParams
			from Trainer.CatBoostRegression.Trainer import Trainer
			return Trainer( HyperParams( **tHyperParams ) ).Train( tThreadCount, tTable )
		elif ( tTrainerType == TrainerType.XGBoostRegression ):
			from Trainer.XGBoostRegression.HyperParams import HyperParams
			from Trainer.XGBoostRegression.Trainer import Trainer
			return Trainer( HyperParams( **tHyperParams ) ).Train( tThreadCount, tTable )

		raise ValueError( f"Unsupported Trainer Type: {tTrainerType}" )