from enum import Enum

class TrainerType( Enum ):
	LGBMBinary = 0
	LGBMRegression = 1
	LGBMRanking = 2
	RidgeRegression = 3
	RandomForestRegression = 4
	LinearSVCBinary = 5
	CatBoostRegression = 6
	XGBoostRegression = 7