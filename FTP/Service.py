import asyncio
from datetime import timedelta
import logging
import os
import tempfile
import time
import aioftp
from Common.TrainedModel import TrainedModel
from Common.TrainerType import TrainerType
from FTP.Settings import Settings

logger = logging.getLogger( __name__ )

class Service:
	_settings: Settings
	_isConverterRegistered: bool = False

	def __init__( self, tSettings: Settings ):
		self._settings = tSettings

	@classmethod
	def SaveONNX( cls, tTrainedModel: TrainedModel ) -> str:
		if ( tTrainedModel is None ):
			raise ValueError( "The Trained Model cannot be None" )

		tempPath = None

		try:
			import onnx
			from skl2onnx import convert_sklearn

			tempTime = time.perf_counter()
			tempModel = None
			tempTrainerType = TrainerType( tTrainedModel.TrainerType )

			# Special Version handling to convert to ONNX
			if ( tempTrainerType == TrainerType.CatBoostRegression ):
				tempExportFile = tempfile.NamedTemporaryFile( suffix = ".onnx", delete = False )
				tempExportPath = tempExportFile.name
				tempExportFile.close()

				try:
					tTrainedModel.Model.save_model( tempExportPath, format = "onnx" )
					tempModel = onnx.load( tempExportPath )
				finally:
					if ( os.path.exists( tempExportPath ) ):
						os.remove( tempExportPath )

				tempOldInputName = tempModel.graph.input[ 0 ].name
				tempModel.graph.input[ 0 ].name = "inputs"

				for tempNode in tempModel.graph.node:
					for i, tempInput in enumerate( tempNode.input ):
						if ( tempInput == tempOldInputName ):
							tempNode.input[ i ] = "inputs"
			elif ( tempTrainerType == TrainerType.XGBoostRegression ):
				from onnxmltools.convert import convert_xgboost
				tempModel = convert_xgboost( tTrainedModel.Model, initial_types = tTrainedModel.InitialTypes, target_opset = 11 )
			elif ( tempTrainerType == TrainerType.LGBMBinary or tempTrainerType == TrainerType.LGBMRegression or tempTrainerType == TrainerType.LGBMRanking ):
				cls._RegisterConverters()
				tempOptions = None

				if ( tempTrainerType == TrainerType.LGBMBinary ):
					tempOptions = { id( tTrainedModel.Model ): { "zipmap": False } }

				tempModel = convert_sklearn( tTrainedModel.Model, initial_types = tTrainedModel.InitialTypes, target_opset = { "": 11, "ai.onnx.ml": 3 }, options = tempOptions )

				for tempImport in tempModel.opset_import:
					if ( tempImport.domain == "" ):
						tempImport.version = 11
			elif ( tempTrainerType == TrainerType.LinearSVCBinary ):
				tempOptions = { "svc": { "raw_scores": False } }
				
				tempModel = convert_sklearn( tTrainedModel.Model, initial_types = tTrainedModel.InitialTypes, target_opset = { "": 11, "ai.onnx.ml": 3 }, options = tempOptions )
			elif ( tempTrainerType == TrainerType.RidgeRegression or tempTrainerType == TrainerType.RandomForestRegression ):
				tempModel = convert_sklearn( tTrainedModel.Model, initial_types = tTrainedModel.InitialTypes )
			else:
				raise ValueError( f"Unsupported Trainer Type: {tempTrainerType}" )

			tempElapsed = str( timedelta( seconds = time.perf_counter() - tempTime ) )
			logger.info( f"ONNX Converted: @ {tempElapsed}" )

			# Set output node name for regression models
			if ( tempTrainerType == TrainerType.LGBMRegression or tempTrainerType == TrainerType.LGBMRanking or tempTrainerType == TrainerType.RidgeRegression or tempTrainerType == TrainerType.RandomForestRegression or tempTrainerType == TrainerType.CatBoostRegression or tempTrainerType == TrainerType.XGBoostRegression ):
				tempOldOutputName = tempModel.graph.output[ 0 ].name
				tempModel.graph.output[ 0 ].name = "outputs"

				for tempNode in tempModel.graph.node:
					for i, tempOutput in enumerate( tempNode.output ):
						if ( tempOutput == tempOldOutputName ):
							tempNode.output[ i ] = "outputs"

			tempTime = time.perf_counter()
			tempFile = tempfile.NamedTemporaryFile( suffix = ".onnx", delete = False )
			tempPath = tempFile.name
			tempFile.close()
			onnx.save( tempModel, tempPath )

			tempElapsed = str( timedelta( seconds = time.perf_counter() - tempTime ) )
			logger.info( f"ONNX Temporary Saved: {tempPath} @ {tempElapsed}" )

			return tempPath
		except Exception as tException:
			if ( tempPath is not None and os.path.exists( tempPath ) ):
				os.remove( tempPath )

			logger.error( f"Failed to Save ONNX: {tException}" )
			raise

	@classmethod
	def _RegisterConverters( cls ):
		if ( cls._isConverterRegistered ):
			return

		from lightgbm import LGBMClassifier
		from lightgbm import LGBMRegressor
		from lightgbm import LGBMRanker
		from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
		from skl2onnx import update_registered_converter
		from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
		from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes

		update_registered_converter(
			LGBMClassifier,
			"LightGbmLGBMClassifier",
			calculate_linear_classifier_output_shapes,
			convert_lightgbm,
			options = {
				"zipmap": [ True, False, "columns" ],
				"nocl": [ True, False ],
				"native": [ True, False ]
			}
		)
		update_registered_converter(
			LGBMRegressor,
			"LightGbmLGBMRegressor",
			calculate_linear_regressor_output_shapes,
			convert_lightgbm,
			options = {
				"nocl": [ True, False ],
				"native": [ True, False ]
			}
		)

		update_registered_converter(
			LGBMRanker,
			"LightGbmLGBMRanker",
			calculate_linear_regressor_output_shapes,
			convert_lightgbm,
			options = {
				"nocl": [ True, False ],
				"native": [ True, False ]
			}
		)

		cls._isConverterRegistered = True

	@staticmethod
	def ValidateONNX( tPath: str ) -> bool:
		if ( tPath is None ):
			return False

		try:
			import numpy as np
			import onnx
			import onnxruntime as ort
			import psutil

			# Precheck
			tempTime = time.perf_counter()
			tempModel = onnx.load( tPath )
			onnx.checker.check_model( tempModel )
			logger.info( f"ONNX Prechecked: {tPath}" )

			# I/O Names
			try:
				tempProcess = psutil.Process()
				tempCPUs = len( tempProcess.cpu_affinity() ) if hasattr( tempProcess, "cpu_affinity" ) else os.cpu_count()
			except Exception as tException:
				logger.error( f"Error accessing CPU info: {tException}" )
				tempCPUs = os.cpu_count()

			if ( tempCPUs is None or tempCPUs <= 0 ):
				tempCPUs = 1

			tempSessionOptions = ort.SessionOptions()
			tempSessionOptions.intra_op_num_threads = tempCPUs
			tempSessionOptions.inter_op_num_threads = 1
			logger.info( f"ONNX CPU Count: {tempCPUs}" )

			tempSession = ort.InferenceSession( tPath, sess_options = tempSessionOptions, providers = [ "CPUExecutionProvider" ] )
			tempInputNames = [ tempInput.name for tempInput in tempSession.get_inputs() ]
			tempOutputNames = [ tempOutput.name for tempOutput in tempSession.get_outputs() ]
			logger.info( f"ONNX Input Names: {tempInputNames}" )
			logger.info( f"ONNX Output Names: {tempOutputNames}" )

			# Run Test
			tempInputShapes = [ tempInput.shape for tempInput in tempSession.get_inputs() ]
			tempDummyInputs = {}

			for tempInputName, tempInputShape in zip( tempInputNames, tempInputShapes ):
				tempRealInputShape = [ tempDimension if ( isinstance( tempDimension, int ) and tempDimension > 0 ) else 1 for tempDimension in tempInputShape ]
				logger.info( f"ONNX Shape for {tempInputName}: {tempRealInputShape}" )
				tempDummyInputs[ tempInputName ] = np.zeros( tempRealInputShape, dtype = np.float32 )

			tempSession.run( tempOutputNames, tempDummyInputs )
			tempElapsed = str( timedelta( seconds = time.perf_counter() - tempTime ) )
			logger.info( f"ONNX Validated: {tPath} @ {tempElapsed}" )

			return True
		except Exception as tException:
			logger.error( f"Failed to Validate ONNX: {tException}" )
			raise

	async def UploadModelAsync( self, tMasterId: int, tJobId: int, tPath: str ) -> bool:
		if ( tPath is None ):
			return False

		tempRelativePath = self._settings.ModelFile.format( tMasterId, tJobId )
		tempRemotePath = self._settings.RootPath + tempRelativePath

		for i in range( 0, self._settings.RetryAmount + 1 ):
			try:
				tempTime = time.perf_counter()

				async with aioftp.Client.context( self._settings.Host, self._settings.Port, self._settings.User, self._settings.Password ) as tempClient:
					await tempClient.upload( tPath, tempRemotePath, write_into = True )

				tempElapsed = str( timedelta( seconds = time.perf_counter() - tempTime ) )
				logger.info( f"Model Uploaded: {tempRelativePath} @ {tempElapsed}" )

				return True
			except Exception as tException:
				logger.error( f"Failed to Upload Model: {tempRelativePath} @ Attempt {i + 1}/{self._settings.RetryAmount + 1}: {tException}" )

				if ( i < self._settings.RetryAmount ):
					await asyncio.sleep( 1 )

		return False