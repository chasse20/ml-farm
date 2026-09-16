from __future__ import annotations

import gc
from io import BytesIO
import os
import signal
import struct
import time
import traceback
from datetime import timedelta
from Common.DataType import DataType
from Slave.JobTrainSet import JobTrainSet
from Slave.ModelClaimedJob import ModelClaimedJob
from Slave.ModelWorkerResult import ModelWorkerResult
from Common.Row import Row
from Slave.SweeperClaimedJob import SweeperClaimedJob
from Common.SweeperData import SweeperData
from Sweeper.SweeperResult import SweeperResult
from Slave.SweeperWorkerResult import SweeperWorkerResult
from Common.Table import Table
from Common.TrainRow import TrainRow
from Common.TrainerType import TrainerType
from Sweeper.Settings import Settings as SweeperSettings
from typing import TYPE_CHECKING

if ( TYPE_CHECKING ):
	from Trainer.Service import Service as TrainerService
	from Sweeper.Service import Service as SweeperService

class Worker:
	@staticmethod
	def Execute( tWorkerIndex: int, tCPUIds: list[ int ], tThreadCount: int, tSweeperSettings: SweeperSettings, tInputQueue, tOutputQueue ):
		# Set Threads
		tempThreadCount = str( tThreadCount )

		os.environ[ "OMP_NUM_THREADS" ] = tempThreadCount
		os.environ[ "OMP_THREAD_LIMIT" ] = tempThreadCount
		os.environ[ "OPENBLAS_NUM_THREADS" ] = tempThreadCount
		os.environ[ "MKL_NUM_THREADS" ] = tempThreadCount
		os.environ[ "BLIS_NUM_THREADS" ] = tempThreadCount
		os.environ[ "NUMEXPR_NUM_THREADS" ] = tempThreadCount
		os.environ[ "VECLIB_MAXIMUM_THREADS" ] = tempThreadCount
		os.environ[ "OMP_DYNAMIC" ] = "FALSE"
		os.environ[ "MKL_DYNAMIC" ] = "FALSE"

		# Parent ownership for exiting
		for tempSignal in ( signal.SIGINT, signal.SIGTERM ):
			try:
				signal.signal( tempSignal, signal.SIG_IGN )
			except Exception:
				pass

		import psutil

		tempProcess = psutil.Process()
		tempProcess.cpu_affinity( tCPUIds )

		# Instantiate Trainer service
		from Sweeper.Service import Service as SweeperService
		from Trainer.Service import Service as TrainerService

		tempTrainer = TrainerService()
		tempSweeper = SweeperService( tSweeperSettings )

		# Claim Tasks
		while ( True ):
			tempItem = tInputQueue.get()

			if ( tempItem is None ):
				break

			tempStartTime = time.perf_counter()
			
			if ( isinstance( tempItem, ModelClaimedJob ) ):
				tempResult = Worker._GetModelResult( tempTrainer, tThreadCount, tempItem, tempStartTime )
			elif ( isinstance( tempItem, SweeperClaimedJob ) ):
				tempResult = Worker._GetSweeperResult( tempTrainer, tempSweeper, tThreadCount, tempItem, tempStartTime )
			else:
				raise TypeError( f"Worker {tWorkerIndex} received an invalid work item" )

			try:
				tOutputQueue.put( tempResult )
			except Exception:
				if ( isinstance( tempResult, ModelWorkerResult ) and tempResult.Path is not None and os.path.exists( tempResult.Path ) ):
					os.remove( tempResult.Path )
				raise
			finally:
				tempItem = None
				tempResult = None
				gc.collect()

	@staticmethod
	def _GetModelResult( tTrainer: TrainerService, tThreadCount: int, tClaimedJob: ModelClaimedJob, tStartTime: float ) -> ModelWorkerResult:
		tempPath = None
		tempError = None

		try:
			tempPath = Worker._Train( tTrainer, tThreadCount, tClaimedJob )
		except Exception:
			tempError = traceback.format_exc()

			if ( tempPath is not None and os.path.exists( tempPath ) ):
				os.remove( tempPath )

			tempPath = None

		return ModelWorkerResult( tClaimedJob.Job.JobId, tempPath, timedelta( seconds = time.perf_counter() - tStartTime ), tempError )

	@staticmethod
	def _GetSweeperResult( tTrainer: TrainerService, tSweeper: SweeperService, tThreadCount: int, tClaimedJob: SweeperClaimedJob, tStartTime: float ) -> SweeperWorkerResult:
		tempData = None
		tempError = None

		try:
			tempDeserializedData = Worker._GetSweeperData( tClaimedJob )

			if ( tempDeserializedData is None ):
				raise RuntimeError( f"Sweeper Job has no data: {tClaimedJob.Job.JobId}" )

			tempResult = tSweeper.Sweep( tTrainer, tThreadCount, TrainerType( tClaimedJob.Group.Type ), tClaimedJob.Group.HyperParams, tempDeserializedData )
			tempData = Worker._GetSerializedSweeperData( tempResult )
		except Exception:
			tempError = traceback.format_exc()

		return SweeperWorkerResult( tClaimedJob.Job.JobId, tempData, timedelta( seconds = time.perf_counter() - tStartTime ), tempError )

	@staticmethod
	def _Train( tTrainer: TrainerService, tThreadCount: int, tClaimedJob: ModelClaimedJob ) -> str:
		from FTP.Service import Service as FTPService
		tempPath = None

		try:
			tempDeserializedTable = Worker._GetModelTable( tClaimedJob )

			if ( tempDeserializedTable is None ):
				raise RuntimeError( f"Job has no training table: {tClaimedJob.Job.JobId}" )

			tempTrainedModel = tTrainer.Train( tThreadCount, TrainerType( tClaimedJob.Group.Type ), tClaimedJob.Group.HyperParams, tempDeserializedTable )

			if ( tempTrainedModel is None ):
				raise RuntimeError( f"Trainer returned no model: {tClaimedJob.Job.JobId}" )

			tempPath = FTPService.SaveONNX( tempTrainedModel )

			if ( tempPath is None ):
				raise RuntimeError( f"ONNX conversion returned no path: {tClaimedJob.Job.JobId}" )
			elif ( not FTPService.ValidateONNX( tempPath ) ):
				raise RuntimeError( f"ONNX validation failed: {tClaimedJob.Job.JobId}" )

			return tempPath
		except Exception:
			if ( tempPath is not None and os.path.exists( tempPath ) ):
				os.remove( tempPath )
			raise

	@staticmethod
	def _GetModelTable( tClaimedJob: ModelClaimedJob ) -> Table:
		if ( tClaimedJob.Group is None ):
			return None

		if ( tClaimedJob.Table is not None ):
			tempStream = BytesIO( tClaimedJob.Table.Data )

			tempTable = Table( DataType.Float, [] )
			tempListLength = struct.unpack( "<i", tempStream.read( 4 ) )[ 0 ]
			tempTable.Rows = []
			
			for _ in range( tempListLength ):
				tempRow = Row( [], [] )
				tempRow.Deserialize( tempStream )
				tempTable.Rows.append( tempRow )

			if ( tClaimedJob.JobTrainSet is not None ):
				tempStream = BytesIO( tClaimedJob.JobTrainSet.Data )
				tempJobTrainSet = JobTrainSet( [] )
				tempJobTrainSet.Deserialize( tempStream )
				tempRowsDict = {}

				for tempTrainRow in tempJobTrainSet.TrainRows:
					tempBaseTrainRow = tempTable.Rows[ tempTrainRow.RowIndex ].TrainRows[ tempTrainRow.TrainRowIndex ]

					if ( tempTrainRow.RowIndex not in tempRowsDict ):
						tempRowsDict[ tempTrainRow.RowIndex ] = []

					tempRowsDict[ tempTrainRow.RowIndex ].append( tempBaseTrainRow )

				for i in range( 0, len( tempTable.Rows ) ):
					tempTable.Rows[ i ].TrainRows = tempRowsDict[ i ] if i in tempRowsDict else []

			return tempTable
		
		return None

	@staticmethod
	def _GetSweeperData( tClaimedJob: SweeperClaimedJob ) -> SweeperData:
		if ( tClaimedJob.Job is None or tClaimedJob.Data is None ):
			return None

		tempStream = BytesIO( tClaimedJob.Data )
		tempData = SweeperData( Table( DataType.Float, [] ), [] )

		# Meta
		tempDimensions = struct.unpack( "<i", tempStream.read( 4 ) )[ 0 ]

		# Table
		tempObservationCount = struct.unpack( "<i", tempStream.read( 4 ) )[ 0 ]
		tempObservationFeatures = []

		for _ in range( tempObservationCount ):
			tempFeatures = []

			for _ in range( tempDimensions ):
				tempFeatures.append( struct.unpack( "<f", tempStream.read( 4 ) )[ 0 ] )

			tempObservationFeatures.append( tempFeatures )

		for i in range( tempObservationCount ):
			tempLabel = struct.unpack( "<f", tempStream.read( 4 ) )[ 0 ]
			tempData.Table.Rows.append( Row( [ TrainRow( tempLabel, 1.0, tempObservationFeatures[ i ] ) ], [] ) )

		# Candidate Features
		tempCandidateCount = struct.unpack( "<i", tempStream.read( 4 ) )[ 0 ]

		for _ in range( tempCandidateCount ):
			tempFeatures = []

			for _ in range( tempDimensions ):
				tempFeatures.append( struct.unpack( "<f", tempStream.read( 4 ) )[ 0 ] )

			tempData.CandidateFeatures.append( tempFeatures )

		return tempData

	@staticmethod
	def _GetSerializedSweeperData( tResult: SweeperResult ) -> bytes:
		if ( tResult is None ):
			return None

		tempStream = BytesIO()

		# Means
		tempStream.write( struct.pack( "<i", len( tResult.Means ) ) )

		for tempMean in tResult.Means:
			tempStream.write( struct.pack( "<f", tempMean ) )

		# Standard Deviations
		tempStream.write( struct.pack( "<i", len( tResult.StandardDeviations ) ) )

		for tempStandardDeviation in tResult.StandardDeviations:
			tempStream.write( struct.pack( "<f", tempStandardDeviation ) )

		return tempStream.getvalue()