import asyncio
import gc
import logging
import multiprocessing
import os
import queue
import time
import asyncpg
import psutil
from datetime import timedelta
from Slave.ModelClaimedJob import ModelClaimedJob
from Slave.SweeperClaimedJob import SweeperClaimedJob
from Slave.SweeperWorkerResult import SweeperWorkerResult
from Slave.ModelWorkerResult import ModelWorkerResult
from SQL.Group import Group as SQLGroup
from SQL.Job import Job as SQLJob
from SQL.Table import Table as SQLTable
from SQL.JobTrainSet import JobTrainSet as SQLJobTrainSet
from Slave.Settings import Settings
from FTP.Service import Service as FTPService
from Utility.CancellationToken import CancellationToken
from Slave.Worker import Worker
from Sweeper.Settings import Settings as SweeperSettings

logger = logging.getLogger( __name__ )

class Service:
	_settings: Settings
	_activeJobIds: set[ int ]
	_checkoutLock: asyncio.Lock
	_workerProcesses: list
	_workerInputQueues: list
	_workerOutputQueues: list
	_originalCPUIds: list[ int ]

	def __init__( self, tSettings: Settings ):
		self._settings = tSettings
		self._activeJobIds = set()
		self._checkoutLock = None
		self._workerProcesses = []
		self._workerInputQueues = []
		self._workerOutputQueues = []
		self._originalCPUIds = []

	async def ExecuteAsync( self, tFTP: FTPService, tSweeperSettings: SweeperSettings, tCancel: CancellationToken ):
		if ( tCancel.IsCancelled ):
			return
		logger.info( f"Started Service: {self._settings.Name}" )

		# Initialize
		tempProcess = psutil.Process()
		tempContext = multiprocessing.get_context( "spawn" )
		tempRequiredThreadCount = self._settings.WorkerProcessCount * self._settings.WorkerThreadCount
		tempControllerTasks = []
		tempControllerGroup = None
		tempOriginalThreadEnvironment = {}
		tempIsThreadEnvironmentConfigured = False

		self._activeJobIds = set()
		self._checkoutLock = asyncio.Lock()
		self._workerProcesses = []
		self._workerInputQueues = []
		self._workerOutputQueues = []
		self._originalCPUIds = []

		# Calculate CPU allocations
		try:
			if ( not hasattr( tempProcess, "cpu_affinity" ) ):
				raise RuntimeError( "CPU affinity is not supported on this platform" )

			self._originalCPUIds = list( tempProcess.cpu_affinity() )

			if ( tempRequiredThreadCount + self._settings.WorkerThreadCount > len( self._originalCPUIds ) ):
				raise RuntimeError( f"Insufficient CPU capacity: {len( self._originalCPUIds )} available, {tempRequiredThreadCount} required by workers, {self._settings.WorkerThreadCount} required as reserve" )

			await self._EnsureSlaveAsync( tCancel )

			if ( tCancel.IsCancelled ):
				return

			tempWorkerCPUIds = []

			for i in range( 0, self._settings.WorkerProcessCount ):
				tempStartIndex = i * self._settings.WorkerThreadCount
				tempEndIndex = tempStartIndex +  self._settings.WorkerThreadCount
				tempWorkerCPUIds.append( self._originalCPUIds[ tempStartIndex:tempEndIndex ] )

			tempParentCPUIds = self._originalCPUIds[ tempRequiredThreadCount: ]

			logger.info( f"Available CPUs: {len( self._originalCPUIds )}" )
			logger.info( f"Worker processes: {self._settings.WorkerProcessCount}" )
			logger.info( f"Threads per worker: { self._settings.WorkerThreadCount}" )
			logger.info( f"Worker CPU budget: {tempRequiredThreadCount}" )
			logger.info( f"Parent CPU reserve: {len( tempParentCPUIds )}" )

			# Set the environment, store originals
			tempThreadCountString = str( self._settings.WorkerThreadCount );

			tempThreadEnvironment = {
				"OMP_NUM_THREADS": tempThreadCountString,
				"OMP_THREAD_LIMIT": tempThreadCountString,
				"OPENBLAS_NUM_THREADS": tempThreadCountString,
				"MKL_NUM_THREADS": tempThreadCountString,
				"BLIS_NUM_THREADS": tempThreadCountString,
				"NUMEXPR_NUM_THREADS": tempThreadCountString,
				"VECLIB_MAXIMUM_THREADS": tempThreadCountString,
				"OMP_DYNAMIC": "FALSE",
				"MKL_DYNAMIC": "FALSE"
			}

			for tempKey, tempValue in tempThreadEnvironment.items():
				tempOriginalThreadEnvironment[ tempKey ] = os.environ.get( tempKey )
				os.environ[ tempKey ] = tempValue

			tempIsThreadEnvironmentConfigured = True

			# Start Workers
			for i in range( 0, self._settings.WorkerProcessCount ):
				tempInputQueue = tempContext.Queue( maxsize = 1 )
				tempOutputQueue = tempContext.Queue( maxsize = 1 )
				tempWorkerProcess = tempContext.Process(
					name = f"Worker-{i}",
					target = Worker.Execute,
					args = (
						i,
						tempWorkerCPUIds[ i ],
						self._settings.WorkerThreadCount,
						tSweeperSettings,
						tempInputQueue,
						tempOutputQueue
					),
					daemon = False
				)

				try:
					tempWorkerProcess.start()
				except Exception:
					tempInputQueue.cancel_join_thread()
					tempInputQueue.close()
					tempOutputQueue.cancel_join_thread()
					tempOutputQueue.close()
					raise

				self._workerInputQueues.append( tempInputQueue )
				self._workerOutputQueues.append( tempOutputQueue )
				self._workerProcesses.append( tempWorkerProcess )

				logger.info( f"Worker {i} started: PID {tempWorkerProcess.pid} @ CPUs {tempWorkerCPUIds[ i ]}")

			# Restore parent environment
			for tempKey, tempValue in tempOriginalThreadEnvironment.items():
				if ( tempValue is None ):
					os.environ.pop( tempKey, None )
				else:
					os.environ[ tempKey ] = tempValue

			tempIsThreadEnvironmentConfigured = False

			# Parent CPU affinity
			tempProcess.cpu_affinity( tempParentCPUIds )
			logger.info( f"Parent CPUs: {tempParentCPUIds}" )

			# Give each Worker a Controller
			for i in range( 0, self._settings.WorkerProcessCount ):
				tempControllerTasks.append( asyncio.create_task( self._ExecuteWorkerAsync( tFTP, i, tCancel ), name = f"WorkerController-{i}" ) )

			tempControllerGroup = asyncio.gather( *tempControllerTasks )
			await asyncio.shield( tempControllerGroup )
		except BaseException:
			tCancel.Cancel()

			if ( len( tempControllerTasks ) > 0 ):
				await asyncio.shield( asyncio.gather( *tempControllerTasks, return_exceptions = True ) )

			if ( tempControllerGroup is not None and tempControllerGroup.done() ):
				try:
					tempControllerGroup.exception()
				except BaseException:
					pass

			raise
		finally:
			# Reset environment
			if ( tempIsThreadEnvironmentConfigured ):
				for tempKey, tempValue in tempOriginalThreadEnvironment.items():
					if ( tempValue is None ):
						os.environ.pop( tempKey, None )
					else:
						os.environ[ tempKey ] = tempValue

			# Stop Workers
			for i in range( 0, len( self._workerProcesses ) ):
				tempWorkerProcess = self._workerProcesses[ i ]

				if ( tempWorkerProcess.is_alive() ):
					try:
						self._workerInputQueues[ i ].put( None )
					except Exception as tException:
						logger.error( f"Failed to stop Worker {i}: {tException}" )

			for i in range( 0, len( self._workerProcesses ) ):
				tempWorkerProcess = self._workerProcesses[ i ]

				try:
					if ( tempWorkerProcess.is_alive() ):
						await asyncio.to_thread( tempWorkerProcess.join )
					else:
						tempWorkerProcess.join()

					logger.info( f"Worker {i} stopped: ExitCode {tempWorkerProcess.exitcode}" )
				except Exception as tException:
					logger.error( f"Failed to join Worker {i}: {tException}" )

			for tempOutputQueue in self._workerOutputQueues:
				while ( True ):
					try:
						tempResult = tempOutputQueue.get_nowait()
					except queue.Empty:
						break

					if ( isinstance( tempResult, ModelWorkerResult ) and tempResult.Path is not None and os.path.exists( tempResult.Path ) ):
						try:
							os.remove( tempResult.Path )
						except Exception as tException:
							logger.error( f"Failed to remove orphaned ONNX Path {tempResult.Path}: {tException}" )

			# Release queues
			for tempWorkerQueue in self._workerInputQueues + self._workerOutputQueues:
				try:
					tempWorkerQueue.cancel_join_thread()
					tempWorkerQueue.close()
				except Exception as tException:
					logger.error( f"Failed to close Worker Queue: {tException}" )

			# Restore CPU affinity
			if ( len( self._originalCPUIds ) > 0 ):
				try:
					tempProcess.cpu_affinity( self._originalCPUIds )
				except Exception as tException:
					logger.error( f"Failed to restore Parent CPU affinity: {tException}" )

			self._activeJobIds.clear()
			self._workerProcesses = []
			self._workerInputQueues = []
			self._workerOutputQueues = []

			gc.collect()

			logger.info( f"Stopped Service: {self._settings.Name}" )

	async def _ExecuteWorkerAsync( self, tFTP: FTPService, tWorkerIndex: int, tCancel: CancellationToken ):
		while ( not tCancel.IsCancelled ):
			tempWorkerProcess = self._workerProcesses[ tWorkerIndex ]

			if ( not tempWorkerProcess.is_alive() ):
				raise RuntimeError( f"Worker process exited: {tWorkerIndex}, exit code {tempWorkerProcess.exitcode}" )

			tempClaimedSweeperJob = await self._CheckoutSweeperJobAsync( tCancel )

			if ( tempClaimedSweeperJob is not None ):
				await self._ExecuteSweeperWorkerAsync( tWorkerIndex, tempClaimedSweeperJob, tCancel )
			else:
				tempClaimedModelJob = await self._CheckoutModelJobAsync( tCancel )

				if ( tempClaimedModelJob is None ):
					await self._WaitForWorkAsync( tCancel )
				else:
					await self._ExecuteModelWorkerAsync( tFTP, tWorkerIndex, tempClaimedModelJob, tCancel )

	async def _ExecuteSweeperWorkerAsync( self, tWorkerIndex: int, tClaimedJob: SweeperClaimedJob, tCancel: CancellationToken ):
		if ( tCancel.IsCancelled ):
			return

		tempJobId = tClaimedJob.Job.JobId
		tempStartTime = time.perf_counter()
		tempShouldPoll = False

		try:
			logger.info( f"Sweeper Job started: {tempJobId} @ Worker {tWorkerIndex}" )

			await asyncio.to_thread( self._workerInputQueues[ tWorkerIndex ].put, tClaimedJob )

			tempResult = await self._GetWorkerResultAsync( tWorkerIndex )

			if ( tempResult is None ):
				raise RuntimeError( f"Worker {tWorkerIndex} returned no result for Sweeper Job {tempJobId}" )
			elif ( not isinstance( tempResult, SweeperWorkerResult ) ):
				raise RuntimeError( f"Worker {tWorkerIndex} returned an invalid result type for Sweeper Job {tempJobId}: {type( tempResult ).__name__}" )
			elif ( tempResult.JobId != tempJobId ):
				raise RuntimeError( f"Worker {tWorkerIndex} returned Job {tempResult.JobId} while Sweeper Job {tempJobId} was expected" )
			elif ( tempResult.Error is not None ):
				logger.error( f"Sweeper Job failed in Worker {tWorkerIndex}: {tempJobId}\n{tempResult.Error}" )
				tempShouldPoll = True
			elif ( tempResult.Data is None ):
				logger.error( f"Worker {tWorkerIndex} returned no data for Sweeper Job {tempJobId}" )
				tempShouldPoll = True
			else:
				tempElapsed = timedelta( seconds = time.perf_counter() - tempStartTime )
				tempIsUpdated = await self._UpdateSweeperJobAsync( tempJobId, tempResult.Data, tempElapsed )

				if ( tempIsUpdated ):
					logger.info( f"Sweeper Job completed: {tempJobId} @ {tempElapsed}" )
				else:
					logger.info( f"Sweeper Job abandoned before completion: {tempJobId}" )
		except Exception as tException:
			logger.exception( f"Failed to execute Sweeper Job {tempJobId} in Worker {tWorkerIndex}: {tException}" )
			tempShouldPoll = True

			if ( not self._workerProcesses[ tWorkerIndex ].is_alive() ):
				raise
		finally:
			self._activeJobIds.discard( tempJobId )
			gc.collect()

		if ( tempShouldPoll and not tCancel.IsCancelled ):
			await self._WaitForWorkAsync( tCancel )

	async def _UpdateSweeperJobAsync( self, tJobId: int, tResults: bytes, tElapsed: timedelta ) -> bool:
		tempSQL = None

		try:
			tempSQL = await self._GetSQLAsync()
			tempQuery = """
				WITH "UpdatedSweeperJob" AS (
					UPDATE "SweeperJob"
					SET "Results" = $1
					WHERE "JobId" = $2
					RETURNING "JobId"
				)
				UPDATE "Job"
				SET "Elapsed" = $3
				WHERE "JobId" IN (
					SELECT "JobId"
					FROM "UpdatedSweeperJob"
				)
				RETURNING "JobId"
			"""
			tempUpdatedJobId = await tempSQL.fetchval( tempQuery, tResults, tJobId, tElapsed )

			return tempUpdatedJobId is not None
		except Exception as tException:
			logger.error( f"Failed to update Sweeper Job to DB: {tException}" )
			raise
		finally:
			if ( tempSQL is not None ):
				await tempSQL.close()

	async def _ExecuteModelWorkerAsync( self, tFTP: FTPService, tWorkerIndex: int, tClaimedJob: ModelClaimedJob, tCancel: CancellationToken ):
		if ( tCancel.IsCancelled ):
			return

		tempJobId = tClaimedJob.Job.JobId
		tempMasterId = tClaimedJob.Group.MasterId
		tempStartTime = time.perf_counter()
		tempResult = None
		tempPath = None
		tempShouldPoll = False

		try:
			logger.info( f"Job started: {tempJobId} @ Worker {tWorkerIndex}" )

			await asyncio.to_thread( self._workerInputQueues[ tWorkerIndex ].put, tClaimedJob )

			tempResult = await self._GetWorkerResultAsync( tWorkerIndex )

			if ( tempResult.JobId != tempJobId ):
				raise RuntimeError( f"Worker {tWorkerIndex} returned Job {tempResult.JobId} while Job {tempJobId} was expected" )
			elif ( tempResult.Error is not None ):
				logger.error( f"Job failed in Worker {tWorkerIndex}: {tempJobId}\n{tempResult.Error}" )
				tempShouldPoll = True

			tempPath = tempResult.Path

			if ( tempPath is None ):
				logger.error( f"Worker {tWorkerIndex} returned no ONNX path for Job {tempJobId}" )
				tempShouldPoll = True
			else:
				logger.info( f"Job processed: {tempJobId} @ Worker {tWorkerIndex} @ {tempResult.Elapsed}" )
				tempIsUploaded = await tFTP.UploadModelAsync( tempMasterId, tempJobId, tempPath )

				if ( not tempIsUploaded ):
					logger.error( f"Job failed to upload: {tempJobId}" )
					tempShouldPoll = True
				else:
					tempElapsed = timedelta( seconds = time.perf_counter() - tempStartTime )
					tempIsUpdated = await self._UpdateJobElapsedAsync( tempJobId, tempElapsed )

					if ( tempIsUpdated ):
						logger.info( f"Job completed: {tempJobId} @ {tempElapsed}" )
					else:
						logger.info( f"Job abandoned before completion: {tempJobId}" )
		except Exception as tException:
			logger.exception( f"Failed to execute Job {tempJobId} in Worker {tWorkerIndex}: {tException}" )
			tempShouldPoll = True

			if ( not self._workerProcesses[ tWorkerIndex ].is_alive() ):
				raise
		finally:
			if ( tempPath is not None and os.path.exists( tempPath ) ):
				try:
					os.remove( tempPath )
				except Exception as tException:
					logger.error( f"Failed to remove ONNX Path {tempPath}: {tException}" )

			self._activeJobIds.discard( tempJobId )
			gc.collect()

		if ( tempShouldPoll and not tCancel.IsCancelled ):
			await self._WaitForWorkAsync( tCancel )

	async def _WaitForWorkAsync( self, tCancel: CancellationToken ):
		if ( tCancel.IsCancelled ):
			return

		tempCancelTask = asyncio.create_task( tCancel.WaitForCancel() )

		try:
			await asyncio.wait( [ tempCancelTask ], timeout = self._settings.PollFrequency, return_when = asyncio.FIRST_COMPLETED )
		finally:
			if ( not tempCancelTask.done() ):
				tempCancelTask.cancel()

			await asyncio.gather( tempCancelTask, return_exceptions = True )

	async def _GetWorkerResultAsync( self, tWorkerIndex: int ) -> ModelWorkerResult | SweeperWorkerResult:
		tempOutputQueue = self._workerOutputQueues[ tWorkerIndex ]
		tempWorkerProcess = self._workerProcesses[ tWorkerIndex ]

		while ( True ):
			try:
				return tempOutputQueue.get_nowait()
			except queue.Empty:
				if ( not tempWorkerProcess.is_alive() ):
					try:
						return tempOutputQueue.get_nowait()
					except queue.Empty:
						raise RuntimeError( f"Worker process exited: {tWorkerIndex}, exit code {tempWorkerProcess.exitcode}" )

				await asyncio.sleep( 0.1 )

	async def _GetSQLAsync( self ):
		tempConnection = await asyncpg.connect(
			host = self._settings.SQLHost,
			port = self._settings.SQLPort,
			user = self._settings.SQLUser,
			password = self._settings.SQLPassword,
			database = self._settings.SQLDB
		)
		await tempConnection.execute( "SET search_path TO farm" )

		return tempConnection

	async def _EnsureSlaveAsync( self, tCancel: CancellationToken ):
		if ( tCancel.IsCancelled ):
			return

		tempSQL = None

		try:
			tempSQL = await self._GetSQLAsync()
			tempQuery = """
				INSERT INTO "Slave" (
					"SlaveId",
					"Name"
				)
				VALUES (
					$1,
					$2
				)
				ON CONFLICT ("SlaveId")
				DO UPDATE SET
					"Name" = EXCLUDED."Name"
			"""
			await tempSQL.execute( tempQuery, self._settings.Id, self._settings.Name )
		except Exception as tException:
			logger.error( f"Failed to write Slave to DB: {tException}" )
			raise
		finally:
			if ( tempSQL is not None ):
				await tempSQL.close()

	async def _CheckoutSweeperJobAsync( self, tCancel: CancellationToken ) -> SweeperClaimedJob:
		if ( tCancel.IsCancelled ):
			return None

		async with self._checkoutLock:
			if ( tCancel.IsCancelled ):
				return None

			tempSQL = None

			try:
				tempSQL = await self._GetSQLAsync()

				async with tempSQL.transaction():
					tempQuery = """
						SELECT
							"Job"."JobId",
							"Job"."SlaveId",
							"Job"."GroupId",
							"Job"."Index",
							"Job"."Elapsed",
							"Job"."IsReady",
							"Group"."MasterId",
							"Group"."Key",
							"Group"."Type",
							"Group"."HyperParams",
							"SweeperJob"."Data",
							"SweeperJob"."Results"
						FROM "SweeperJob"
						JOIN "Job"
							ON "SweeperJob"."JobId" = "Job"."JobId"
						JOIN "Group"
							ON "Job"."GroupId" = "Group"."GroupId"
						WHERE
							"Job"."IsReady" = TRUE
							AND "Job"."Elapsed" IS NULL
							AND "SweeperJob"."Results" IS NULL
							AND (
								"Job"."SlaveId" = $1
								OR "Job"."SlaveId" IS NULL
							)
							AND NOT (
								"Job"."JobId" = ANY( $2::int[] )
							)
						ORDER BY
							("Job"."SlaveId" = $1) DESC NULLS LAST,
							"Job"."GroupId" ASC,
							"Job"."Index" ASC
						LIMIT 1
						FOR UPDATE OF "Job" SKIP LOCKED
					"""

					tempResult = await tempSQL.fetchrow( tempQuery, self._settings.Id, list( self._activeJobIds ) )

					if ( tempResult is None ):
						return None

					tempGroupSQL = SQLGroup( tempResult[ 2 ], tempResult[ 6 ], tempResult[ 7 ], tempResult[ 8 ], tempResult[ 9 ] )
					tempJobSQL = SQLJob( tempResult[ 0 ], tempResult[ 1 ], tempResult[ 2 ], tempResult[ 3 ], tempResult[ 4 ], tempResult[ 5 ] )
					tempClaimedJob = SweeperClaimedJob( tempGroupSQL, tempJobSQL, tempResult[ 10 ], tempResult[ 11 ] )

					if ( tempJobSQL.SlaveId is None ):
						tempStatus = await tempSQL.execute(
							"""
								UPDATE "Job"
								SET "SlaveId" = $1
								WHERE
									"JobId" = $2
									AND "SlaveId" IS NULL
							""",
							self._settings.Id,
							tempJobSQL.JobId
						)

						if ( tempStatus != "UPDATE 1" ):
							return None

						tempJobSQL.SlaveId = self._settings.Id

				self._activeJobIds.add( tempClaimedJob.Job.JobId )

				return tempClaimedJob
			except Exception as tException:
				logger.exception( f"Failed to get Claimed Sweeper Job: {tException}" )
				raise
			finally:
				if ( tempSQL is not None ):
					await tempSQL.close()

	async def _CheckoutModelJobAsync( self, tCancel: CancellationToken ) -> ModelClaimedJob:
		if ( tCancel.IsCancelled ):
			return None

		async with self._checkoutLock:
			if ( tCancel.IsCancelled ):
				return None

			tempSQL = None

			try:
				tempSQL = await self._GetSQLAsync()

				async with tempSQL.transaction():
					tempQuery = """
						SELECT
							"Job"."JobId",
							"Job"."SlaveId",
							"Job"."GroupId",
							"Job"."Index",
							"Job"."Elapsed",
							"Job"."IsReady",
							"Group"."MasterId",
							"Group"."Key",
							"Group"."Type",
							"Group"."HyperParams"
						FROM "Job"
						JOIN "Group"
							ON "Job"."GroupId" = "Group"."GroupId"
						LEFT JOIN "SweeperJob"
							ON "SweeperJob"."JobId" = "Job"."JobId"
						WHERE
							"SweeperJob"."JobId" IS NULL
							AND "Job"."IsReady" = TRUE
							AND "Job"."Elapsed" IS NULL
							AND (
								"Job"."SlaveId" = $1
								OR "Job"."SlaveId" IS NULL
							)
							AND NOT (
								"Job"."JobId" = ANY( $2::int[] )
							)
						ORDER BY
							("Job"."SlaveId" = $1) DESC NULLS LAST,
							"Job"."GroupId" ASC,
							"Job"."Index" ASC
						LIMIT 1
						FOR UPDATE OF "Job" SKIP LOCKED
					"""

					tempResult = await tempSQL.fetchrow( tempQuery, self._settings.Id, list( self._activeJobIds ) )

					if ( tempResult is None ):
						return None

					tempGroupSQL = SQLGroup( tempResult[ 2 ], tempResult[ 6 ], tempResult[ 7 ], tempResult[ 8 ], tempResult[ 9 ] )
					tempJobSQL = SQLJob( tempResult[ 0 ], tempResult[ 1 ], tempResult[ 2 ], tempResult[ 3 ], tempResult[ 4 ], tempResult[ 5 ] )

					if ( tempJobSQL.SlaveId is None ):
						tempStatus = await tempSQL.execute(
							"""
								UPDATE "Job"
								SET "SlaveId" = $1
								WHERE
									"JobId" = $2
									AND "SlaveId" IS NULL
							""",
							self._settings.Id,
							tempJobSQL.JobId
						)

						if ( tempStatus != "UPDATE 1" ):
							return None

						tempJobSQL.SlaveId = self._settings.Id

					tempTableSQL = None
					tempJobTrainSetSQL = None
					tempQuery = """
						SELECT
							"Table"."TableId",
							"Table"."MasterId",
							"Table"."Data",
							"JobTrainSet"."JobId",
							"JobTrainSet"."TableId",
							"JobTrainSet"."Data"
						FROM "JobTrainSet"
						JOIN "Table"
							ON "Table"."TableId" = "JobTrainSet"."TableId"
						WHERE "JobTrainSet"."JobId" = $1
						LIMIT 1
					"""
					tempTableResult = await tempSQL.fetchrow( tempQuery, tempJobSQL.JobId )

					if ( tempTableResult is not None ):
						tempTableSQL = SQLTable( tempTableResult[ 0 ], tempTableResult[ 1 ], tempTableResult[ 2 ] )
						tempJobTrainSetSQL = SQLJobTrainSet( tempTableResult[ 3 ], tempTableResult[ 4 ], tempTableResult[ 5 ] )

					tempClaimedJob = ModelClaimedJob( tempGroupSQL, tempJobSQL, tempTableSQL, tempJobTrainSetSQL )

				self._activeJobIds.add( tempClaimedJob.Job.JobId )

				return tempClaimedJob
			except Exception as tException:
				logger.error( f"Failed to get Claimed Job: {tException}" )
				return None
			finally:
				if ( tempSQL is not None ):
					await tempSQL.close()

	async def _UpdateJobElapsedAsync( self, tJobId: int, tElapsed: timedelta ) -> bool:
		tempSQL = None

		try:
			tempSQL = await self._GetSQLAsync()
			tempQuery = """
				UPDATE "Job"
				SET "Elapsed" = $1
				WHERE "JobId" = $2
			"""
			tempStatus = await tempSQL.execute( tempQuery, tElapsed, tJobId )

			return tempStatus == "UPDATE 1"
		except Exception as tException:
			logger.error( f"Failed to update Job to DB: {tException}" )
			raise
		finally:
			if ( tempSQL is not None ):
				await tempSQL.close()