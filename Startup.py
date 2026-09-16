import asyncio
import json
import logging
import logging.config
import os
import signal
import traceback
from Slave.Service import Service as SlaveService
from Slave.Settings import Settings as SlaveSettings
from FTP.Service import Service as FTPService
from FTP.Settings import Settings as FTPSettings
from Sweeper.Settings import Settings as SweeperSettings
from Utility.CancellationToken import CancellationToken

class Startup:
	_slave: SlaveService
	_FTP: FTPService
	_cancellationToken: CancellationToken
	_sweeperSettings: SweeperSettings

	def __init__( self ):
		# Load Config
		with open( "appsettings.json", "r" ) as tempSettings:
			tempConfig = json.load( tempSettings )

		self.UpdateConfig( tempConfig, self.ParseEnvironmentVariables() )

		# Logging
		logging.config.dictConfig( tempConfig[ "Logging" ] )
	
		# Services
		self._cancellationToken = CancellationToken()
		self._slave = SlaveService( SlaveSettings( **tempConfig[ "Slave" ] ) )
		self._FTP = FTPService( FTPSettings( **tempConfig[ "FTP" ] ) )
		self._sweeperSettings = SweeperSettings( **tempConfig[ "Sweeper" ] )

	async def StartAsync( self ):
		# Register Signals for graceful shutdown
		try:
			tempLoop = asyncio.get_running_loop()
			tempSignals = ( signal.SIGINT, signal.SIGTERM )
			
			for tempSignal in tempSignals:
				tempLoop.add_signal_handler( tempSignal, lambda: asyncio.create_task( self.StopAsync() ) )
		except ( NotImplementedError, RuntimeError ):
			pass

		# Slave Service
		try:
			await self._slave.ExecuteAsync( self._FTP, self._sweeperSettings, self._cancellationToken )
		except Exception:
			print( "Stack Trace:" )
			traceback.print_exc()

			await self.StopAsync()
			raise
		
	async def StopAsync( self ):
		if ( not self._cancellationToken.IsCancelled ):
			print( "Shutting Down" )
			self._cancellationToken.Cancel()
		
	def UpdateConfig( self, tOriginal, tUpdates ):
		for key, value in tUpdates.items():
			if ( isinstance( value, dict ) and key in tOriginal ):
				self.UpdateConfig( tOriginal[ key ], value )
			else:
				tOriginal[ key ] = value

	def ParseEnvironmentVariables( self ) -> dict:
		tempOverrides = {}

		for key, value in os.environ.items():
			if ( key.startswith( "FARMPY__" ) ):
				tempParts = key[ len( "FARMPY__" ): ].split( "__" )
			else:
				tempParts = key.split( ":" )

			if ( len( tempParts ) <= 1 ):
				continue

			tempSubOverrides = tempOverrides

			for tempPart in tempParts[ :-1 ]:
				if ( tempPart not in tempSubOverrides ):
					tempSubOverrides[ tempPart ] = {}
				tempSubOverrides = tempSubOverrides[ tempPart ]

			tempSubOverrides[ tempParts[ -1 ] ] = value

		return tempOverrides