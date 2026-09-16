import asyncio
import multiprocessing

if ( __name__ == "__main__" ):
	multiprocessing.freeze_support()

	from Startup import Startup

	tempStartup = Startup()
	asyncio.run( tempStartup.StartAsync() )