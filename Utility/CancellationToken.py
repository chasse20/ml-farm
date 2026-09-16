from asyncio import Event

class CancellationToken:
	_cancelEvent: Event

	def __init__( self ):
		self._cancelEvent = Event()

	def Cancel( self ):
		self._cancelEvent.set()

	async def WaitForCancel( self ):
		await self._cancelEvent.wait()

	@property
	def IsCancelled( self ):
		return self._cancelEvent.is_set()