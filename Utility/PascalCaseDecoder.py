import json
import re

class PascalCaseDecoder( json.JSONDecoder ):
	def decode( self, tString: str, *args, **kwargs ):
		tempData = super().decode( tString, *args, **kwargs )
		return self._ConvertKeys( tempData )

	def _ConvertKeys( self, tObject ):
		if ( isinstance( tObject, dict ) ):
			tempNewObject = {}
			for tempKey, tempValue in tObject.items():
				tempNewKey = self.GetPascaleCase( tempKey )
				tempNewObject[ tempNewKey ] = self._ConvertKeys( tempValue )
				
			return tempNewObject
		elif ( isinstance( tObject, list ) ):
			return [ self._ConvertKeys( i ) for i in tObject ]
		else:
			return tObject
        
	def GetPascaleCase( self, tString: str ):
		return "".join( tempWord.capitalize() for tempWord in re.split( "_|(?=[A-Z])", tString ) )