from dataclasses import dataclass
from io import BytesIO
from typing import List
from Common.Row import Row
from Common.DataType import DataType

@dataclass
class Table():
	Type: DataType
	Rows: List[ Row ]