"""Player — replay Parquet."""
import pyarrow.parquet as pq
from pathlib import Path

class Player:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.table = pq.read_table(str(self.path))
        self.rows = self.table.to_pylist()
        self.idx = 0

    def next(self):
        if self.idx >= len(self.rows):
            return None
        row = self.rows[self.idx]
        self.idx += 1
        return row
