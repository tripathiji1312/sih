"""Recorder — save sessions to Parquet."""
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

class Recorder:
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.frames = []

    def record(self, frame_dict: dict):
        self.frames.append(frame_dict)

    def save(self):
        if not self.frames:
            return
        table = pa.Table.from_pylist(self.frames)
        pq.write_table(table, str(self.output_path))
