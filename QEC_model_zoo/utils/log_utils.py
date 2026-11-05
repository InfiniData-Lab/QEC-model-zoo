import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
import sys
from pathlib import Path

from tqdm import tqdm


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def capture_output_to_file(path: Path) -> None:
    """
    Capture stdout and stderr to a file.

    :param path: Path to file to save to
    """
    buffer = StringIO()

    tee_stdout = Tee(sys.__stdout__, buffer)
    tee_stderr = Tee(sys.__stderr__, buffer)

    try:
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            yield buffer
    finally:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            f.write(buffer.getvalue())

class _TqdmLoggingHandler(logging.Handler):
    """Routes log records through tqdm.write() so they don't mangle the progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)
