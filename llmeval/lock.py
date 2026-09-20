"""One machine-wide GPU lock so two model-loading jobs can never overlap (also honoured by the legacy flock scripts)."""
import contextlib, fcntl, sys

LOCK_PATH = "/tmp/local-llm-gpu.lock"

@contextlib.contextmanager
def gpu_lock(path=LOCK_PATH):
    f = open(path, "w")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("waiting for the GPU lock (another llmeval job is running)...", file=sys.stderr, flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN); f.close()
