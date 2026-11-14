"""Cooperative time/signal limits; completed per-case records remain resumable."""
from contextlib import contextmanager
import signal
import threading
import time


class EvaluationPaused(RuntimeError):
    pass


class RunBudget:
    def __init__(self, max_seconds=None, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.max_seconds = max_seconds
        self.reason = None

    def stop(self, reason="signal"):
        self.reason = reason

    def exhausted(self):
        if self.reason is not None:
            return True
        if self.max_seconds is not None and self.clock() - self.started >= self.max_seconds:
            self.reason = "time_budget"
            return True
        return False

    @contextmanager
    def signals(self):
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGTERM", "SIGUSR1"):
                sig = getattr(signal, name, None)
                if sig is not None:
                    previous[sig] = signal.getsignal(sig)
                    signal.signal(sig, lambda signum, frame: self.stop(f"signal_{signum}"))
        try:
            yield self
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def select_samples(samples, start_index=0, stop_index=None, limit=None):
    if start_index < 0 or (stop_index is not None and stop_index <= start_index):
        raise ValueError("Expected 0 <= start_index < stop_index (stop is exclusive)")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    selected = samples[start_index:stop_index]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Selected sample range is empty")
    return selected
