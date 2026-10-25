"""Human-readable local activity logs, including bounded waiting heartbeats."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4


LOG_PATH = Path(__file__).resolve().parent / "data" / "logs" / "activite_llm.log"
HEARTBEAT_SECONDS = 10
_setup_lock = threading.Lock()
_context = ContextVar("rp_dataset_activity", default={})


def get_logger():
    logger = logging.getLogger("rp_dataset_studio.activity")
    with _setup_lock:
        if not any(getattr(handler, "_rpds_activity", False) for handler in logger.handlers):
            logger.setLevel(logging.INFO)
            logger.propagate = False
            formatter = logging.Formatter("%(asctime)s [ACTIVITÉ] %(message)s", datefmt="%H:%M:%S")
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(formatter)
            console._rpds_activity = True
            logger.addHandler(console)
            try:
                LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                journal = RotatingFileHandler(LOG_PATH, maxBytes=2 * 1024 ** 2, backupCount=2,
                                               encoding="utf-8")
                journal.setFormatter(formatter)
                journal._rpds_activity = True
                logger.addHandler(journal)
            except OSError:
                logger.warning("Journal sur disque indisponible ; les messages restent visibles dans la console.")
    return logger


def _safe(value):
    return " ".join(str(value).replace("\x1b", "").split())[:300]


def event(message, *, level=logging.INFO, **fields):
    values = {**_context.get(), **fields}
    detail = " | ".join(f"{key}={_safe(value)}" for key, value in values.items() if value is not None)
    get_logger().log(level, "%s%s", _safe(message), " | " + detail if detail else "")


class Activity:
    def __init__(self, title, *, heartbeat=True, interval=HEARTBEAT_SECONDS, **fields):
        self.title, self.fields = title, fields
        self.heartbeat, self.interval = heartbeat, interval
        self.stop = threading.Event()
        self.write_lock = threading.Lock()
        self.thread = None

    def __enter__(self):
        self.fields = {**_context.get(), **self.fields}
        self.fields.setdefault("tache", uuid4().hex[:8])
        self.token = _context.set(self.fields)
        self.started = time.monotonic()
        event("DÉBUT — " + self.title)
        if self.heartbeat:
            self.thread = threading.Thread(target=self._wait, daemon=True)
            self.thread.start()
        return self

    def _wait(self):
        while not self.stop.wait(self.interval):
            with self.write_lock:
                if not self.stop.is_set():
                    event("EN COURS — " + self.title, **self.fields,
                          attente_s=round(time.monotonic() - self.started, 1))

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        try:
            with self.write_lock:
                values = {**self.fields, "duree_s": round(time.monotonic() - self.started, 2)}
                if exc_type is None:
                    event("FIN — " + self.title, **values)
                else:
                    event("ÉCHEC — " + self.title, level=logging.ERROR,
                          erreur=f"{exc_type.__name__}: {exc}", **values)
        finally:
            _context.reset(self.token)
        return False


@contextmanager
def shared_lock(lock, title="Attente du moteur partagé"):
    """Only log lock contention, so a queued request cannot look like inference."""
    if not lock.acquire(blocking=False):
        with Activity(title):
            lock.acquire()
    try:
        yield
    finally:
        lock.release()
