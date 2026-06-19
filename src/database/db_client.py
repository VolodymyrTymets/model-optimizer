import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.schema import Base

# Default local store for experiment-progress logging.
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/mode-optimizaer"


class DBClient:
    """Singleton access point to the database.

    Owns the single SQLAlchemy `Engine` and session factory for the process. The
    first construction fixes the connection URL; later calls return the same
    instance and ignore their arguments. Use `session_scope()` for transactional
    work — it commits on success, rolls back on error, and always closes.
    """

    _instance: Optional["DBClient"] = None
    _lock = threading.Lock()

    def __new__(cls, database_url: str = DEFAULT_DATABASE_URL) -> "DBClient":
        # Double-checked locking so concurrent callers share one instance.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._init(database_url)
                    cls._instance = instance
        return cls._instance

    def _init(self, database_url: str) -> None:
        self._database_url = database_url
        self._engine: Engine = create_engine(database_url)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        Base.registry.configure()

    @property
    def engine(self) -> Engine:
        return self._engine

    def create_database(self) -> None:
        """Create all tables declared on `Base` if they do not yet exist."""
        Base.metadata.create_all(self._engine)

    def session(self) -> Session:
        """Return a new session. The caller is responsible for closing it."""
        return self._session_factory()

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Transactional session: commit on success, rollback on error, always close."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
