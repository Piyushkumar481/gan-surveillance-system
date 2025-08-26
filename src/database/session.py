"""
Database session management with connection pooling and context management
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from typing import Generator, Optional
import threading
from src.utils.logger import logger

class DatabaseSessionManager:
    def __init__(self, db_url: str = "sqlite:///anomalies.db", echo: bool = False):
        self.engine = create_engine(
            db_url,
            echo=echo,
            pool_size=10,           # Maximum number of connections in pool
            max_overflow=20,        # Maximum overflow connections
            pool_timeout=30,        # Timeout for getting connection
            pool_recycle=3600,      # Recycle connections after 1 hour
            pool_pre_ping=True      # Test connections for health
        )
        
        self.session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
        
        # Scoped session for thread safety
        self.scoped_session = scoped_session(self.session_factory)
        
        logger.info(f"Database session manager initialized with URL: {db_url}")
    
    @contextmanager
    def get_session(self) -> Generator:
        """Get a database session with context management"""
        session = self.scoped_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
            self.scoped_session.remove()
    
    def get_raw_session(self):
        """Get a raw session without context management"""
        return self.scoped_session()
    
    def close_all_sessions(self):
        """Close all database sessions"""
        self.scoped_session.remove()
        logger.info("All database sessions closed")
    
    def health_check(self) -> bool:
        """Check database connection health"""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_engine_stats(self) -> dict:
        """Get database connection pool statistics"""
        return {
            "checked_out": self.engine.pool.checkedout(),
            "checked_in": self.engine.pool.checkedin(),
            "size": self.engine.pool.size(),
            "overflow": self.engine.pool.overflow()
        }

# Global session manager instance
_session_manager: Optional[DatabaseSessionManager] = None

def init_session_manager(db_url: str = "sqlite:///anomalies.db", echo: bool = False):
    """Initialize the global session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = DatabaseSessionManager(db_url, echo)
    return _session_manager

def get_session_manager() -> DatabaseSessionManager:
    """Get the global session manager"""
    if _session_manager is None:
        raise RuntimeError("Session manager not initialized. Call init_session_manager() first.")
    return _session_manager

@contextmanager
def session_scope() -> Generator:
    """Global session context manager for easy database access"""
    manager = get_session_manager()
    with manager.get_session() as session:
        yield session

# Utility functions for common operations
def execute_query(query: str, params: dict = None):
    """Execute a raw SQL query"""
    with session_scope() as session:
        result = session.execute(query, params or {})
        return result

def bulk_insert(model_class, data: list):
    """Bulk insert data into a table"""
    with session_scope() as session:
        session.bulk_insert_mappings(model_class, data)
        session.commit()

def bulk_update(model_class, data: list, update_fields: list):
    """Bulk update records"""
    with session_scope() as session:
        session.bulk_update_mappings(model_class, data)
        session.commit()

# Thread-local session management for web applications
_thread_local = threading.local()

def get_thread_local_session():
    """Get a session for the current thread (for web applications)"""
    if not hasattr(_thread_local, "session"):
        manager = get_session_manager()
        _thread_local.session = manager.get_raw_session()
    return _thread_local.session

def close_thread_local_session():
    """Close the thread-local session"""
    if hasattr(_thread_local, "session"):
        _thread_local.session.close()
        delattr(_thread_local, "session")