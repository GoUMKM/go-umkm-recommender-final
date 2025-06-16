import pandas as pd
from sqlalchemy import create_engine
from typing import List, Tuple
from src.config import settings  # Perubahan import

class DatabaseConnector:
    def __init__(self):
        self.engine = self._create_engine()
    
    def _create_engine(self):
        """Create and return a database connection engine with connection pooling"""
        connection_string = (
            f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}@"
            f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        return create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=1800
        )
    
    def load_required_tables(self) -> List[Tuple[str, pd.DataFrame]]:
        """Load only required tables into a list of tuples (table_name, DataFrame)"""
        tables = [
            'users', 'umkm_profiles', 'investor_profiles',
            'umkm_kategori_usaha', 'umkm_model_bisnis', 'umkm_skala_usaha', 'umkm_jangkauan_pasar',
            'investor_kategori_usaha', 'investor_model_bisnis', 'investor_skala_usaha', 'investor_jangkauan_pasar'
        ]
        return [(table, pd.read_sql(f"SELECT * FROM {table}", self.engine)) for table in tables]
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'engine'):
            self.engine.dispose()