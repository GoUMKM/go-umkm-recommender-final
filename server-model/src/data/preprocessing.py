import pandas as pd
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = self._create_preprocessor()
    
    def _create_preprocessor(self):
        """Create the preprocessing pipeline"""
        text_features = ['kategori', 'model_bisnis']
        categorical_features = ['skala', 'jangkauan']

        return ColumnTransformer(
            transformers=[
                ('text_kat', TfidfVectorizer(), 'kategori'),
                ('text_model_bisnis', TfidfVectorizer(), 'model_bisnis'),
                ('cat', OneHotEncoder(), categorical_features)
            ])
    
    def merge_dataframes(self, data: List[Tuple[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Process the data and return UMKM and Investor DataFrames"""
        data_dict = {table: df for table, df in data}

        users = data_dict['users']
        umkm_profiles = data_dict['umkm_profiles']
        investor_profiles = data_dict['investor_profiles']

        # Process UMKM data
        umkm = pd.merge(
            users[users['tipe_akun'].str.lower() == 'umkm'],
            umkm_profiles,
            left_on='user_id',
            right_on='umkm_id',
            how='inner'
        )

        # Process UMKM preferences as strings
        for table, column in [
            ('umkm_kategori_usaha', 'kategori'),
            ('umkm_model_bisnis', 'model_bisnis'),
            ('umkm_skala_usaha', 'skala'),
            ('umkm_jangkauan_pasar', 'jangkauan')
        ]:
            grouped = data_dict[table].groupby('umkm_id')[column].apply(lambda x: ', '.join(map(str, x))).reset_index()
            umkm = pd.merge(umkm, grouped, on='umkm_id', how='left')

        umkm = umkm.astype(str)
        umkm = umkm[['user_id', 'umkm_id', 'kategori', 'model_bisnis', 'skala', 'jangkauan']]

        # Process Investor data
        investor = pd.merge(
            users[users['tipe_akun'].str.lower() == 'investor'],
            investor_profiles,
            left_on='user_id',
            right_on='investor_id',
            how='inner'
        )

        # Process Investor preferences as strings
        for table, column in [
            ('investor_kategori_usaha', 'kategori'),
            ('investor_model_bisnis', 'model_bisnis'),
            ('investor_skala_usaha', 'skala'),
            ('investor_jangkauan_pasar', 'jangkauan')
        ]:
            grouped = data_dict[table].groupby('investor_id')[column].apply(lambda x: ', '.join(map(str, x))).reset_index()
            investor = pd.merge(investor, grouped, on='investor_id', how='left')

        investor = investor.astype(str)
        investor = investor[['user_id', 'investor_id', 'kategori', 'model_bisnis', 'skala', 'jangkauan']]

        return {'umkm': umkm, 'investor': investor}
    
    def preprocess_data(self, umkm_df: pd.DataFrame, investor_df: pd.DataFrame) -> pd.DataFrame:
        """Combine and preprocess the data"""
        umkm_clean = umkm_df.drop(columns=['umkm_id'])
        investor_clean = investor_df.drop(columns=['investor_id'])
        users_df = pd.concat([umkm_clean, investor_clean], ignore_index=True)
        
        # Fit and transform the data
        features = self.preprocessor.fit_transform(users_df)
        return features.toarray() if hasattr(features, 'toarray') else features