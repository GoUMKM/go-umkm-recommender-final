from fastapi import FastAPI, HTTPException
import tensorflow as tf
import joblib
import pandas as pd
from pathlib import Path
import requests

from app.database import DatabaseConnector
from app.preprocessing import DataPreprocessor
from app.schemas import RecommendationRequest, RecommendationResponse
from app.config import settings  # Konfigurasi URL model

app = FastAPI(title="Go-UMKM Recommendation API")

# Path dan variable global
artifacts_path = Path("artifacts")
preprocessor = None
model = None
users_df = None
umkm_df = None
investor_df = None
similarity_matrix = None

def download_file(url: str, dest_path: str):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded {dest_path}")
    else:
        raise Exception(f"Failed to download {url}, status: {response.status_code}")

@app.on_event("startup")
async def load_artifacts():
    global preprocessor, model, users_df, umkm_df, investor_df, similarity_matrix
    try:
        # Download dari model registry
        download_file(settings.MODEL_URL, artifacts_path / "similarity_model.h5")
        download_file(settings.PREPROCESSOR_URL, artifacts_path / "preprocessor.joblib")

        # Load model dan preprocessor
        preprocessor = joblib.load(artifacts_path / "preprocessor.joblib")
        model = tf.keras.models.load_model(artifacts_path / "similarity_model.h5")

        # Load dan proses data dari DB
        db_connector = DatabaseConnector()
        data_processor = DataPreprocessor()
        data = db_connector.load_required_tables()
        merged_data = data_processor.merge_dataframes(data)
        umkm_df = merged_data['umkm']
        investor_df = merged_data['investor']

        umkm_clean = umkm_df.drop(columns=['umkm_id'])
        investor_clean = investor_df.drop(columns=['investor_id'])
        users_df = pd.concat([umkm_clean, investor_clean], ignore_index=True)

        # Generate embedding dan similarity matrix
        if model and preprocessor:
            features = preprocessor.transform(users_df)
            dense_features = features.toarray() if hasattr(features, 'toarray') else features
            embeddings = model.predict(dense_features)
            normalized = tf.math.l2_normalize(embeddings, axis=1)
            similarity_matrix = tf.linalg.matmul(normalized, normalized, transpose_b=True)

    except Exception as e:
        print(f"Error during startup: {e}")
    finally:
        if 'db_connector' in locals():
            db_connector.close()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

def get_profile_type(user_id: str) -> str:
    if user_id in umkm_df['user_id'].values:
        return 'umkm'
    elif user_id in investor_df['user_id'].values:
        return 'investor'
    else:
        raise ValueError(f"User ID {user_id} not found")

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        user_type = get_profile_type(request.user_id)
        idx = users_df.index[users_df['user_id'] == request.user_id].tolist()[0]
        sim_scores = list(enumerate(similarity_matrix[idx].numpy()))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:request.top_k+1]
        user_indices = [i[0] for i in sim_scores]
        recommended_user_ids = users_df.iloc[user_indices]['user_id'].tolist()

        if user_type == 'umkm':
            recommended_ids = investor_df[investor_df['user_id'].isin(recommended_user_ids)]['investor_id'].tolist()
        else:
            recommended_ids = umkm_df[umkm_df['user_id'].isin(recommended_user_ids)]['umkm_id'].tolist()

        return RecommendationResponse(recommendations=recommended_ids)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
