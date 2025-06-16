import joblib
from pathlib import Path
from src.data.database import DatabaseConnector
from src.data.preprocessing import DataPreprocessor
from src.models.similarity_model import SimilarityModel
from src.config import settings  # Perubahan import

def train_and_save_model():
    """Main training pipeline"""
    # Initialize components
    db_connector = DatabaseConnector()
    preprocessor = DataPreprocessor()
    
    try:
        # Load and preprocess data
        data = db_connector.load_required_tables()
        merged_data = preprocessor.merge_dataframes(data)
        features = preprocessor.preprocess_data(merged_data['umkm'], merged_data['investor'])
        
        # Train model
        model = SimilarityModel(input_dim=features.shape[1])
        model.compile_model()
        embeddings = model.generate_embeddings(features)
        similarity_matrix = model.compute_similarity_matrix(embeddings)
        
        # Save artifacts
        Path("artifacts").mkdir(exist_ok=True)
        joblib.dump(preprocessor.preprocessor, "artifacts/preprocessor.joblib")
        model.model.save("artifacts/similarity_model.h5")
        
        print("Training completed and artifacts saved successfully.")
        
        return {
            "status": "success",
            "message": "Model trained and artifacts saved",
            "input_dim": features.shape[1],
            "num_samples": features.shape[0]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        db_connector.close()

if __name__ == "__main__":
    train_and_save_model()