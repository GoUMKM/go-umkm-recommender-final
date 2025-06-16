from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    user_id: str
    top_k: int = 10

class RecommendationResponse(BaseModel):
    recommendations: list[str]  # List of UMKM/Investor IDs