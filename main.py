from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from recommender import recommender

app = FastAPI(
    title="Kairo Event Recommendation Engine",
    version="1.0.0",
    description="AI-powered content and behavioral recommendation system matching events to user preferences."
)

# Enable CORS for Next.js communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic input models
class RecommendationRequest(BaseModel):
    userId: str = Field(..., description="Firestore user ID to calculate personalized matches for")
    limit: Optional[int] = Field(20, description="Maximum number of recommendations to return")

class SimilarRequest(BaseModel):
    eventId: str = Field(..., description="Event ID to find matches for")
    limit: Optional[int] = Field(5, description="Maximum number of similar events to return")

@app.on_event("startup")
async def startup_event():
    # Warm up models and caches on server startup
    try:
        recommender.initialize()
        print("Recommender engine warmed up on startup successfully.")
    except Exception as e:
        print(f"Failed to initialize recommender on startup: {e}")

@app.post("/recommendations")
async def get_user_recommendations(req: RecommendationRequest):
    try:
        results = recommender.get_recommendations(user_id=req.userId, limit=req.limit)
        return {
            "success": True,
            "recommendedEvents": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal recommendation computation error: {str(e)}")

@app.post("/similar")
async def get_similar_events(req: SimilarRequest):
    try:
        results = recommender.get_similar_events(event_id=req.eventId, limit=req.limit)
        return {
            "success": True,
            "similarEvents": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal similarity match error: {str(e)}")

@app.post("/sync")
async def trigger_embedding_sync():
    """
    Triggers Firestore database streams to cache missing event embeddings immediately.
    """
    try:
        recommender.sync_embeddings()
        return {
            "success": True,
            "message": "Embeddings sync completed successfully.",
            "totalCached": len(recommender.event_cache)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute database sync: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "firebaseConnected": recommender.db is not None,
        "transformerLoaded": recommender.model is not None,
        "totalCachedEvents": len(recommender.event_cache)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
