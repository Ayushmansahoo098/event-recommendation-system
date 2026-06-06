import os

from fastapi import FastAPI, HTTPException, Query
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

# Pydantic input models for backward-compatibility POST endpoints
class RecommendationRequest(BaseModel):
    userId: str = Field(..., description="Firestore user ID to calculate personalized matches for")
    limit: Optional[int] = Field(20, description="Maximum number of recommendations to return")

class SimilarRequest(BaseModel):
    eventId: str = Field(..., description="Event ID to find matches for")
    limit: Optional[int] = Field(10, description="Maximum number of similar events to return")

@app.on_event("startup")
async def startup_event():
    # Warm up models and caches on server startup
    try:
        recommender.initialize()
        print("Recommender engine warmed up on startup successfully.")
    except Exception as e:
        print(f"Failed to initialize recommender on startup: {e}")

# Phase 3 - GET endpoint for User Recommendations
@app.get("/recommendations")
async def get_user_recommendations_get(
    userId: str = Query(..., description="Firestore user ID to calculate personalized matches for"),
    limit: int = Query(20, description="Maximum number of recommendations to return")
):
    try:
        results = recommender.get_recommendations(user_id=userId, limit=limit)
        return {
            "success": True,
            "recommendedEvents": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal recommendation computation error: {str(e)}")

# Backward-compatibility POST endpoint for User Recommendations
@app.post("/recommendations")
async def get_user_recommendations_post(req: RecommendationRequest):
    try:
        results = recommender.get_recommendations(user_id=req.userId, limit=req.limit)
        return {
            "success": True,
            "recommendedEvents": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal recommendation computation error: {str(e)}")

# Phase 2 - GET endpoint for Similar Events
@app.get("/similar")
async def get_similar_events_get(
    eventId: str = Query(..., description="Event ID to find matches for"),
    limit: int = Query(10, description="Maximum number of similar events to return")
):
    try:
        results = recommender.get_similar_events(event_id=eventId, limit=limit)
        return {
            "success": True,
            "similarEvents": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal similarity match error: {str(e)}")

# Backward-compatibility POST endpoint for Similar Events
@app.post("/similar")
async def get_similar_events_post(req: SimilarRequest):
    try:
        limit_val = req.limit if req.limit is not None else 10
        results = recommender.get_similar_events(event_id=req.eventId, limit=limit_val)
        return {
            "success": True,
            "similarEvents": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal similarity match error: {str(e)}")

# Cache Invalidation Endpoint
@app.post("/recommendations/invalidate")
async def invalidate_user_cache_route(
    userId: str = Query(..., description="User ID to invalidate embedding cache for")
):
    try:
        recommender.invalidate_user_cache(userId)
        return {
            "success": True,
            "message": f"Successfully invalidated cache for user {userId}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Embeddings Sync Endpoint
@app.post("/embeddings/sync")
async def sync_embeddings_route():
    try:
        recommender.sync_embeddings()
        return {
            "success": True,
            "message": "Embeddings sync completed successfully.",
            "totalCached": len(recommender.event_cache)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute database sync: {str(e)}")

# Backward-compatibility /sync POST route
@app.post("/sync")
async def trigger_embedding_sync():
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
    debug_info = recommender.get_cache_debug_info()
    return {
        "status": "healthy",
        "firebaseConnected": recommender.db is not None,
        "transformerLoaded": recommender.model is not None,
        "totalCachedEvents": len(recommender.event_cache),
        "activeEvents": debug_info["activeEvents"],
        "expiredEvents": debug_info["expiredEvents"],
        "totalCachedUserProfiles": len(recommender.user_embedding_cache),
        "averageRecommendationScore": recommender.get_average_recommendation_score(),
        "topCategories": recommender.get_top_categories_stats()
    }

@app.get("/debug/cache")
async def debug_cache():
    """Dev-only endpoint: inspect the event cache for debugging."""
    try:
        info = recommender.get_cache_debug_info()
        return {
            "success": True,
            **info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
