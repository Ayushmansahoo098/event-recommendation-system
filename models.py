from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    userId: Optional[str] = Field(None, description="Optional Firestore user ID")
    conversationId: Optional[str] = Field(None, description="Optional unique conversation session ID")
    message: str = Field(..., description="The natural language query from the user")

class EventResponse(BaseModel):
    id: str = Field(..., description="Event unique identifier (Firestore document ID)")
    title: str = Field(..., description="Event title")
    description: Optional[str] = Field(None, description="Full description of the event")
    category: str = Field(..., description="Event category (e.g. hackathon, workshop)")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    city: str = Field(..., description="City where the event is located")
    date: str = Field(..., description="Date of the event (YYYY-MM-DD)")
    time: Optional[str] = Field(None, description="Time of the event")
    location: Optional[str] = Field(None, description="Detailed location or venue")
    isOnline: bool = Field(..., description="Whether the event is online/virtual")
    registrationUrl: Optional[str] = Field(None, description="URL for registration")
    bannerImage: Optional[str] = Field(None, description="Banner image URL")
    organizer: Optional[str] = Field(None, description="Event organizer name")
    viewsCount: Optional[int] = Field(0, alias="views", description="Views count")
    savesCount: Optional[int] = Field(0, alias="saves", description="Saves count")
    registrationsCount: Optional[int] = Field(0, alias="registrations", description="Registrations count")
    score: Optional[float] = Field(None, description="Relevance score (cosine similarity)")
    matchScore: Optional[int] = Field(None, description="Match score out of 100")
    reason: Optional[str] = Field(None, description="Human-readable citation/reason for match")

    class Config:
        populate_by_name = True

class ChatResponse(BaseModel):
    intent: str = Field(..., description="Classified intent (e.g. find_events, recommend, compare, etc.)")
    reply: str = Field(..., description="Natural language response from the AI assistant")
    events: List[EventResponse] = Field(default_factory=list, description="List of matched events from context")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestion query templates")
