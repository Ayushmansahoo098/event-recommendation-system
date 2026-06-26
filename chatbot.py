import time
import json
import uuid
import datetime
from typing import Dict, Any, List, Tuple, Optional
from recommender import recommender
from gemini_client import gemini_client
from prompts import SYSTEM_PROMPT_TEMPLATE
from event_search import parse_query_filters, merge_filters, search_events

class ConversationMemoryManager:
    def __init__(self):
        # Maps conversationId -> {"history": [messages], "active_filters": {filters}, "updated_at": datetime}
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get_session(self, conversation_id: str) -> Dict[str, Any]:
        """Retrieves or creates a session by its conversationId."""
        if not conversation_id:
            # Create a transient session ID
            conversation_id = str(uuid.uuid4())
            
        if conversation_id not in self._sessions:
            if len(self._sessions) >= 100:
                oldest_conv = min(self._sessions.keys(), key=lambda k: self._sessions[k]["updated_at"])
                del self._sessions[oldest_conv]
                
            self._sessions[conversation_id] = {
                "history": [],
                "active_filters": {},
                "updated_at": datetime.datetime.now()
            }
        else:
            self._sessions[conversation_id]["updated_at"] = datetime.datetime.now()
            
        return conversation_id, self._sessions[conversation_id]

    def add_message(self, conversation_id: str, role: str, content: str):
        """Appends a message to the conversation history (keeps last 10 messages)."""
        _, session = self.get_session(conversation_id)
        session["history"].append({"role": role, "content": content})
        # Keep sliding window of last 10 messages
        if len(session["history"]) > 10:
            session["history"] = session["history"][-10:]

    def update_filters(self, conversation_id: str, filters: Dict[str, Any]):
        """Saves current active filters in the session."""
        _, session = self.get_session(conversation_id)
        session["active_filters"] = filters

# Global memory manager instance
memory_manager = ConversationMemoryManager()

def get_user_profile(user_id: Optional[str]) -> Tuple[List[str], List[str]]:
    """Fetches user interests and preferred cities, using memory cache first to reduce Firestore reads."""
    if not user_id:
        return [], []

    # Check in-memory cache first
    cached = recommender.user_embedding_cache.get(user_id)
    if cached:
        return cached.get("interests", []), cached.get("preferred_cities_lower", [])

    if not recommender.db:
        return [], []
    try:
        user_snap = recommender.db.collection("users").document(user_id).get()
        if user_snap.exists:
            data = user_snap.to_dict()
            return data.get("interests", []) or [], data.get("preferredCities", []) or []
    except Exception as e:
        print(f"Error fetching user profile in chatbot.py: {e}")
    return [], []

def format_events_context(events: List[Dict[str, Any]]) -> str:
    """Formats events list into structured text block for Gemini grounding context."""
    if not events:
        return "No events found matching the criteria."
        
    lines = []
    for idx, e in enumerate(events):
        lines.append(f"Event {idx+1}:")
        lines.append(f"  ID: {e.get('id')}")
        lines.append(f"  Title: {e.get('title')}")
        lines.append(f"  Category: {e.get('category')}")
        lines.append(f"  City: {e.get('city')}")
        lines.append(f"  Date: {e.get('date')}")
        lines.append(f"  Time: {e.get('time') or 'Not specified'}")
        lines.append(f"  Location: {e.get('location') or 'Not specified'}")
        lines.append(f"  Online: {e.get('isOnline')}")
        lines.append(f"  Organizer: {e.get('organizer') or 'Unknown'}")
        lines.append(f"  Tags: {', '.join(e.get('tags', []))}")
        lines.append(f"  Description: {e.get('description') or 'No description available'}")
        lines.append(f"  Registration Link: {e.get('registrationUrl') or 'Check details'}")
        lines.append("")
    return "\n".join(lines)

async def handle_agent_chat(
    message: str,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """
    Orchestrates the AI Event Agent workflow:
    1. Retrieve conversation history & filters state.
    2. Extract filters from message & merge with previous active filters.
    3. Filter in-memory events cache first (Search-First).
    4. Compile Gemini prompt.
    5. Call Gemini in JSON mode (getting intent, reply, suggestions).
    6. Log latencies & token counts.
    """
    total_start = time.perf_counter()

    # 1. Session and Profile Setup
    conv_id, session = memory_manager.get_session(conversation_id)
    interests, preferred_cities = get_user_profile(user_id)

    # 2. Extract and merge filters (Context-Aware parsing)
    new_filters = parse_query_filters(message, preferred_cities)
    merged_filters = merge_filters(session["active_filters"], new_filters)
    memory_manager.update_filters(conv_id, merged_filters)

    # 3. Search-First Filtering
    search_start = time.perf_counter()
    matching_events = search_events(message, merged_filters, user_id)
    search_latency_ms = (time.perf_counter() - search_start) * 1000

    # 4. Construct System Prompt & context
    events_text = format_events_context(matching_events)
    current_dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    system_instruction = SYSTEM_PROMPT_TEMPLATE.format(
        events_context=events_text,
        user_interests=", ".join(interests) if interests else "None",
        current_date=current_dt_str
    )

    # 5. Call Gemini
    response_json, prompt_tokens, response_tokens, gemini_latency = await gemini_client.generate_response(
        system_instruction=system_instruction,
        history=session["history"],
        message=message
    )
    gemini_latency_ms = gemini_latency * 1000

    # 6. Update Session History with user query and model reply
    memory_manager.add_message(conv_id, "user", message)
    memory_manager.add_message(conv_id, "model", response_json.get("reply", ""))

    total_latency_ms = (time.perf_counter() - total_start) * 1000

    # 7. Print structured JSON performance logs
    log_payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "conversation_id": conv_id,
        "user_id": user_id,
        "search_time_ms": round(search_latency_ms, 2),
        "gemini_time_ms": round(gemini_latency_ms, 2),
        "total_time_ms": round(total_latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens
    }
    print(f"PERFORMANCE_METRICS: {json.dumps(log_payload)}")

    return response_json, matching_events, conv_id
