import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from recommender import recommender, calculate_recency_score, _is_active_event

# Normalize common city spellings
CITY_MAPPING = {
    "bengaluru": "bangalore",
    "bombay": "mumbai",
    "new delhi": "delhi",
    "gurugram": "gurgaon",
    "secunderabad": "hyderabad",
}

# List of known categories
KNOWN_CATEGORIES = ["hackathon", "meetup", "workshop", "startup", "concert", "music", "course", "tournament", "gaming", "conference", "summit"]

def parse_query_filters(message: str, preferred_cities: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parses a user message to extract structured filters:
    category, city, online, free, date_range, and keywords.
    """
    message_lower = message.lower().strip()
    filters = {}

    # 1. Parse City
    detected_city = None
    detected_city_pos = -1
    # Check for direct city mentions
    cities = ["bangalore", "bengaluru", "hyderabad", "delhi", "mumbai", "pune", "chennai", "kolkata", "noida", "gurgaon", "gurugram"]
    for city in cities:
        match = re.search(r"\b" + re.escape(city) + r"\b", message_lower)
        if match:
            pos = match.start()
            if pos > detected_city_pos:
                detected_city = CITY_MAPPING.get(city, city)
                detected_city_pos = pos
            
    # Check for "near me" or "nearby"
    if "near me" in message_lower or "nearby" in message_lower:
        if preferred_cities:
            # Fallback to the first preferred city of the user
            detected_city = preferred_cities[0].lower()
            detected_city = CITY_MAPPING.get(detected_city, detected_city)
            
    if detected_city:
        filters["city"] = detected_city

    # 2. Parse Category
    detected_cat = None
    detected_cat_pos = -1
    for cat in KNOWN_CATEGORIES:
        # Match singular or plural category names
        pattern = r"\b" + re.escape(cat) + r"s?\b"
        match = re.search(pattern, message_lower)
        if match:
            pos = match.start()
            if pos > detected_cat_pos:
                detected_cat = cat
                detected_cat_pos = pos
                
    if detected_cat:
        filters["category"] = detected_cat

    # 3. Parse Online/Offline
    if any(k in message_lower for k in ["online", "virtual", "remote", "zoom", "webinar"]):
        filters["isOnline"] = True
    elif any(k in message_lower for k in ["offline", "in-person", "physical", "venue", "near me"]):
        filters["isOnline"] = False

    # 4. Parse Free/Paid
    if any(k in message_lower for k in ["free", "no cost", "unpaid", "free of cost"]):
        filters["free"] = True
    elif any(k in message_lower for k in ["paid", "ticket", "charge", "price", "cost", "buy", "₹", "rs"]):
        filters["free"] = False

    # 5. Parse Date Filters
    today = datetime.now().date()
    start_date = None
    end_date = None

    if "today" in message_lower:
        start_date = today
        end_date = today
    elif "tomorrow" in message_lower:
        start_date = today + timedelta(days=1)
        end_date = today + timedelta(days=1)
    elif "this weekend" in message_lower or "weekend" in message_lower:
        # Weekend = Friday to Sunday
        if today.weekday() < 4:  # Mon - Thu
            start_date = today + timedelta(days=(4 - today.weekday()))
        else:
            start_date = today
        end_date = today + timedelta(days=(6 - today.weekday()))
    elif "this week" in message_lower:
        start_date = today
        # End of current week (Sunday)
        end_date = today + timedelta(days=(6 - today.weekday()))
    elif "this month" in message_lower:
        start_date = today
        # End of current calendar month
        next_month = today.replace(day=28) + timedelta(days=4)
        end_date = next_month - timedelta(days=next_month.day)
    else:
        # Match specific weekdays
        weekdays = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        for day_name, day_idx in weekdays.items():
            if day_name in message_lower:
                days_ahead = day_idx - today.weekday()
                if days_ahead <= 0:  # Target day is today or already passed this week
                    days_ahead += 7
                target_date = today + timedelta(days=days_ahead)
                start_date = target_date
                end_date = target_date
                break

    if start_date and end_date:
        filters["date_range"] = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    # 6. Check if personalized recommendation is requested
    if any(k in message_lower for k in ["recommend events based on my interests", "recommend based on my interests", "recommend events for me", "my interests", "my profile"]):
        filters["personal_recommendation"] = True

    # 7. Check if comparison is requested
    if any(k in message_lower for k in ["compare", "difference between", "versus", "vs"]):
        filters["compare"] = True

    return filters

def merge_filters(old_filters: Dict[str, Any], new_filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges filters from a previous conversational turn with filters from the latest message.
    New filters overwrite old filters, unless the user asks a follow-up filter query
    (e.g., "Are any of them free?" or "How about online?").
    """
    merged = old_filters.copy()
    
    # Overwrite/add new fields
    for k, v in new_filters.items():
        merged[k] = v
        
    # If the user specified a new city or category, we might want to clear previous conflicting filter states
    if "city" in new_filters and "personal_recommendation" in merged:
        merged.pop("personal_recommendation", None)
        
    return merged

def check_event_is_free(event: Dict[str, Any]) -> bool:
    """Helper to detect if an event is free based on tags and description."""
    title = event.get("title", "").lower()
    description = event.get("description", "").lower()
    tags = [t.lower() for t in event.get("tags", [])]
    return "free" in title or "free" in description or "free" in tags

def search_events(message: str, filters: Dict[str, Any], user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Applies hard filters to recommender.event_cache and ranks candidates
    using TF-IDF cosine similarity on the query.
    """
    recommender.initialize()
    if not recommender.event_cache:
        recommender.sync_embeddings()

    # Step 1: Handle personalized recommendation intent
    if filters.get("personal_recommendation") and user_id:
        # Run existing recommendation algorithm
        recs = recommender.get_recommendations(user_id=user_id, limit=20)
        # Convert back to event dict format
        results = []
        for rec in recs:
            evt_id = rec["eventId"]
            if evt_id in recommender.event_cache:
                event_data = recommender.event_cache[evt_id].copy()
                event_data["id"] = evt_id
                event_data["score"] = rec["score"]
                event_data["matchScore"] = rec["matchScore"]
                event_data["reason"] = rec["reason"]
                results.append(event_data)
        return results

    # Step 2: Extract hard parameters
    filter_city = filters.get("city")
    filter_cat = filters.get("category")
    filter_online = filters.get("isOnline")
    filter_free = filters.get("free")
    filter_date_range = filters.get("date_range")

    # Step 3: Hard filter loop over cached active events
    candidates = []
    for event_id, event in recommender.event_cache.items():
        if not _is_active_event(event):
            continue

        # City filter
        if filter_city:
            evt_city = event.get("city", "").lower()
            evt_city = CITY_MAPPING.get(evt_city, evt_city)
            if evt_city != filter_city:
                continue

        # Category filter
        if filter_cat:
            if event.get("category", "").lower() != filter_cat:
                continue

        # Online/Offline filter
        if filter_online is not None:
            if event.get("isOnline", False) != filter_online:
                continue

        # Free/Paid filter
        if filter_free is not None:
            is_free = check_event_is_free(event)
            if is_free != filter_free:
                continue

        # Date range filter
        if filter_date_range:
            evt_date_str = event.get("date", "")
            if not evt_date_str:
                continue
            start_str, end_str = filter_date_range
            if not (start_str <= evt_date_str <= end_str):
                continue

        # Add matching candidate
        evt_copy = event.copy()
        evt_copy["id"] = event_id
        candidates.append(evt_copy)

    if not candidates:
        return []

    # Step 4: Compute similarity scores for matched events using TF-IDF Vectorizer
    if message and recommender.vectorizer:
        try:
            query_vector = recommender.vectorizer.transform([message])
            for item in candidates:
                item_vector = item["embedding"]
                similarity = recommender._cosine_similarity(query_vector, item_vector)
                
                # Blended search score: similarity (70%) + recency score (20%) + popularity (10%)
                recency = calculate_recency_score(item.get("date", ""))
                
                views = item.get("views", 0)
                saves = item.get("saves", 0)
                regs = item.get("registrations", 0)
                pop_raw = views + saves * 3 + regs * 5
                popularity = min(1.0, pop_raw / 100.0)

                final_score = 0.70 * similarity + 0.20 * recency + 0.10 * popularity
                item["score"] = round(final_score, 4)
                item["matchScore"] = int(round(final_score * 100))
                
                # Generate matching reason/citation
                reasons = []
                if similarity > 0.15:
                    reasons.append("Semantic match")
                if filter_city:
                    reasons.append(f"Located in {item.get('city').title()}")
                if filter_online:
                    reasons.append("Virtual event")
                if filter_free:
                    reasons.append("Free entry")
                if recency >= 0.85:
                    reasons.append("Upcoming soon")
                
                item["reason"] = " • ".join(reasons) if reasons else "Recommended match"
        except Exception as e:
            print(f"Error calculating similarity scores: {e}")
            for item in candidates:
                item["score"] = 0.5
                item["matchScore"] = 50
                item["reason"] = "Relevant event match"

    # Sort descending by score
    candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return candidates[:10]  # Cap at top 10 relevant events
