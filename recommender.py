import os
import math
import random
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
import firebase_admin
# pyrefly: ignore [missing-import]
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Interaction weights
WEIGHT_INTEREST = 4.0
WEIGHT_SEARCH = 2.0
WEIGHT_VIEW = 3.0
WEIGHT_SAVE = 7.0
WEIGHT_REGISTRATION = 10.0

# Base scoring weights
WEIGHT_SIMILARITY = 0.70
WEIGHT_POPULARITY_SCORE = 0.15
WEIGHT_RECENCY_SCORE = 0.10
WEIGHT_LOCATION_SCORE = 0.05

# Dwell-time multipliers
DWELL_DEEP = 2.0    # >= 30 seconds
DWELL_NORMAL = 1.0  # >= 10 seconds
DWELL_BOUNCE = 0.3  # < 10 seconds

# Diversity settings
DIVERSITY_POOL_SIZE = 60       # Take top N before diversity re-ranking
DIVERSITY_MIN_CATEGORIES = 3   # Ensure at least this many categories in output

# Tag overlap boost (added to final score when event tags match user interests)
WEIGHT_TAG_BOOST = 0.08


def calculate_recency_score(date_str: str) -> float:
    """
    Computes date proximity score with smooth decay:
    - past event: 0.0
    - today or tomorrow: 1.0
    - 2-7 days: 0.85
    - 8-14 days: 0.65
    - 15-30 days: 0.45
    - 31-60 days: 0.25
    - >60 days: 0.10
    """
    if not date_str:
        return 0.15
    try:
        today = datetime.now().date()
        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_until = (event_date - today).days

        if days_until < 0:
            return 0.0
        elif days_until <= 1:
            return 1.0
        elif days_until <= 7:
            return 0.85
        elif days_until <= 14:
            return 0.65
        elif days_until <= 30:
            return 0.45
        elif days_until <= 60:
            return 0.25
        else:
            return 0.10
    except Exception as e:
        print(f"Error parsing event date '{date_str}': {e}")
        return 0.15


def _is_active_event(event: dict) -> bool:
    """Check if an event is active (not expired, not in the past)."""
    status = event.get("status", "active")
    if status == "expired":
        return False
    date_str = event.get("date", "")
    if date_str:
        try:
            today = datetime.now().date()
            event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if event_date < today:
                return False
        except Exception:
            pass
    return True


class EventRecommender:
    def __init__(self):
        self.model = None
        self.db = None
        self.event_cache = {}  # event_id -> dict with embedding, title, city, category, tags, date, views, saves, registrations, status
        self.user_embedding_cache = {}  # user_id -> {"embedding": vector, "timestamp": datetime, ...}
        self.recommendation_scores_history = []
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return

        print("Initializing Recommendation Service...")

        # 1. Initialize Firebase Admin SDK
        firebase_project_id = os.getenv("FIREBASE_PROJECT_ID")
        firebase_client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
        firebase_private_key = os.getenv("FIREBASE_PRIVATE_KEY")

        if not firebase_project_id or not firebase_client_email or not firebase_private_key:
            print("WARNING: Firebase env variables are not fully configured. Running in offline/mock mode.")
            self.initialized = True
            return

        try:
            private_key = firebase_private_key.replace("\\n", "\n")
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": firebase_project_id,
                "client_email": firebase_client_email,
                "private_key": private_key,
                "token_uri": "https://oauth2.googleapis.com/token",
            })

            # Avoid initializing multiple times
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            print("Firebase Admin SDK connected successfully.")
        except Exception as e:
            print(f"Failed to initialize Firebase Admin SDK: {e}")

        # 2. Load Sentence Transformer Model
        try:
            print("Loading sentence-transformers (all-MiniLM-L6-v2)...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer loaded.")
        except Exception as e:
            print(f"Failed to load SentenceTransformer: {e}")

        self.initialized = True

        # 3. Initial embedding sync
        self.sync_embeddings()

    def sync_embeddings(self):
        """
        Fetches all events from Firestore, generates embeddings for
        title, description, tags, category, and stores them in memory.
        Also loads dates, views, saves, registrations, and status for filtering.
        """
        if not self.db or not self.model:
            print("Cannot sync: Recommender is in offline/mock mode.")
            return

        try:
            print("Syncing event embeddings (in-memory)...")
            events_ref = self.db.collection("events")
            docs = events_ref.stream()

            events_to_embed = []
            events_metadata = []

            for doc in docs:
                event_id = doc.id
                data = doc.to_dict()

                title = data.get("title") or ""
                description = data.get("description") or ""
                category = data.get("category") or ""
                tags = data.get("tags") or []
                city = data.get("city") or ""
                source = data.get("source") or ""
                event_date_str = data.get("date") or ""
                status = data.get("status") or "active"
                organizer = data.get("organizer") or ""

                views = data.get("viewsCount") or 0
                saves = data.get("savesCount") or 0
                regs = data.get("registrationsCount") or 0

                if isinstance(tags, list):
                    tags_str = " ".join([str(t) for t in tags if t])
                else:
                    tags_str = str(tags) if tags else ""

                # Combine text fields for embedding
                text_components = [title, description, category, tags_str]
                text_to_embed = " ".join([tc.strip() for tc in text_components if tc.strip()])

                events_to_embed.append(text_to_embed)
                events_metadata.append((
                    event_id, title, category, tags, city, source,
                    event_date_str, views, saves, regs, status, organizer
                ))

            if events_to_embed:
                print(f"Generating embeddings for {len(events_to_embed)} events...")
                embeddings = self.model.encode(events_to_embed)

                for i, (event_id, title, category, tags, city, source,
                        event_date_str, views, saves, regs, status, organizer) in enumerate(events_metadata):
                    self.event_cache[event_id] = {
                        "embedding": embeddings[i],
                        "title": title,
                        "category": category,
                        "tags": tags,
                        "city": city,
                        "source": source,
                        "date": event_date_str,
                        "views": views,
                        "saves": saves,
                        "registrations": regs,
                        "status": status,
                        "organizer": organizer,
                    }
                print(f"Successfully generated and cached {len(events_to_embed)} event embeddings in memory.")
            else:
                print("No events found in database.")
        except Exception as e:
            print(f"Error during in-memory embedding generation: {e}")
            raise e

    def _diversity_rerank(self, scored_items: list, limit: int) -> list:
        """
        Re-rank scored items using round-robin category interleaving.
        Ensures the output has representation from multiple categories
        while still respecting score ordering within each category.
        """
        if not scored_items:
            return []

        # Group by category
        category_buckets = defaultdict(list)
        for item in scored_items:
            cat = item.get("_category", "unknown")
            category_buckets[cat].append(item)

        # Sort each bucket by score descending (should already be, but ensure)
        for cat in category_buckets:
            category_buckets[cat].sort(key=lambda x: x["score"], reverse=True)

        # Round-robin interleave: pick from the category with the highest next score
        result = []
        category_counts = defaultdict(int)
        pointers = {cat: 0 for cat in category_buckets}

        while len(result) < limit:
            best_candidate = None
            best_score = -1
            best_cat = None

            for cat, items in category_buckets.items():
                ptr = pointers[cat]
                if ptr >= len(items):
                    continue

                candidate = items[ptr]
                # Apply a mild penalty if this category already has many picks
                # This encourages diversity without destroying score ordering
                count = category_counts[cat]
                diversity_penalty = 1.0 / (1.0 + 0.15 * count)
                adjusted_score = candidate["score"] * diversity_penalty

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_candidate = candidate
                    best_cat = cat

            if best_candidate is None:
                break

            result.append(best_candidate)
            category_counts[best_cat] += 1
            pointers[best_cat] += 1

        # Clean up internal fields
        for item in result:
            item.pop("_category", None)

        return result

    def get_similar_events(self, event_id: str, limit: int = 10):
        """
        Calculate cosine similarity between target event and all other active events.
        """
        self.initialize()

        if not self.event_cache:
            self.sync_embeddings()

        if event_id not in self.event_cache:
            print(f"Event ID {event_id} not found in cache.")
            return []

        target_event = self.event_cache[event_id]
        target_vector = target_event["embedding"].reshape(1, -1)
        target_title = target_event.get("title", "this event")
        target_category = target_event.get("category", "")

        similarities = []
        for other_id, other_event in self.event_cache.items():
            if other_id == event_id:
                continue

            # Skip expired/past events
            if not _is_active_event(other_event):
                continue

            other_vector = other_event["embedding"].reshape(1, -1)
            score = float(cosine_similarity(target_vector, other_vector)[0][0])
            score = max(0.0, score)

            # Generate reason
            other_category = other_event.get("category", "")
            other_title = other_event.get("title", "")
            if other_category and other_category.lower() == target_category.lower():
                reason = f"Related {other_category.title()} event similar to '{target_title}'"
            else:
                reason = f"Highly similar to '{target_title}'"

            similarities.append({
                "eventId": other_id,
                "score": round(score, 2),
                "matchScore": int(round(score * 100)),
                "reason": reason
            })

        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:limit]

    def get_recommendations(self, user_id: str, limit: int = 20):
        """
        Generates user recommendations using user profile vector and event features.
        Includes dwell-time weighting, expired event filtering, category diversity
        re-ranking, and specific reason generation.
        """
        self.initialize()

        if not self.event_cache:
            self.sync_embeddings()

        if not self.event_cache:
            return []

        # Check in-memory user profile cache
        user_profile_vector = None
        interests = []
        preferred_cities_lower = []
        user_registered_categories = {}   # category -> most recent event title
        user_saved_categories = {}        # category -> most recent event title
        user_viewed_categories = {}       # category -> most recent event title
        user_interest_tags = set()        # set of lowercase interest strings
        user_searches = []
        has_registered_event_ids = set()

        cached_entry = self.user_embedding_cache.get(user_id)
        if cached_entry:
            cached_at = cached_entry.get("timestamp")
            elapsed = (datetime.utcnow() - cached_at).total_seconds()
            if elapsed < 30 * 60:  # 30 minutes TTL
                user_profile_vector = cached_entry.get("embedding")
                interests = cached_entry.get("interests", [])
                preferred_cities_lower = cached_entry.get("preferred_cities_lower", [])
                user_registered_categories = cached_entry.get("registered_categories", {})
                user_saved_categories = cached_entry.get("saved_categories", {})
                user_viewed_categories = cached_entry.get("viewed_categories", {})
                user_interest_tags = cached_entry.get("interest_tags", set())
                user_searches = cached_entry.get("searches", [])
                has_registered_event_ids = cached_entry.get("registered_event_ids", set())
                print(f"User embedding cache HIT for user: {user_id} (elapsed: {elapsed:.1f}s)")
            else:
                print(f"User embedding cache EXPIRED for user: {user_id} (elapsed: {elapsed:.1f}s)")

        if user_profile_vector is None:
            # Cache MISS -> Query Firestore
            if self.db:
                try:
                    user_doc_ref = self.db.collection("users").document(user_id)
                    user_snap = user_doc_ref.get()
                    if user_snap.exists:
                        user_data = user_snap.to_dict()
                        interests = user_data.get("interests", []) or []
                        preferred_cities = user_data.get("preferredCities", []) or []
                        preferred_cities_lower = [c.lower() for c in preferred_cities if c]
                except Exception as e:
                    print(f"Error fetching user document: {e}")

            # Build interest tags set for tag-level matching
            user_interest_tags = set(i.lower() for i in interests if i)

            # Fetch User Analytics Activity logs
            activity_logs = []
            saved_event_ids = []
            if self.db:
                try:
                    activities_ref = self.db.collection("analytics_events")
                    query_snap = activities_ref.where("userId", "==", user_id).stream()
                    for doc in query_snap:
                        activity_logs.append(doc.to_dict())
                except Exception as query_err:
                    print(f"Error fetching analytics events for user {user_id}: {query_err}")

                # Fetch user saved bookmarks from subcollection
                try:
                    bookmarks_ref = self.db.collection("users").document(user_id).collection("bookmarks").stream()
                    for b in bookmarks_ref:
                        saved_event_ids.append(b.id)
                except Exception as b_err:
                    print(f"Error fetching user bookmarks: {b_err}")

            # Build vectors & weights arrays to calculate user centroid
            vectors = []
            weights = []

            # A. Process Explicit Interests (Weight = 4 each)
            if interests:
                interest_vectors = self.model.encode(interests)
                for vec in interest_vectors:
                    vectors.append(vec)
                    weights.append(WEIGHT_INTEREST)

            # B. Process Bookmarked Saved Events (Weight = 7)
            for doc_id in saved_event_ids:
                if doc_id in self.event_cache:
                    vectors.append(self.event_cache[doc_id]["embedding"])
                    weights.append(WEIGHT_SAVE)
                    cat = self.event_cache[doc_id].get("category", "")
                    title = self.event_cache[doc_id].get("title", "")
                    if cat:
                        user_saved_categories[cat.lower()] = title

            # C. Process Activity Log events with dwell-time weighting
            for act in activity_logs:
                action = act.get("action")
                event_id = act.get("eventId")
                dwell_time = act.get("dwellTime")

                if action == "register" and event_id in self.event_cache:
                    vectors.append(self.event_cache[event_id]["embedding"])
                    weights.append(WEIGHT_REGISTRATION)
                    has_registered_event_ids.add(event_id)
                    cat = self.event_cache[event_id].get("category", "")
                    title = self.event_cache[event_id].get("title", "")
                    if cat:
                        user_registered_categories[cat.lower()] = title

                elif action == "save" and event_id in self.event_cache:
                    vectors.append(self.event_cache[event_id]["embedding"])
                    weights.append(WEIGHT_SAVE)
                    cat = self.event_cache[event_id].get("category", "")
                    title = self.event_cache[event_id].get("title", "")
                    if cat:
                        user_saved_categories[cat.lower()] = title

                elif action == "view" and event_id in self.event_cache:
                    vectors.append(self.event_cache[event_id]["embedding"])
                    # Apply dwell-time tiered weighting
                    if dwell_time is not None and isinstance(dwell_time, (int, float)):
                        if dwell_time >= 30:
                            w = WEIGHT_VIEW * DWELL_DEEP
                        elif dwell_time >= 10:
                            w = WEIGHT_VIEW * DWELL_NORMAL
                        else:
                            w = WEIGHT_VIEW * DWELL_BOUNCE
                    else:
                        w = WEIGHT_VIEW * DWELL_NORMAL
                    weights.append(w)
                    cat = self.event_cache[event_id].get("category", "")
                    title = self.event_cache[event_id].get("title", "")
                    if cat:
                        user_viewed_categories[cat.lower()] = title

                elif action == "search" and act.get("query"):
                    query_str = act.get("query")
                    query_vec = self.model.encode([query_str])[0]
                    vectors.append(query_vec)
                    weights.append(WEIGHT_SEARCH)
                    user_searches.append(query_str)

            # Also populate saved categories from bookmark subcollection
            for doc_id in saved_event_ids:
                if doc_id in self.event_cache:
                    cat = self.event_cache[doc_id].get("category", "")
                    title = self.event_cache[doc_id].get("title", "")
                    if cat:
                        user_saved_categories[cat.lower()] = title

            # Generate Combined User Profile Vector
            if not vectors:
                print(f"User {user_id} has no profile features. Returning smart fallback.")
                return self._get_popularity_fallback(limit, preferred_cities_lower)

            user_profile_vector = np.average(vectors, axis=0, weights=weights).reshape(1, -1)

            # Save in-memory cache
            self.user_embedding_cache[user_id] = {
                "embedding": user_profile_vector,
                "interests": interests,
                "preferred_cities_lower": preferred_cities_lower,
                "registered_categories": user_registered_categories,
                "saved_categories": user_saved_categories,
                "viewed_categories": user_viewed_categories,
                "interest_tags": user_interest_tags,
                "searches": user_searches,
                "registered_event_ids": has_registered_event_ids,
                "timestamp": datetime.utcnow()
            }
            print(f"User embedding cache STORED for user: {user_id}")

        # Score all cached ACTIVE events
        max_views = max([e.get("views", 0) for e in self.event_cache.values()] or [1])
        max_saves = max([e.get("saves", 0) for e in self.event_cache.values()] or [1])
        max_regs = max([e.get("registrations", 0) for e in self.event_cache.values()] or [1])

        if max_views == 0: max_views = 1
        if max_saves == 0: max_saves = 1
        if max_regs == 0: max_regs = 1

        recommendations = []
        for other_id, other_event in self.event_cache.items():
            # Skip registered events
            if other_id in has_registered_event_ids:
                continue

            # Skip expired/past events
            if not _is_active_event(other_event):
                continue

            # A. Similarity (70%)
            other_vector = other_event["embedding"].reshape(1, -1)
            similarity = float(cosine_similarity(user_profile_vector, other_vector)[0][0])
            similarity = max(0.0, similarity)

            # B. Popularity (15%)
            views = other_event.get("views", 0)
            saves = other_event.get("saves", 0)
            regs = other_event.get("registrations", 0)

            norm_views = views / max_views
            norm_saves = saves / max_saves
            norm_regs = regs / max_regs

            popularity = 0.2 * norm_views + 0.3 * norm_saves + 0.5 * norm_regs

            # C. Recency (10%)
            event_date_str = other_event.get("date", "")
            recency = calculate_recency_score(event_date_str)

            # D. Location (5%)
            event_city = other_event.get("city", "").lower()
            is_location_match = False
            location = 0.0
            if event_city and event_city in preferred_cities_lower:
                location = 1.0
                is_location_match = True

            # Tag-level semantic boost when event tags overlap user interests
            event_tags_lower = {t.lower() for t in (other_event.get("tags") or []) if t}
            matching_tags = user_interest_tags.intersection(event_tags_lower)
            tag_boost = min(WEIGHT_TAG_BOOST, 0.04 * len(matching_tags)) if matching_tags else 0.0

            # Score Formula
            score = (
                WEIGHT_SIMILARITY * similarity +
                WEIGHT_POPULARITY_SCORE * popularity +
                WEIGHT_RECENCY_SCORE * recency +
                WEIGHT_LOCATION_SCORE * location +
                tag_boost
            )
            score = min(1.0, score)
            match_score = int(round(score * 100))

            # Rich reason generation
            reason = self._generate_reason(
                other_event, event_city, is_location_match,
                user_registered_categories, user_saved_categories,
                user_viewed_categories, user_interest_tags, user_searches,
                interests
            )

            recommendations.append({
                "eventId": other_id,
                "score": round(score, 4),
                "matchScore": match_score,
                "reason": reason,
                "_category": other_event.get("category", "unknown"),  # internal, for diversity
            })

        # Sort by score descending
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        # Apply diversity re-ranking on top pool
        pool = recommendations[:DIVERSITY_POOL_SIZE]
        diverse_recs = self._diversity_rerank(pool, limit)

        for rec in diverse_recs:
            self.recommendation_scores_history.append(rec["score"])
        return diverse_recs

    def _generate_reason(
        self, event: dict, event_city: str, is_location_match: bool,
        registered_cats: dict, saved_cats: dict, viewed_cats: dict,
        interest_tags: set, searches: list, interests: list
    ) -> str:
        """Generate a specific, human-readable reason for why this event is recommended."""
        event_category = event.get("category", "").lower()
        event_title = event.get("title", "")
        event_tags = [t.lower() for t in (event.get("tags") or [])]
        event_city_display = event.get("city", "").title()

        # A. Match categories of registrations — most specific
        if event_category in registered_cats:
            ref_title = registered_cats[event_category]
            return f"Because you registered for '{ref_title}'"

        # B. Match categories of saved events
        if event_category in saved_cats:
            ref_title = saved_cats[event_category]
            return f"Because you saved '{ref_title}'"

        # C. Tag-level interest match (e.g., user interest "AI" matches event tag "ai")
        matching_tags = interest_tags.intersection(set(event_tags))
        if matching_tags:
            tag = list(matching_tags)[0].title()
            return f"Matches your interest in {tag}"

        # D. Category-level interest match
        if event_category in [i.lower() for i in interests]:
            return f"Matches your interest in {event_category.title()}"

        # E. Match categories of viewed events
        if event_category in viewed_cats:
            ref_title = viewed_cats[event_category]
            return f"Similar to '{ref_title}' which you explored"

        # F. Match search queries
        if searches:
            for q in searches:
                if q.lower() in event_title.lower():
                    return f"Related to your search for '{q}'"
            return f"Related to your recent search activity"

        # G. Location match
        if is_location_match:
            return f"Popular {event_category.title()} event in {event_city_display}"

        # H. Recency-based
        recency = calculate_recency_score(event.get("date", ""))
        if recency >= 0.85:
            return f"Upcoming {event_category.title()} event this week"
        elif recency >= 0.45:
            return f"Trending {event_category.title()} event coming soon"

        # I. Default
        return f"Recommended {event_category.title()} event"

    def _get_popularity_fallback(self, limit: int, preferred_cities_lower: list = None):
        """
        Smart fallback for cold users: category-diverse, recency-weighted,
        with random jitter and specific reasons.
        """
        if not preferred_cities_lower:
            preferred_cities_lower = []

        scored = []
        for event_id, event in self.event_cache.items():
            # Skip expired/past events
            if not _is_active_event(event):
                continue

            event_category = event.get("category", "unknown").lower()
            event_city = event.get("city", "").lower()
            event_city_display = event.get("city", "").title()
            event_date_str = event.get("date", "")

            # Recency as primary signal (not flat 0.5)
            recency = calculate_recency_score(event_date_str)

            # Popularity as secondary
            views = event.get("views", 0)
            saves = event.get("saves", 0)
            regs = event.get("registrations", 0)
            pop_raw = views + saves * 3 + regs * 5
            # Normalize loosely (cap at 50)
            pop_norm = min(1.0, pop_raw / 50.0) if pop_raw > 0 else 0.0

            # Location bonus
            location = 1.0 if (event_city and event_city in preferred_cities_lower) else 0.0
            is_location_match = location > 0

            # Score: recency-primary for cold users (no flat similarity baseline)
            score = (
                0.55 * recency +
                0.30 * pop_norm +
                0.15 * location
            )
            # Random jitter to break ties and vary the feed between requests (±0.05)
            score += random.uniform(-0.05, 0.05)
            score = max(0.0, min(1.0, score))
            match_score = int(round(score * 100))

            # Category-specific reasons for cold users
            if is_location_match:
                if recency >= 0.85:
                    reason = f"Upcoming {event_category.title()} in {event_city_display} this week"
                else:
                    reason = f"Popular {event_category.title()} event in {event_city_display}"
            elif recency >= 0.85:
                reason = f"Happening this week: {event_category.title()} event"
            elif recency >= 0.45:
                reason = f"Coming soon: {event_category.title()} event"
            elif pop_norm > 0.3:
                reason = f"Trending {event_category.title()} event"
            else:
                reason = f"Discover {event_category.title()} events"

            scored.append({
                "eventId": event_id,
                "score": round(score, 4),
                "matchScore": match_score,
                "reason": reason,
                "_category": event_category,  # internal, for diversity
            })

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Apply diversity re-ranking
        pool = scored[:DIVERSITY_POOL_SIZE]
        diverse_results = self._diversity_rerank(pool, limit)

        for item in diverse_results:
            self.recommendation_scores_history.append(item["score"])
        return diverse_results

    def invalidate_user_cache(self, user_id: str):
        """Manually invalidates the cached embedding of a specific user."""
        if user_id in self.user_embedding_cache:
            del self.user_embedding_cache[user_id]
            print(f"User cache invalidated for user: {user_id}")
        else:
            print(f"No cache entry found to invalidate for user: {user_id}")

    def get_average_recommendation_score(self) -> float:
        if not self.recommendation_scores_history:
            return 0.0
        return float(np.mean(self.recommendation_scores_history))

    def get_top_categories_stats(self) -> dict:
        category_counts = {}
        for event in self.event_cache.values():
            category = event.get("category")
            if category:
                category = category.lower()
                category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts

    def get_cache_debug_info(self) -> dict:
        """Returns debug information about the event cache."""
        active_count = 0
        expired_count = 0
        categories = defaultdict(int)
        sample_ids = []

        for event_id, event in self.event_cache.items():
            if _is_active_event(event):
                active_count += 1
            else:
                expired_count += 1
            cat = event.get("category", "unknown")
            categories[cat] += 1
            if len(sample_ids) < 20:
                sample_ids.append({
                    "id": event_id,
                    "title": event.get("title", "")[:60],
                    "category": cat,
                    "active": _is_active_event(event),
                })

        cache_bytes = sum(
            (event.get("embedding").nbytes if event.get("embedding") is not None else 0)
            for event in self.event_cache.values()
        )

        return {
            "totalCached": len(self.event_cache),
            "activeEvents": active_count,
            "expiredEvents": expired_count,
            "categoryCounts": dict(categories),
            "sampleEvents": sample_ids,
            "cacheSizeBytes": cache_bytes,
        }


recommender = EventRecommender()
