import os
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load env variables
load_dotenv()

class EventRecommender:
    def __init__(self):
        self.model = None
        self.db = None
        self.event_cache = {}  # event_id -> dict with embedding, title, city, category, tags
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
        Fetches all events from Firestore. If an event lacks an embedding in the DB,
        generates the embedding and writes it back to Firestore to cache it.
        """
        if not self.db or not self.model:
            print("Cannot sync: Recommender is in offline/mock mode.")
            return

        try:
            print("Syncing event embeddings with Firestore...")
            events_ref = self.db.collection("events")
            docs = events_ref.stream()
            
            events_to_embed = []
            events_metadata = []

            for doc in docs:
                event_id = doc.id
                data = doc.to_dict()
                
                title = data.get("title", "")
                description = data.get("description", "")
                category = data.get("category", "")
                tags = data.get("tags", [])
                city = data.get("city", "")
                source = data.get("source", "")
                
                # Check if embedding already exists in DB
                db_embedding = data.get("embedding")
                
                if db_embedding and isinstance(db_embedding, list) and len(db_embedding) == 384:
                    # Load directly from cache
                    self.event_cache[event_id] = {
                        "embedding": np.array(db_embedding),
                        "title": title,
                        "category": category,
                        "tags": tags,
                        "city": city,
                        "source": source,
                    }
                else:
                    # Queue for embedding generation
                    tags_str = " ".join(tags)
                    text_to_embed = f"{title} {description} {category} {tags_str}"
                    events_to_embed.append(text_to_embed)
                    events_metadata.append((event_id, title, category, tags, city, source))

            if events_to_embed:
                print(f"Generating embeddings for {len(events_to_embed)} new events...")
                embeddings = self.model.encode(events_to_embed)
                
                # Write back to Firestore and update cache
                for i, (event_id, title, category, tags, city, source) in enumerate(events_metadata):
                    vector = embeddings[i]
                    vector_list = vector.tolist()
                    
                    # Update cache
                    self.event_cache[event_id] = {
                        "embedding": vector,
                        "title": title,
                        "category": category,
                        "tags": tags,
                        "city": city,
                        "source": source,
                    }
                    
                    # Write to DB (non-blocking update)
                    try:
                        events_ref.document(event_id).update({
                            "embedding": vector_list
                        })
                    except Exception as update_err:
                        print(f"Failed to update embedding for event {event_id} in Firestore: {update_err}")
                
                print(f"Generated and cached {len(events_to_embed)} event embeddings.")
            else:
                print("All events are already embedded.")
        except Exception as e:
            print(f"Error during embedding sync: {e}")

    def get_similar_events(self, event_id: str, limit: int = 5):
        """
        Calculate cosine similarity between the target event and all other events.
        """
        self.initialize()
        
        # If cache is empty, try to sync
        if not self.event_cache:
            self.sync_embeddings()

        if event_id not in self.event_cache:
            print(f"Event ID {event_id} not found in cache.")
            return []

        target_event = self.event_cache[event_id]
        target_vector = target_event["embedding"].reshape(1, -1)
        
        similarities = []
        for other_id, other_event in self.event_cache.items():
            if other_id == event_id:
                continue
            
            other_vector = other_event["embedding"].reshape(1, -1)
            score = float(cosine_similarity(target_vector, other_vector)[0][0])
            
            # Normalize negative similarities to 0
            score = max(0.0, score)

            similarities.append({
                "eventId": other_id,
                "score": round(score, 2)
            })

        # Sort by similarity score descending
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:limit]

    def get_recommendations(self, user_id: str, limit: int = 20):
        """
        Generates weighted user profile vector using user interests, search history, preferred cities,
        and weighted activity telemetry:
        - Registrations: 10
        - Saves: 7
        - Views: 3 * min(4, 1 + ln(dwellTime))
        - Searches: 2
        - Explicit Interests: 5
        """
        self.initialize()

        if not self.db or not self.model:
            # Fallback mock recommendations if Firestore is not connected
            print("DB not initialized. Returning fallback/trending recommendations.")
            return [{"eventId": eid, "score": 0.85} for eid in list(self.event_cache.keys())[:limit]]

        # If cache is empty, try to sync
        if not self.event_cache:
            self.sync_embeddings()

        if not self.event_cache:
            return []

        # 1. Fetch User document
        user_doc_ref = self.db.collection("users").document(user_id)
        user_snap = user_doc_ref.get()
        if not user_snap.exists:
            print(f"User {user_id} not found. Returning top popularity events.")
            return self._get_popularity_fallback(limit)

        user_data = user_snap.to_dict()
        interests = user_data.get("interests", [])
        preferred_cities = user_data.get("preferredCities", [])
        preferred_cities_lower = [c.lower() for c in preferred_cities]

        # 2. Fetch User Analytics Activity logs
        activity_logs = []
        try:
            activities_ref = self.db.collection("analytics_events")
            query_snap = activities_ref.where("userId", "==", user_id).stream()
            for doc in query_snap:
                activity_logs.append(doc.to_dict())
        except Exception as query_err:
            print(f"Error fetching analytics events for user {user_id}: {query_err}")

        # Also fetch user saved bookmarks
        saved_event_ids = []
        try:
            bookmarks_ref = self.db.collection("users").document(user_id).collection("bookmarks").stream()
            for b in bookmarks_ref:
                saved_event_ids.append(b.id)
        except Exception as b_err:
            print(f"Error fetching user bookmarks: {b_err}")

        # Build vectors & weights arrays to calculate user centroid
        vectors = []
        weights = []

        # A. Process Explicit Interests (Weight = 5 each)
        if interests:
            interest_vectors = self.model.encode(interests)
            for i, vec in enumerate(interest_vectors):
                vectors.append(vec)
                weights.append(5.0)

        # B. Process Saved Events (from subcollection or logs, Weight = 7)
        for doc_id in saved_event_ids:
            if doc_id in self.event_cache:
                vectors.append(self.event_cache[doc_id]["embedding"])
                weights.append(7.0)

        # C. Process Activity Log events (View = 3 with dwell scaling, Register = 10, Search = 2)
        for act in activity_logs:
            action = act.get("action")
            event_id = act.get("eventId")
            
            if action == "register" and event_id in self.event_cache:
                vectors.append(self.event_cache[event_id]["embedding"])
                weights.append(10.0)
                
            elif action == "save" and event_id in self.event_cache:
                vectors.append(self.event_cache[event_id]["embedding"])
                weights.append(7.0)
                
            elif action == "view" and event_id in self.event_cache:
                dwell = act.get("dwellTime") or 0
                multiplier = 1.0
                if dwell > 0:
                    multiplier = min(4.0, 1.0 + math.log(dwell))
                vectors.append(self.event_cache[event_id]["embedding"])
                weights.append(3.0 * multiplier)
                
            elif action == "search" and act.get("query"):
                query_str = act.get("query")
                query_vec = self.model.encode([query_str])[0]
                vectors.append(query_vec)
                weights.append(2.0)

        # 3. Generate Combined User Profile Vector
        if not vectors:
            # Fallback to general popularity rankings if user has no profiles or activities
            print(f"User {user_id} has no profile features. Returning popularity fallback.")
            return self._get_popularity_fallback(limit, preferred_cities_lower)

        user_profile_vector = np.average(vectors, axis=0, weights=weights).reshape(1, -1)

        # 4. Score all cached events
        recommendations = []
        for other_id, other_event in self.event_cache.items():
            # Skip events user has already registered for to keep the feed fresh
            has_registered = any(act.get("eventId") == other_id and act.get("action") == "register" for act in activity_logs)
            if has_registered:
                continue

            other_vector = other_event["embedding"].reshape(1, -1)
            score = float(cosine_similarity(user_profile_vector, other_vector)[0][0])
            score = max(0.0, score)

            # Location intelligence boost: if event matches user preferred cities, boost the score
            event_city = other_event["city"].lower()
            if event_city in preferred_cities_lower:
                score += 0.15  # 15% flat score boost

            recommendations.append({
                "eventId": other_id,
                "score": round(min(1.0, score), 2)
            })

        # Sort recommendations by score descending
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:limit]

    def _get_popularity_fallback(self, limit: int, preferred_cities_lower: list = None):
        """
        Fallback scoring based on event popularityScore when no user activity profiles exist.
        """
        if not preferred_cities_lower:
            preferred_cities_lower = []
            
        scored = []
        for event_id, event in self.event_cache.items():
            # Read popularityScore or fallback to tag sizes
            score = 0.5  # Base score
            event_city = event["city"].lower()
            if event_city in preferred_cities_lower:
                score += 0.15
            scored.append({
                "eventId": event_id,
                "score": round(score, 2)
            })
        return scored[:limit]

recommender = EventRecommender()
