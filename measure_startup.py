import time
from recommender import EventRecommender

rec = EventRecommender()
start = time.perf_counter()
rec.initialize()
elapsed = time.perf_counter() - start
print(f"initialize_elapsed_seconds:{elapsed:.3f}")
# Print cached events count
print(f"cached_events:{len(rec.event_cache)}")
# Print vectorizer and embedding types
print(f"vectorizer_loaded:{rec.vectorizer is not None}")
