#!/usr/bin/env python3
"""Offline audit tests for the Kairo recommendation engine."""

from datetime import datetime, timedelta
import sys

import numpy as np

from recommender import (
    EventRecommender,
    _is_active_event,
    calculate_recency_score,
    WEIGHT_VIEW,
    DWELL_DEEP,
    DWELL_NORMAL,
    DWELL_BOUNCE,
    WEIGHT_TAG_BOOST,
)


def _future_date(days: int) -> str:
    return (datetime.now().date() + timedelta(days=days)).strftime("%Y-%m-%d")


def _past_date(days: int) -> str:
    return (datetime.now().date() - timedelta(days=days)).strftime("%Y-%m-%d")


def _make_embedding(seed: int, dim: int = 384) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(dim) / (np.linalg.norm(rng.randn(dim)) + 1e-9)


def _make_event(event_id, category, date, *, status="active", tags=None, seed=0, city="Bangalore"):
    return {
        "embedding": _make_embedding(seed),
        "title": f"{category.title()} Event {event_id}",
        "category": category,
        "tags": tags or [],
        "city": city,
        "source": "test",
        "date": date,
        "views": 0,
        "saves": 0,
        "registrations": 0,
        "status": status,
        "organizer": "Test Org",
    }


def build_mock_recommender() -> EventRecommender:
    rec = EventRecommender()
    rec.initialized = True
    cache = {}
    seed = 1
    for i, cat in enumerate(["hackathon", "meetup", "workshop", "startup", "conference"]):
        for j in range(8):
            eid = f"{cat}-{j}"
            cache[eid] = _make_event(eid, cat, _future_date(1 + (i * 3 + j) % 45), seed=seed, tags=["ai"] if cat == "meetup" else [cat])
            seed += 1
    cache["expired-status"] = _make_event("expired-status", "meetup", _future_date(10), status="expired", seed=900)
    cache["past-date"] = _make_event("past-date", "hackathon", _past_date(5), seed=901)
    rec.event_cache = cache
    return rec


def test_active_event_filter():
    assert _is_active_event(_make_event("a", "meetup", _future_date(3), seed=10))
    assert not _is_active_event(_make_event("b", "meetup", _future_date(3), status="expired", seed=11))
    assert not _is_active_event(_make_event("c", "meetup", _past_date(2), seed=12))
    print("  PASS  active event filter")


def test_cold_user_diversity_and_scores():
    rec = build_mock_recommender()
    results = rec._get_popularity_fallback(limit=20)
    scores = {r["score"] for r in results}
    categories = {rec.event_cache[r["eventId"]]["category"] for r in results}
    ids = {r["eventId"] for r in results}

    assert len(results) == 20
    assert len(scores) >= 8, f"Expected >=8 distinct scores, got {len(scores)}"
    assert len(categories) >= 3, f"Expected >=3 categories, got {categories}"
    assert "expired-status" not in ids and "past-date" not in ids
    print(f"  PASS  cold user fallback ({len(scores)} scores, {len(categories)} categories)")


def test_diversity_rerank():
    rec = build_mock_recommender()
    scored = []
    hack_vec = rec.event_cache["hackathon-0"]["embedding"]
    for eid, event in rec.event_cache.items():
        if not _is_active_event(event):
            continue
        score = max(0.0, float(np.dot(hack_vec, event["embedding"])))
        if event["category"] == "hackathon":
            score += 0.3
        scored.append({"eventId": eid, "score": score, "matchScore": int(score * 100), "reason": "test", "_category": event["category"]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    diverse = rec._diversity_rerank(scored[:60], 20)
    cats = {rec.event_cache[r["eventId"]]["category"] for r in diverse}
    assert len(cats) >= 3
    print(f"  PASS  diversity rerank ({len(cats)} categories)")


def test_reason_specificity():
    rec = build_mock_recommender()
    reason = rec._generate_reason(
        rec.event_cache["meetup-0"], "bangalore", False,
        {}, {"meetup": "AI Engineering 2026 Meetup"}, {}, {"ai"}, [], ["AI"],
    )
    assert "AI Engineering 2026 Meetup" in reason
    print(f"  PASS  event-specific reason")


def test_similar_events_skips_expired():
    rec = build_mock_recommender()
    ids = {s["eventId"] for s in rec.get_similar_events("meetup-0", limit=50)}
    assert "expired-status" not in ids and "past-date" not in ids
    print("  PASS  similar events filter")


def test_cache_debug_info():
    rec = build_mock_recommender()
    info = rec.get_cache_debug_info()
    assert info["expiredEvents"] >= 2 and info["cacheSizeBytes"] > 0
    print(f"  PASS  cache debug ({info['activeEvents']} active, {info['expiredEvents']} expired)")


def main():
    tests = [
        test_active_event_filter,
        test_cold_user_diversity_and_scores,
        test_diversity_rerank,
        test_reason_specificity,
        test_similar_events_skips_expired,
        test_cache_debug_info,
    ]
    print("Running recommendation engine audit tests...\n")
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print()
    if failed:
        print(f"{failed} test(s) failed.")
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
