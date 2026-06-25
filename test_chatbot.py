#!/usr/bin/env python3
"""Offline audit tests for the Kairo Event Agent Chatbot."""

import sys
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from event_search import parse_query_filters, merge_filters, search_events, check_event_is_free
from test_recommender import build_mock_recommender
from chatbot import handle_agent_chat, format_events_context, memory_manager

def _future_date(days: int) -> str:
    return (datetime.now().date() + timedelta(days=days)).strftime("%Y-%m-%d")

def setup_mock_data():
    rec = build_mock_recommender()
    from recommender import recommender
    # Inject the mock vectorizer and cache into the global recommender singleton
    recommender.event_cache = rec.event_cache
    recommender.vectorizer = rec.vectorizer
    recommender.initialized = True

def test_filter_parsing():
    print("Running filter parsing tests...")
    
    # 1. City & Category extraction
    f = parse_query_filters("Find AI workshops in Bangalore")
    assert f.get("city") == "bangalore", f"Expected bangalore, got {f.get('city')}"
    assert f.get("category") == "workshop", f"Expected workshop, got {f.get('category')}"

    f = parse_query_filters("Any startup events in Hyderabad?")
    assert f.get("city") == "hyderabad", f"Expected hyderabad, got {f.get('city')}"
    assert f.get("category") == "startup", f"Expected startup, got {f.get('category')}"

    # 2. Online/Offline & Free/Paid extraction
    f = parse_query_filters("Show online courses")
    assert f.get("isOnline") is True, f"Expected online=True, got {f.get('isOnline')}"
    
    f = parse_query_filters("Find free events near me")
    assert f.get("free") is True, f"Expected free=True, got {f.get('free')}"
    assert f.get("isOnline") is False, f"Expected online=False (near me), got {f.get('isOnline')}"

    # 3. Date range extraction
    f = parse_query_filters("Show me hackathons this weekend")
    assert "date_range" in f, "Expected date_range filter to be present"
    
    print("  PASS  filter parsing")

def test_filter_merging():
    print("Running filter merging tests...")
    # Turn 1: User asks for hackathons in Bangalore
    f1 = parse_query_filters("Find hackathons in Bangalore")
    # Turn 2: User asks "Are any of them free?"
    f2 = parse_query_filters("Are any of them free?")
    
    merged = merge_filters(f1, f2)
    assert merged.get("city") == "bangalore", f"Expected city to remain bangalore, got {merged.get('city')}"
    assert merged.get("category") == "hackathon", f"Expected category to remain hackathon, got {merged.get('category')}"
    assert merged.get("free") is True, f"Expected free to be True, got {merged.get('free')}"
    
    # Turn 3: User says "Forget Bangalore, show Hyderabad"
    f3 = parse_query_filters("Forget Bangalore, show Hyderabad")
    merged2 = merge_filters(merged, f3)
    assert merged2.get("city") == "hyderabad", f"Expected city to update to hyderabad, got {merged2.get('city')}"
    assert merged2.get("category") == "hackathon", "Expected category to remain hackathon"
    
    print("  PASS  filter merging")

def test_search_filtering():
    print("Running cache search filtering tests...")
    setup_mock_data()
    
    # Test category filtering
    results = search_events("", {"category": "meetup"})
    assert len(results) > 0, "Expected to find meetups in mock cache"
    for r in results:
        assert r["category"] == "meetup", f"Expected category meetup, got {r['category']}"
        
    # Test combined category & city filtering
    results = search_events("", {"city": "bangalore", "category": "hackathon"})
    for r in results:
        assert r["city"].lower() == "bangalore"
        assert r["category"] == "hackathon"

    print("  PASS  search filtering")

async def test_chatbot_agent_orchestration():
    print("Running chatbot agent orchestration tests...")
    setup_mock_data()
    
    mock_response = {
        "intent": "find_events",
        "reply": "I found 3 matching hackathons in Bangalore.",
        "suggestions": ["Show online only", "This weekend"]
    }
    
    # Mock the generate_response call on gemini_client
    with patch("gemini_client.gemini_client.generate_response", new_callable=AsyncMock) as mock_gen:
        with patch("gemini_client.gemini_client.is_api_key_available") as mock_key:
            mock_key.return_value = True
            mock_gen.return_value = (mock_response, 120, 45, 0.4)
            
            response_json, matching_events, conv_id = await handle_agent_chat(
                message="Show me hackathons in Bangalore",
                user_id="user_123",
                conversation_id="conv_abc"
            )
            
            assert response_json["intent"] == "find_events"
            assert "hackathons in Bangalore" in response_json["reply"]
            assert len(response_json["suggestions"]) == 2
            assert conv_id == "conv_abc"
            assert len(matching_events) > 0
            
            # Verify memory contains the exchange
            history = memory_manager._sessions[conv_id]["history"]
            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[1]["role"] == "model"
            
    print("  PASS  chatbot agent orchestration")

def main():
    print("Running Kairo Event Agent Chatbot tests...\n")
    failed = 0
    
    # Synchronous tests
    sync_tests = [
        test_filter_parsing,
        test_filter_merging,
        test_search_filtering,
    ]
    for test in sync_tests:
        try:
            test()
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test.__name__}: {e}")
            
    # Asynchronous tests
    try:
        asyncio.run(test_chatbot_agent_orchestration())
    except Exception as e:
        failed += 1
        print(f"  FAIL  test_chatbot_agent_orchestration: {e}")
        
    print()
    if failed:
        print(f"{failed} test(s) failed.")
        sys.exit(1)
    print("All chatbot tests passed successfully.")

if __name__ == "__main__":
    main()
