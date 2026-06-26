import os
import json
import time
import asyncio
import google.generativeai as genai
from typing import Dict, Any, Tuple, List

class GeminiClient:
    def __init__(self):
        self.model_name = "gemini-1.5-flash"
        self._initialized = False

    def is_api_key_available(self) -> bool:
        """Checks if the Gemini API key is configured in the environment."""
        return bool(os.getenv("GEMINI_API_KEY"))

    def _ensure_initialized(self):
        if self._initialized:
            return
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        
        genai.configure(api_key=api_key)
        self._initialized = True

    async def generate_response(
        self, 
        system_instruction: str, 
        history: List[Dict[str, str]], 
        message: str
    ) -> Tuple[Dict[str, Any], int, int, float]:
        """
        Invokes Gemini Flash using the given system instructions, message history, and user message.
        Returns:
            Tuple of (response_json_dict, prompt_tokens, response_tokens, gemini_latency_seconds)
        """
        self._ensure_initialized()
        
        # Build contents structure conforming to the SDK requirements
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [msg["content"]]})
        
        contents.append({"role": "user", "parts": [message]})

        # Instantiate model with specific system instructions
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={"response_mime_type": "application/json"},
            system_instruction=system_instruction
        )

        start_time = time.perf_counter()
        
        # Call API synchronously in a thread to prevent blocking the async loop
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, contents), 
                timeout=15.0
            )
        except asyncio.TimeoutError:
            print("Gemini API call timed out after 15 seconds.")
            return {
                "intent": "general_help",
                "reply": "I'm sorry, I'm taking too long to think. Let's try asking something else.",
                "suggestions": ["Show all events", "Clear filters"]
            }, 0, 0, time.perf_counter() - start_time
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return {
                "intent": "general_help",
                "reply": "I'm having trouble connecting right now. Please try again later.",
                "suggestions": ["Show all events"]
            }, 0, 0, time.perf_counter() - start_time
        
        latency = time.perf_counter() - start_time

        # Extract token counts
        prompt_tokens = 0
        response_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", 0)
            response_tokens = getattr(usage, "candidates_token_count", 0)

        # Parse structured JSON response
        try:
            response_text = response.text
            response_json = json.loads(response_text)
        except Exception as e:
            print(f"Error parsing Gemini JSON response: {e}. Raw: {response.text}")
            response_json = {
                "intent": "general_help",
                "reply": response.text or "I encountered an error parsing the response.",
                "suggestions": ["Show all events", "AI events in Bangalore"]
            }

        return response_json, prompt_tokens, response_tokens, latency

    async def generate_response_stream(
        self,
        system_instruction: str,
        history: List[Dict[str, str]],
        message: str
    ):
        """
        Placeholder/Stub for streaming responses in the future.
        Yields chunks of text or status updates.
        """
        self._ensure_initialized()
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [msg["content"]]})
        contents.append({"role": "user", "parts": [message]})

        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={"response_mime_type": "application/json"},
            system_instruction=system_instruction
        )
        
        response_stream = model.generate_content_stream(contents)
        for chunk in response_stream:
            yield chunk.text

gemini_client = GeminiClient()
