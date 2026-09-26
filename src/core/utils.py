"""
Lab 11 — Helper Utilities
"""
from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner

import asyncio


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    Includes retry logic for transient Gemini 503 errors.
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    max_retries = 5
    base_delay = 10  # seconds

    for attempt in range(max_retries + 1):
        try:
            user_id = "student"
            app_name = runner.app_name

            session = None
            if session_id is not None:
                try:
                    session = await runner.session_service.get_session(
                        app_name=app_name, user_id=user_id, session_id=session_id
                    )
                except (ValueError, KeyError):
                    pass

            if session is None:
                try:
                    session = await runner.session_service.create_session(
                        app_name=app_name, user_id=user_id
                    )
                except Exception:
                    session = await runner.session_service.create_session(
                        app_name=app_name, user_id=user_id
                    )

            content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)],
            )

            final_response = ""
            async for event in runner.run_async(
                user_id=user_id, session_id=session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text

            return final_response, session

        except Exception as e:
            err_str = str(e)
            is_retryable = (
                "503" in err_str
                or "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "UNAVAILABLE" in err_str
                or "overloaded" in err_str.lower()
                or "quota" in err_str.lower()
                or "rate" in err_str.lower()
            )
            if is_retryable and attempt < max_retries:
                import re
                match = re.search(r"retry in ([0-9\.]+)s", err_str)
                if match:
                    delay = float(match.group(1)) + 2.0
                else:
                    match_delay = re.search(r"retryDelay': '(\d+)s'", err_str)
                    if match_delay:
                        delay = float(match_delay.group(1)) + 2.0
                    else:
                        delay = min(base_delay * (2 ** attempt), 60)
                print(f"[Retry {attempt+1}/{max_retries}] Gemini rate/quota (429/503) — waiting {delay:.1f}s...")
                await asyncio.sleep(delay)
                continue
            raise

