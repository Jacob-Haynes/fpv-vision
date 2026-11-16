"""Optional cloud scene description via Google Gemini (``google-genai`` SDK).

Off unless ``GEMINI_API_KEY`` is set. Every call is rate-limited (``GEMINI_MIN_INTERVAL``).
NOTE: cloud mode uploads the snapshot JPEG to Google — use demo footage only.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

PROMPT = (
    "You are assisting a military ISR training exercise. Describe this single video frame "
    "from a drone or handheld camera as a short observation report: what people, vehicles "
    "and equipment are visible, how many, what they appear to be doing, and where they are "
    "in the frame relative to terrain features (e.g. treeline, road, building). "
    "One or two sentences, under 45 words, plain text, no speculation about identity. "
    "If nothing of interest is visible, say so."
)


class RateLimited(Exception):
    def __init__(self, retry_in: float):
        super().__init__(f"rate limited, retry in {retry_in:.0f}s")
        self.retry_in = retry_in


class GeminiDescriber:
    def __init__(self, api_key: str, model: str, min_interval: float = 10.0, timeout_s: float = 25.0):
        self.api_key = api_key
        self.model = model
        self.min_interval = max(1.0, min_interval)
        self.timeout_s = timeout_s
        self._last_call = 0.0
        self._lock = threading.Lock()
        self._client = None
        self.calls = 0
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(api_key=self.api_key,
                                        http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))
        return self._client

    def retry_in(self) -> float:
        return max(0.0, self._last_call + self.min_interval - time.time())

    def describe(self, jpeg: bytes, context: str = "") -> str:
        """Blocking call; raises RateLimited if called too soon, RuntimeError on API failure."""
        if not self.enabled:
            raise RuntimeError("Gemini not configured (no GEMINI_API_KEY)")
        with self._lock:
            wait = self.retry_in()
            if wait > 0:
                raise RateLimited(wait)
            self._last_call = time.time()
        from google.genai import types

        prompt = PROMPT + (f"\nLocal detector context (may be incomplete): {context}" if context else "")
        try:
            resp = self._get_client().models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
            )
            self.calls += 1
            self.last_error = ""
            text = (resp.text or "").strip()
            return text or "(no description returned)"
        except Exception as e:  # noqa: BLE001 - network/API errors fall back to local summary
            self.last_error = str(e)[:200]
            log.warning("Gemini call failed: %s", e)
            raise RuntimeError(self.last_error) from e

    def status(self) -> dict:
        return {"enabled": self.enabled, "model": self.model if self.enabled else None,
                "min_interval": self.min_interval, "retry_in": round(self.retry_in(), 1),
                "calls": self.calls, "error": self.last_error}
