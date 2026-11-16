"""Scene description by a vision-language model, with an offline-first fallback chain.

VLM_BACKEND:
  openai_compat  an OpenAI-compatible endpoint, e.g. Ollama on the HQ Mac
                 (VLM_URL = bare base URL such as http://isr-hq.local:11434; we append
                 /v1/chat/completions). Works with no internet.
  gemini         Google Gemini (needs GEMINI_API_KEY and internet; images go to Google).
  off            local detector summary only.
  auto (default) openai_compat if VLM_URL is set, else gemini if a key is set, else off.

If the chosen backend is unreachable we fall back: openai_compat -> gemini (if a key is
set) -> off (local summary). The UI shows which one actually answered.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request

from fpv_vision.gemini import PROMPT, GeminiDescriber, RateLimited

log = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = "qwen3-vl:8b"


class OpenAICompatDescriber:
    """Minimal OpenAI-style /v1/chat/completions client (stdlib only)."""

    name = "openai_compat"

    def __init__(self, url: str, model: str, api_key: str = "", min_interval: float = 3.0, timeout_s: float = 45.0):
        self.base = url.rstrip("/")
        for suffix in ("/v1/chat/completions", "/v1", "/api/chat", "/api"):
            if self.base.endswith(suffix):  # tolerate a URL with a path despite the contract
                self.base = self.base[: -len(suffix)]
        self.model = model or DEFAULT_VLM_MODEL
        self.api_key = api_key
        self.min_interval = max(0.5, min_interval)
        self.timeout_s = timeout_s
        self._last_call = 0.0
        self._lock = threading.Lock()
        self._reach_cache: tuple[float, bool] = (0.0, False)
        self._probing = False
        self.calls = 0
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return bool(self.base)

    def retry_in(self) -> float:
        return max(0.0, self._last_call + self.min_interval - time.time())

    def reachable(self, max_age: float = 15.0) -> bool:
        """Non-blocking: last known reachability; refreshes in the background when stale."""
        ts, ok = self._reach_cache
        if time.time() - ts >= max_age and not self._probing:
            self._probing = True
            threading.Thread(target=self.probe, daemon=True).start()
        return ok

    def probe(self) -> bool:
        """Blocking reachability check (GET /v1/models, 2.5 s timeout)."""
        ok = False
        try:
            with urllib.request.urlopen(f"{self.base}/v1/models", timeout=2.5) as r:
                ok = r.status == 200
        except (OSError, urllib.error.URLError) as e:
            self.last_error = f"unreachable: {e}"
        self._reach_cache = (time.time(), ok)
        self._probing = False
        return ok

    def describe(self, jpeg: bytes, context: str = "") -> str:
        with self._lock:
            wait = self.retry_in()
            if wait > 0:
                raise RateLimited(wait)
            self._last_call = time.time()
        prompt = PROMPT + (f"\nLocal detector context (may be incomplete): {context}" if context else "")
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
            ]}],
            "max_tokens": 600,
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                data = json.loads(r.read().decode())
            text = data["choices"][0]["message"].get("content") or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            self.calls += 1
            self.last_error = ""
            self._reach_cache = (time.time(), True)
            return text or "(no description returned)"
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)[:200]
            self._reach_cache = (time.time(), False)
            log.warning("VLM call to %s failed: %s", self.base, e)
            raise RuntimeError(self.last_error) from e

    def status(self) -> dict:
        return {"enabled": self.enabled, "url": self.base, "model": self.model, "calls": self.calls,
                "error": self.last_error, "retry_in": round(self.retry_in(), 1)}


class SceneDescriber:
    """Fallback chain over the configured backends; 'off' always means local summary."""

    def __init__(self, backend: str, openai: OpenAICompatDescriber | None, gemini: GeminiDescriber):
        backend = (backend or "auto").strip().lower()
        if backend == "auto":
            backend = "openai_compat" if openai and openai.enabled else ("gemini" if gemini.enabled else "off")
        if backend not in ("openai_compat", "gemini", "off"):
            log.warning("unknown VLM_BACKEND %r, using off", backend)
            backend = "off"
        self.selected = backend
        self.openai = openai
        self.gemini = gemini
        chain: list = []
        if backend == "openai_compat" and openai and openai.enabled:
            chain.append(openai)
        if backend in ("openai_compat", "gemini") and gemini.enabled:
            chain.append(gemini)
        self.chain = chain
        self.last_used = "off"

    @property
    def enabled(self) -> bool:
        return bool(self.chain)

    @property
    def min_interval(self) -> float:
        return min((b.min_interval for b in self.chain), default=0.0)

    def retry_in(self) -> float:
        return min((b.retry_in() for b in self.chain), default=0.0)

    def active(self) -> str:
        """Backend that would answer right now (cheap reachability check for local VLM)."""
        for b in self.chain:
            if b is self.openai:
                if b.reachable():
                    return "openai_compat"
                continue
            return "gemini"
        return "off"

    def describe(self, jpeg: bytes, context: str = "") -> tuple[str, str]:
        """Returns (text, backend_name). Raises RateLimited / RuntimeError if every backend failed."""
        errors: list[str] = []
        limited: RateLimited | None = None
        for b in self.chain:
            name = "openai_compat" if b is self.openai else "gemini"
            if b is self.openai and not (b.reachable() or b.probe()):
                errors.append(f"{name}: unreachable")
                continue
            try:
                text = b.describe(jpeg, context)
                self.last_used = name
                return text, name
            except RateLimited as e:
                limited = limited or e
            except RuntimeError as e:
                errors.append(f"{name}: {e}")
        self.last_used = "off"
        if limited and not errors:
            raise limited
        raise RuntimeError("; ".join(errors) or "no VLM backend configured")

    def mode_label(self) -> str:
        a = self.active()
        if a == "openai_compat":
            return "vlm"
        return a if a == "gemini" else "local"

    def status(self) -> dict:
        active = self.active()
        return {
            "selected": self.selected,
            "active": active,
            "chain": ["openai_compat" if b is self.openai else "gemini" for b in self.chain] + ["off"],
            "enabled": self.enabled,
            "model": (self.openai.model if active == "openai_compat" and self.openai else
                      self.gemini.model if active == "gemini" else None),
            "min_interval": self.min_interval,
            "retry_in": round(self.retry_in(), 1),
            "last_used": self.last_used,
            "openai_compat": self.openai.status() if self.openai else None,
            "gemini": self.gemini.status(),
        }
