"""
llm.py — provider-pluggable, per-role LLM client (+ deterministic offline mock).

`get_client(role)` reads config.MODELS and returns a client bound to that role's
provider + model, so different agents can run on different models (e.g. a
distinct reviewer model for independent, cross-model review).

The Gemini SDK is imported LAZILY (only when a GeminiClient is constructed), so
importing this module — and using MockClient — needs no third-party packages and
no API key. That keeps the offline selftest dependency-free.
"""
from __future__ import annotations

import json
import os

import config


class LLMError(Exception):
    """Raised on missing credentials, unwired providers, or unparseable output."""


def parse_json(text: str) -> dict:
    """Best-effort parse of a JSON object from model output (the parse-ladder).

    1) parse as-is; 2) if that fails, extract the first '{' .. last '}' span,
    which transparently handles ```json fences and surrounding prose. Raises
    LLMError if no JSON object can be recovered.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise LLMError("could not parse a JSON object from model output")


class LLMClient:
    """Base client. Subclasses implement `_raw` (and optionally `embed`)."""

    model = "base"

    def _raw(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def complete_json(self, system: str, user: str, temperature: float = 0.0) -> dict:
        """Return parsed JSON. Retries ONCE with an error nudge on a parse failure
        (cheap insurance against an occasional malformed reply; no latency hit on
        the happy path)."""
        try:
            return _require_dict(parse_json(self._raw(system, user, temperature)))
        except LLMError:
            nudge = user + "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object."
            return _require_dict(parse_json(self._raw(system, nudge, temperature)))

    def embed(self, texts, is_query: bool = False):
        raise NotImplementedError("this client does not support embeddings")


def _require_dict(obj):
    if not isinstance(obj, dict):
        raise LLMError("expected a JSON object from the model")
    return obj


# --- Model fallback: self-heal when a model is missing, busy, or times out --- #
# Candidates are tried in order, intersected with what the API actually lists.
# Rolling aliases first (robust to version churn), then concrete current models.
GENERATE_FALLBACKS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-3.5-flash",
                      "gemini-flash-lite-latest", "gemini-2.5-flash-lite",
                      "gemini-3.1-flash-lite", "gemini-pro-latest"]
EMBED_FALLBACKS = ["gemini-embedding-001", "gemini-embedding-latest", "text-embedding-004"]


def _is_recoverable(code, message) -> bool:
    """True for errors worth retrying on a *different* model (missing / busy / slow)."""
    if code in (404, 408, 409, 429, 500, 502, 503, 504):
        return True
    m = (message or "").lower()
    return any(t in m for t in ("not_found", "not found", "unavailable", "overloaded",
                                "deadline", "timeout", "try again", "resource_exhausted"))


def _pick_model(requested, available, fallbacks, exclude=None):
    """Pick a usable model from `available`: keep `requested` if present, else the first
    `fallbacks` entry that's available, else any available — never `exclude`. None if empty."""
    cands = [m for m in available if m != exclude]
    if not cands:
        return None
    for c in [requested, *fallbacks]:
        if c != exclude and c in cands:
            return c
    return cands[0]


class GeminiClient(LLMClient):
    """Google Gemini via the google-genai SDK (imported lazily).

    Self-heals: if a model is not found / overloaded / times out, it lists the
    available models, picks a valid alternative, retries once, and remembers it.
    """

    _catalog_cache = None      # (generate_models, embed_models) — listed once per process
    _resolved: dict = {}       # requested model -> working model (shared across roles)

    def __init__(self, model: str, api_key: "str | None" = None):
        self.model = model
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMError("no Gemini API key (set GEMINI_API_KEY in env or .streamlit/secrets.toml)")
        from google import genai  # lazy import: keeps the module/selftest SDK-free
        self._client = genai.Client(api_key=key)

    def _load_catalog(self):
        """List models once; split into generate-capable and embed-capable short names."""
        if GeminiClient._catalog_cache is not None:
            return GeminiClient._catalog_cache
        gen, emb = [], []
        try:
            for m in self._client.models.list():
                short = (getattr(m, "name", "") or "").split("/")[-1]
                if not short:
                    continue
                actions = (getattr(m, "supported_actions", None)
                           or getattr(m, "supported_generation_methods", None) or [])
                has_gen = any(a in ("generateContent", "generate_content") for a in actions)
                has_emb = any(a in ("embedContent", "embed_content") for a in actions)
                if not actions:                       # attribute absent -> infer from the name
                    has_emb = "embedding" in short
                    has_gen = not has_emb
                if has_gen:
                    gen.append(short)
                if has_emb:
                    emb.append(short)
        except Exception:
            pass
        GeminiClient._catalog_cache = (gen, emb)
        return GeminiClient._catalog_cache

    def _call_with_fallback(self, requested, kind, fn):
        model = GeminiClient._resolved.get(requested, requested)
        try:
            return fn(model)
        except Exception as exc:
            if not _is_recoverable(getattr(exc, "code", None), str(exc)):
                raise
            gen, emb = self._load_catalog()
            available = gen if kind == "gen" else emb
            fallbacks = GENERATE_FALLBACKS if kind == "gen" else EMBED_FALLBACKS
            alt = _pick_model(requested, available, fallbacks, exclude=model)
            if not alt or alt == model:
                raise
            out = fn(alt)                              # retry once on the alternative
            GeminiClient._resolved[requested] = alt    # remember for subsequent calls
            return out

    def _generate(self, model, system, user, temperature):
        from google.genai import types
        resp = self._client.models.generate_content(
            model=model, contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, response_mime_type="application/json",
                temperature=temperature))
        return resp.text or ""

    def _raw(self, system, user, temperature=0.0):
        return self._call_with_fallback(
            self.model, "gen", lambda m: self._generate(m, system, user, temperature))

    def embed(self, texts, is_query=False):
        from google.genai import types
        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"

        def _do(model):
            r = self._client.models.embed_content(
                model=model, contents=list(texts),
                config=types.EmbedContentConfig(task_type=task))
            return [list(e.values) for e in r.embeddings]

        return self._call_with_fallback(self.model, "emb", _do)


# Deterministic, role-shaped canned responses for offline runs/tests.
_MOCK_RESPONSES = {
    "detector": {"has_flaw": True, "flaw_type": "mock_flaw", "severity": "low",
                 "confidence": 0.9, "cited_practice_id": "P01", "rationale": "mock rationale"},
    "test_author": {"cases": [{"args": [[]], "expect": {"kind": "not_raises"}}], "rationale": "mock test"},
    "refactorer": {"code": "def _mock():\n    return None\n", "explanation": "mock refactor"},
    "reviewer": {"approved": True, "comments": "mock approval", "cited_practice_id": "P01"},
    "judge": {"grounded": True, "unsupported_claims": []},
}


def _mock_vector(text: str, dim: int = 16):
    """Deterministic pseudo-embedding from text — stable across runs, no network."""
    v = [0.0] * dim
    for i, ch in enumerate(text):
        v[i % dim] += (ord(ch) % 17) / 17.0
    return v


# --- Oracle replay -------------------------------------------------------- #
# When offline, if a prompt is about a known dataset snippet, the mock returns
# that snippet's *reference* solution. This makes offline runs show real,
# correct per-snippet behavior (a ground-truth ceiling — not a live model), so
# the app demos end-to-end with no API key. Best-effort: degrades to the flat
# canned responses for unknown code (e.g. custom paste) or a missing dataset.
_ORACLE_INDEX = None


def _oracle_index() -> list:
    global _ORACLE_INDEX
    if _ORACLE_INDEX is None:
        try:
            with open("snippets.json", encoding="utf-8") as fh:
                _ORACLE_INDEX = json.load(fh)
        except Exception:
            _ORACLE_INDEX = []
    return _ORACLE_INDEX


def _match_snippet(user: str):
    text = user or ""
    for s in _oracle_index():
        code = (s.get("code") or "").strip()
        if code and code in text:
            return s
    return None


def _oracle_response(role: str, s: dict):
    edge = s.get("edge_case")
    has_flaw = edge != "no_flaw"
    if role == "detector":
        return {"has_flaw": has_flaw, "flaw_type": s.get("flaw_type", "none"),
                "severity": s.get("severity", "low"),
                "confidence": 0.9 if has_flaw else 0.95, "cited_practice_id": None,
                "ambiguous": edge in ("subtle", "multi"), "rationale": s.get("notes", "")}
    if role == "test_author":
        return {"cases": s.get("test_cases", []), "rationale": "Reference test for the labeled flaw."}
    if role == "refactorer":
        return {"code": s.get("reference_fix", s.get("code", "")),
                "explanation": "Reference fix (offline replay)."}
    if role == "reviewer":
        return {"approved": True, "cited_practice_id": None,
                "comments": "Reference fix — correct, minimal, behavior-preserving."}
    if role == "judge":
        return {"grounded": True, "unsupported_claims": []}
    return None


class MockClient(LLMClient):
    """Offline stand-in: role-shaped JSON + deterministic embeddings, no key/network."""

    def __init__(self, role: str = "detector", model: str = "mock"):
        self.role = role
        self.model = model

    def _raw(self, system, user, temperature=0.0):
        snip = _match_snippet(user)
        if snip is not None:
            payload = _oracle_response(self.role, snip)
            if payload is not None:
                return json.dumps(payload)
        return json.dumps(_MOCK_RESPONSES.get(self.role, {}))

    def embed(self, texts, is_query=False):
        return [_mock_vector(t) for t in texts]


def get_client(role: str, overrides: "dict | None" = None, api_key: "str | None" = None) -> LLMClient:
    """Return a client for `role`, routed by config.MODELS (offline -> MockClient)."""
    spec = config.model_for(role, overrides)
    provider, model = spec["provider"], spec["model"]
    if provider == "mock":
        return MockClient(role=role, model=model)
    if provider == "gemini":
        return GeminiClient(model=model, api_key=api_key)
    raise LLMError(
        f"provider {provider!r} is not wired yet (supported: gemini, mock; "
        f"anthropic/openai need an adapter + their key)"
    )
