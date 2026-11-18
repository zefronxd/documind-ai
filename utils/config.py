"""Central configuration. Values come from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

APP_NAME = "DocuMind AI"
APP_TAGLINE = "Intelligent Document Analysis & Knowledge Extraction System"
DATA_DIR = BASE_DIR / "data"


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


MAX_UPLOAD_MB = _int_env("MAX_UPLOAD_MB", 25)
MAX_PAGES_PER_DOC = _int_env("MAX_PAGES_PER_DOC", 500)
SCANNED_MIN_CHARS_PER_PAGE = _int_env("SCANNED_MIN_CHARS_PER_PAGE", 50)

OCR_DPI = _int_env("OCR_DPI", 200)
OCR_MAX_PAGES = _int_env("OCR_MAX_PAGES", 50)

# Model is set directly in code (no .env needed). gemini-2.5-flash is retired by Google.
GEMINI_MODEL = "gemini-3.8-flash"
# Tried in order if the main model returns "not found".
GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]
# Set in code (ignores .env). Newer Gemini models "think" first, so they need more time.
LLM_TIMEOUT_SECONDS = 180
MAX_LLM_CHARS = _int_env("MAX_LLM_CHARS", 120000)

CHUNK_SIZE = _int_env("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int_env("CHUNK_OVERLAP", 200)
TOP_K = _int_env("TOP_K", 5)
MIN_RELEVANCE = _float_env("MIN_RELEVANCE", 0.20)


def get_gemini_api_key() -> str | None:
    """Read the key at call time. Never log or display the returned value."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


def has_api_key() -> bool:
    return get_gemini_api_key() is not None
