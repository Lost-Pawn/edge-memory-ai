"""Lightweight, offline PII detection.

Deliberately dependency-free (regex only) so it runs on any edge device with no model
load. Detects the common identifiers that must never leave the device: emails, phone
numbers, credit cards, and government IDs (SSN-shaped).
"""

from __future__ import annotations

import re

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
}


def detect(text: str) -> list[str]:
    """Return the list of PII categories found in ``text``."""
    return [name for name, pat in _PATTERNS.items() if pat.search(text)]


def contains_pii(text: str) -> bool:
    return bool(detect(text))


def redact(text: str) -> str:
    """Replace detected PII with category tags (used for shareable summaries)."""
    out = text
    for name, pat in _PATTERNS.items():
        out = pat.sub(f"[{name.upper()}]", out)
    return out
