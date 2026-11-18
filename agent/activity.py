"""Activity-level helpers shared by the nutrition agent and the chat API.

Everything here is deterministic code: no model output decides a user's activity level.
"""
from __future__ import annotations

import re

ACTIVITY_LEVELS = ("sedentary", "lightly_active", "moderately_active", "very_active", "athlete")

ACTIVITY_LABELS = {
    "sedentary": "Sedentary (little or no exercise)",
    "lightly_active": "Lightly active (light exercise 1-3 days a week)",
    "moderately_active": "Moderately active (exercise 3-5 days a week)",
    "very_active": "Very active (hard exercise 6-7 days a week)",
    "athlete": "Athlete (intense daily training or a physical job)",
}

ACTIVITY_QUESTION_NO_PLAN = (
    "You have no ongoing workout plan, so I can't tell how active you are. "
    "What's your activity level? You can type it or pick one: sedentary, lightly active, "
    "moderately active, very active, or athlete."
)
ACTIVITY_QUESTION_RETRY = (
    "I couldn't match that to an activity level. Please type or pick one of: sedentary, "
    "lightly active, moderately active, very active, or athlete."
)


def activity_options() -> list[dict[str, str]]:
    """Selectable options for a client, as stable values plus display labels."""
    return [{"value": level, "label": ACTIVITY_LABELS[level]} for level in ACTIVITY_LEVELS]


_PATTERNS = {
    "sedentary": re.compile(r"\b(sedentary|inactive|desk job)\b"),
    "lightly_active": re.compile(r"\blight(?:ly)?(?:\s+active)?\b"),
    "moderately_active": re.compile(r"\bmoderate(?:ly)?(?:\s+active)?\b"),
    "very_active": re.compile(r"\b(very\s+active|extra\s+active)\b"),
    "athlete": re.compile(r"\b(athlete|extremely\s+active)\b"),
}
_NEGATION = re.compile(r"\bnot\b|n't\b|\bnever\b")


def parse_activity_level(text: str | None) -> str | None:
    """Map a typed or selected answer to one allowed level, or None when it is unclear.

    Accepts the stored values ("lightly_active") and plain phrases ("lightly active").
    Negated or ambiguous answers ("not very active", "light and heavy") return None
    so the caller asks again instead of guessing.
    """
    if not isinstance(text, str):
        return None
    normalized = re.sub(r"[_\-]+", " ", text.strip().lower())
    if not normalized or len(normalized) > 60 or _NEGATION.search(normalized):
        return None
    matches = [level for level, pattern in _PATTERNS.items() if pattern.search(normalized)]
    return matches[0] if len(matches) == 1 else None


def derive_activity_level(sessions_per_week: int | None) -> str | None:
    """Suggest an activity level from the saved workout plan's weekly frequency.

    Only the middle three levels are derived. "sedentary" and "athlete" depend on things a
    plan cannot show (daily movement, a physical job, competition), so only the user sets them.
    """
    if not isinstance(sessions_per_week, int) or isinstance(sessions_per_week, bool):
        return None
    if 1 <= sessions_per_week <= 2:
        return "lightly_active"
    if 3 <= sessions_per_week <= 4:
        return "moderately_active"
    if 5 <= sessions_per_week <= 7:
        return "very_active"
    return None
