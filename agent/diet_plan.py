"""Diet plan rules that need no model: what the user must provide first, and when a saved plan is out of date.

The nutrition agent writes the plan text. Everything here is deterministic code.
"""
from __future__ import annotations

import hashlib
import json

from agent.activity import ACTIVITY_LABELS, ACTIVITY_LEVELS, derive_activity_level

# A fixed request, never free user text. It must match the nutrition agent's diet-plan pattern.
DIET_REQUEST_MESSAGE = (
    "Create a personalized diet plan for me. Use my saved profile and, when I have one, "
    "my workout plan. Structure it around my meals per day."
)

DIETARY_LABELS = {
    "omnivore": "Eats everything",
    "non_vegetarian": "Non-vegetarian",
    "vegetarian": "Vegetarian",
    "eggetarian": "Vegetarian plus eggs",
    "vegan": "Vegan",
    "pescatarian": "Fish but no meat",
    "jain": "Jain",
}
SEX_LABELS = {"male": "Male", "female": "Female"}

# What a plan needs. The nutrition agent may not invent targets, so these must be real values.
_NUMBER_FIELDS = {
    "age": "Your age in years",
    "height_cm": "Your height in cm",
    "weight_kg": "Your weight in kg",
}


def _choice(options: dict[str, str]) -> list[dict[str, str]]:
    return [{"value": value, "label": label} for value, label in options.items()]


def _question(field: str) -> dict:
    if field == "activity_level":
        return {"field": "activity_level", "type": "choice",
                "prompt": "What's your activity level?",
                "options": _choice({level: ACTIVITY_LABELS[level] for level in ACTIVITY_LEVELS})}
    if field == "dietary_pattern":
        return {"field": "dietary_pattern", "type": "choice",
                "prompt": "What best describes how you eat?", "options": _choice(DIETARY_LABELS)}
    return {"field": field, "type": "number", "prompt": _NUMBER_FIELDS[field], "options": []}


def effective_activity(profile, training_context) -> tuple[str | None, str | None]:
    """(level, source). A level the user set wins; otherwise it comes from the saved workout plan."""
    stated = getattr(profile, "activity_level", "unspecified")
    if stated and stated != "unspecified":
        return stated, "profile"
    derived = derive_activity_level(getattr(training_context, "sessions_per_week", None))
    if derived:
        return derived, "workout_plan"
    return None, None


def readiness(profile, training_context) -> dict:
    """Say whether a diet plan can be made now, and which questions are still open."""
    level, source = effective_activity(profile, training_context)
    missing: list[str] = []
    if profile is None:
        missing = ["activity_level", "dietary_pattern", *_NUMBER_FIELDS]
    else:
        if level is None:
            missing.append("activity_level")
        if profile.dietary_pattern == "unspecified":
            missing.append("dietary_pattern")
        for field in _NUMBER_FIELDS:
            if getattr(profile, field, None) is None:
                missing.append(field)
    return {
        "ready": not missing,
        "missing": missing,
        "questions": [_question(field) for field in missing],
        "has_workout_plan": training_context is not None,
        "activity_level": level,
        "activity_source": source,
    }


def fingerprint(profile, training_context) -> str:
    """Hash of the inputs a plan depends on, so a saved plan can be marked out of date.

    Only plan-level workout facts are included. Swapping one exercise does not change the diet.
    """
    level, source = effective_activity(profile, training_context)
    data = {
        "activity": level,
        "activity_source": source,
        "profile": {
            key: getattr(profile, key, None)
            for key in ("goal", "age", "sex", "height_cm", "weight_kg", "dietary_pattern",
                        "preferred_foods", "excluded_foods", "allergies", "intolerances",
                        "meals_per_day", "daily_food_budget_inr")
        },
        "training": None if training_context is None else {
            "goal": training_context.program_goal,
            "level": training_context.program_level,
            "split": training_context.split,
            "sessions_per_week": training_context.sessions_per_week,
        },
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
