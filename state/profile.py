"""User profile model: who the user is, what they can train with, and health limits."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

GYM_EQUIPMENT = {"barbell", "squat_rack", "cable", "machine"}
KNOWN_EQUIPMENT = {"bodyweight", "dumbbells", "bench", "pull_up_bar", "row_bar", "cardio_machine"} | GYM_EQUIPMENT | {"full_gym"}
KNOWN_LIMITATIONS = {"knee", "lower_back", "shoulder", "wrist", "elbow"}
Screening = Literal["not_asked", "clear", "cleared_by_doctor", "needs_clearance"]


class UserProfile(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    age: int | None = Field(default=None, ge=10, le=100)
    sex: Literal["unspecified", "male", "female"] = "unspecified"
    height_cm: float | None = Field(default=None, ge=100, le=250)
    weight_kg: float | None = Field(default=None, ge=30, le=300)
    # Optional circumference inputs for the US Navy body-fat estimate.
    waist_cm: float | None = Field(default=None, ge=40, le=200)
    neck_cm: float | None = Field(default=None, ge=20, le=80)
    hip_cm: float | None = Field(default=None, ge=50, le=220)
    goal: Literal["strength", "muscle", "fat_loss", "endurance", "general"] = "general"
    experience: Literal["beginner", "intermediate", "advanced"] = "beginner"
    equipment: list[str] | None = None
    session_minutes: int = Field(default=45, ge=5, le=180)
    limitations: list[str] = Field(default_factory=list)
    screening: Screening = "not_asked"
    activity_level: Literal[
        "unspecified", "sedentary", "lightly_active", "moderately_active",
        "very_active", "athlete",
    ] = "unspecified"
    dietary_pattern: Literal[
        "unspecified", "omnivore", "non_vegetarian", "vegetarian",
        "eggetarian", "vegan", "pescatarian", "jain",
    ] = "unspecified"
    preferred_foods: list[str] = Field(default_factory=list, max_length=30)
    excluded_foods: list[str] = Field(default_factory=list, max_length=30)
    allergies: list[str] = Field(default_factory=list, max_length=30)
    intolerances: list[str] = Field(default_factory=list, max_length=30)
    meals_per_day: int = Field(default=3, ge=1, le=8)
    daily_food_budget_inr: int | None = Field(default=None, ge=20, le=10000)

    @field_validator("preferred_foods", "excluded_foods", "allergies", "intolerances")
    @classmethod
    def _clean_food_lists(cls, value):
        cleaned, seen = [], set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("food preferences must be text")
            item = item.strip()
            if not item or len(item) > 60:
                raise ValueError("food preference entries must contain 1-60 characters")
            key = item.casefold()
            if key not in seen:
                cleaned.append(item)
                seen.add(key)
        return cleaned

    @field_validator("equipment")
    @classmethod
    def _check_equipment(cls, value):
        if value is None:
            return None
        cleaned = list(dict.fromkeys(item.strip().lower() for item in value))
        unknown = set(cleaned) - KNOWN_EQUIPMENT
        if unknown:
            raise ValueError(f"unknown equipment: {sorted(unknown)}")
        if "full_gym" in cleaned:
            cleaned = list(dict.fromkeys(cleaned + sorted(KNOWN_EQUIPMENT - {"bodyweight", "full_gym"})))
        return [item for item in cleaned if item != "bodyweight"]

    @field_validator("limitations")
    @classmethod
    def _check_limitations(cls, value):
        cleaned = list(dict.fromkeys(item.strip().lower() for item in value))
        unknown = set(cleaned) - KNOWN_LIMITATIONS
        if unknown:
            raise ValueError(f"unknown limitation: {sorted(unknown)}")
        return cleaned

    @model_validator(mode="after")
    def _validate_measurements(self):
        if self.waist_cm is not None and self.neck_cm is not None and self.waist_cm <= self.neck_cm:
            raise ValueError("waist_cm must be greater than neck_cm")
        if (
            self.sex == "female" and self.waist_cm is not None
            and self.hip_cm is not None and self.neck_cm is not None
            and self.waist_cm + self.hip_cm <= self.neck_cm
        ):
            raise ValueError("waist_cm + hip_cm must be greater than neck_cm")
        return self


def merged_profile(existing, patch):
    """Apply only fields the client sent; older clients must not reset newer fields."""
    base = existing.model_dump() if existing is not None else {}
    base.update(patch)
    return UserProfile(**base)
