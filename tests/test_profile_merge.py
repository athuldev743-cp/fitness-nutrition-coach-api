import pytest
from pydantic import ValidationError

from state.profile import UserProfile, merged_profile


def _saved():
    return UserProfile(
        name="Asha", goal="muscle", equipment=["dumbbells"],
        dietary_pattern="vegetarian", allergies=["peanut"],
        meals_per_day=4, daily_food_budget_inr=250,
    )


def test_fields_the_client_omits_keep_their_saved_values():
    patch = UserProfile(name="Asha", goal="strength").model_dump(exclude_unset=True)
    result = merged_profile(_saved(), patch)
    assert result.goal == "strength"
    assert result.dietary_pattern == "vegetarian"
    assert result.allergies == ["peanut"]
    assert result.meals_per_day == 4
    assert result.daily_food_budget_inr == 250
    assert result.equipment == ["dumbbells"]


def test_fields_the_client_sends_replace_saved_values_even_when_empty_or_none():
    patch = UserProfile(
        name="Asha", allergies=[], daily_food_budget_inr=None, equipment=None
    ).model_dump(exclude_unset=True)
    result = merged_profile(_saved(), patch)
    assert result.allergies == []
    assert result.daily_food_budget_inr is None
    assert result.equipment is None
    assert result.dietary_pattern == "vegetarian"


def test_first_save_without_an_existing_profile_uses_defaults():
    result = merged_profile(None, {"name": "Asha", "goal": "fat_loss"})
    assert result.goal == "fat_loss"
    assert result.dietary_pattern == "unspecified"
    assert result.meals_per_day == 3


def test_invalid_values_are_still_rejected():
    with pytest.raises(ValidationError):
        merged_profile(_saved(), {"meals_per_day": 0})