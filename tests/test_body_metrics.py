import pytest
from pydantic import ValidationError

from state.body_metrics import calculate_body_metrics
from state.profile import UserProfile
from state.store import ProfileStore


def test_bmi_and_mifflin_st_jeor_are_calculated_from_profile():
    profile = UserProfile(name="Test", age=30, sex="male", height_cm=180, weight_kg=80)
    result = calculate_body_metrics(profile)
    assert result["bmi"] == 24.7
    assert result["bmr_kcal_per_day"] == 1780
    assert result["estimated_body_fat_percent"] is None


def test_male_navy_estimate_requires_circumferences():
    profile = UserProfile(
        name="Test", age=30, sex="male", height_cm=180, weight_kg=80,
        waist_cm=88, neck_cm=38,
    )
    result = calculate_body_metrics(profile)
    assert result["body_fat_method"] == "US Navy circumference estimate"
    assert result["estimated_body_fat_percent"] is not None
    assert result["estimated_fat_mass_kg"] + result["estimated_lean_mass_kg"] == 80.0


def test_female_navy_estimate_requires_hip_measurement():
    profile = UserProfile(
        name="Test", age=30, sex="female", height_cm=165, weight_kg=65,
        waist_cm=75, neck_cm=32, hip_cm=98,
    )
    result = calculate_body_metrics(profile)
    assert result["estimated_body_fat_percent"] is not None
    assert result["body_fat_method"] == "US Navy circumference estimate"


def test_female_without_hip_does_not_return_body_fat_estimate():
    profile = UserProfile(
        name="Test", sex="female", height_cm=165, weight_kg=65,
        waist_cm=75, neck_cm=32,
    )
    result = calculate_body_metrics(profile)
    assert result["estimated_body_fat_percent"] is None
    assert "hip_cm" in result["missing_body_fat_inputs"]


def test_profile_rejects_invalid_circumference_relationship():
    with pytest.raises(ValidationError):
        UserProfile(name="Test", sex="male", waist_cm=35, neck_cm=40)


def test_profile_store_round_trips_new_body_measurements():
    store = ProfileStore()
    try:
        profile = UserProfile(
            name="Test", age=25, sex="male", height_cm=175, weight_kg=75,
            waist_cm=82, neck_cm=36,
        )
        store.save(profile)
        loaded = store.get()
        assert loaded.sex == "male"
        assert loaded.waist_cm == 82
        assert loaded.neck_cm == 36
        assert loaded.hip_cm is None
    finally:
        store.close()


def test_legacy_profile_defaults_new_fields_without_losing_profile(tmp_path):
    store = ProfileStore(tmp_path / "legacy.db", user_id="legacy")
    try:
        store.conn.execute(
            """INSERT INTO user_profile
               (user_id, name, age, height_cm, weight_kg, goal, experience, equipment,
                session_minutes, limitations, screening)
               VALUES ('legacy', 'Legacy', 28, 170, 70, 'general', 'beginner',
                       NULL, 45, '[]', 'not_asked')"""
        )
        store.conn.commit()
        profile = store.get()
        assert profile.sex == "unspecified"
        assert profile.waist_cm is None
        assert profile.neck_cm is None
        assert profile.hip_cm is None
    finally:
        store.close()
