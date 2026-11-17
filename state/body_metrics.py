"""Deterministic anthropometric estimates. These are screening estimates, not diagnoses."""
from __future__ import annotations

from math import log10

from state.profile import UserProfile


def calculate_body_metrics(profile: UserProfile) -> dict:
    height = profile.height_cm
    weight = profile.weight_kg
    bmi = round(weight / ((height / 100) ** 2), 1) if height and weight else None

    bmr = None
    if all((profile.age, height, weight)) and profile.sex in {"male", "female"}:
        offset = 5 if profile.sex == "male" else -161
        bmr = round(10 * weight + 6.25 * height - 5 * profile.age + offset)

    body_fat = None
    method = None
    if height and profile.neck_cm and profile.waist_cm and profile.sex == "male":
        if profile.waist_cm > profile.neck_cm:
            body_fat = 86.010 * log10((profile.waist_cm - profile.neck_cm) / 2.54)
            body_fat -= 70.041 * log10(height / 2.54)
            body_fat += 36.76
            method = "US Navy circumference estimate"
    elif height and profile.neck_cm and profile.waist_cm and profile.hip_cm and profile.sex == "female":
        circumference = profile.waist_cm + profile.hip_cm - profile.neck_cm
        if circumference > 0:
            body_fat = 163.205 * log10(circumference / 2.54)
            body_fat -= 97.684 * log10(height / 2.54)
            body_fat -= 78.387
            method = "US Navy circumference estimate"

    if body_fat is not None and not 2 <= body_fat <= 75:
        body_fat, method = None, None
    body_fat = round(body_fat, 1) if body_fat is not None else None
    fat_mass = round(weight * body_fat / 100, 1) if weight and body_fat is not None else None
    lean_mass = round(weight - fat_mass, 1) if weight and fat_mass is not None else None

    return {
        "bmi": bmi,
        "bmr_kcal_per_day": bmr,
        "estimated_body_fat_percent": body_fat,
        "body_fat_method": method,
        "estimated_fat_mass_kg": fat_mass,
        "estimated_lean_mass_kg": lean_mass,
        "missing_body_fat_inputs": (
            [] if body_fat is not None else
            ["sex", "height_cm", "waist_cm", "neck_cm"] +
            (["hip_cm"] if profile.sex == "female" else [])
        ),
        "disclaimer": "Estimates only. Circumference-based body-fat results can differ from DEXA and depend on accurate measurement technique.",
    }
