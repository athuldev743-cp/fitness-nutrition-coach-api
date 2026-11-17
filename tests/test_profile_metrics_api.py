from fastapi.testclient import TestClient

import api.app as api
from state.profile import UserProfile


def test_authenticated_profile_metrics_endpoint_returns_estimates(monkeypatch):
    class FakeProfileStore:
        def get(self):
            return UserProfile(
                name="Test", age=30, sex="male", height_cm=180, weight_kg=80,
                waist_cm=88, neck_cm=38,
            )

    monkeypatch.setattr(api, "stores", lambda user: {"profile": FakeProfileStore()})
    monkeypatch.setattr(api, "close", lambda stores: None)
    api.app.dependency_overrides[api.current_user] = lambda: "metrics-user"
    try:
        response = TestClient(api.app).get("/v1/profile/metrics")
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["bmi"] == 24.7
    assert data["bmr_kcal_per_day"] == 1780
    assert data["estimated_body_fat_percent"] is not None
    assert data["estimated_fat_mass_kg"] + data["estimated_lean_mass_kg"] == 80.0


def test_profile_metrics_endpoint_requires_a_saved_profile(monkeypatch):
    class FakeProfileStore:
        def get(self):
            return None

    monkeypatch.setattr(api, "stores", lambda user: {"profile": FakeProfileStore()})
    monkeypatch.setattr(api, "close", lambda stores: None)
    api.app.dependency_overrides[api.current_user] = lambda: "metrics-user"
    try:
        response = TestClient(api.app).get("/v1/profile/metrics")
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 404
