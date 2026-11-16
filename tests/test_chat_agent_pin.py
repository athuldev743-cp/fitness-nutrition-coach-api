from fastapi.testclient import TestClient

import api.app as api
from agent.contracts import AgentName, AgentResult, AgentStatus


class FakeStore:
    def get(self):
        return None


class StubNutritionAgent:
    name = AgentName.NUTRITION

    def __init__(self, llm):
        pass

    def run(self, request):
        return AgentResult(
            request_id=request.request_id, agent=self.name, status=AgentStatus.OK,
            message="Potassium guidance [S1].",
            metadata={"citation_validation": "passed"},
        )


def post(monkeypatch, payload):
    monkeypatch.setattr(api, "stores", lambda user: {
        "profile": FakeStore(), "programs": object(), "history": object(), "traces": FakeStore(),
    })
    monkeypatch.setattr(api, "close", lambda stores: None)
    monkeypatch.setattr(api, "model", lambda: object())
    monkeypatch.setattr(api, "exercises", lambda: [])
    monkeypatch.setattr("agent.nutrition.NutritionAgent", StubNutritionAgent)
    client = TestClient(api.app)
    api.app.dependency_overrides[api.current_user] = lambda: "verified-user"
    try:
        return client.post("/v1/chat", json=payload)
    finally:
        api.app.dependency_overrides.clear()


def test_pinned_nutrition_agent_answers_a_question_with_no_routing_keywords(monkeypatch):
    response = post(monkeypatch, {"message": "potassium intake recommendations", "agent": "nutrition"})
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["agent"] == "nutrition"
    assert body["tool"] == "nutrition_agent"
    assert body["reply"] == "Potassium guidance [S1]."
    assert body["metadata"]["nutrition"]["citation_validation"] == "passed"


def test_the_same_question_without_a_pin_still_asks_which_area(monkeypatch):
    response = post(monkeypatch, {"message": "potassium intake recommendations"})

    assert response.status_code == 200
    assert response.json()["status"] == "needs_info"


def test_unknown_agent_value_is_rejected(monkeypatch):
    response = post(monkeypatch, {"message": "hello", "agent": "dietitian"})

    assert response.status_code == 422
