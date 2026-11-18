"""Nutrition agent: ask for an activity level when there is no workout plan, and derive one when there is."""
import pytest
from fastapi.testclient import TestClient

import api.app as api
from agent.activity import (
    ACTIVITY_LEVELS,
    ACTIVITY_QUESTION_RETRY,
    derive_activity_level,
    parse_activity_level,
)
from agent.contracts import AgentName, AgentRequest, AgentResult, AgentStatus, TrainingContext
from agent.nutrition import NutritionAgent, _DIET_PLAN_REQUEST
from state.profile import UserProfile


class ExplodingLLM:
    def create_chat_completion(self, **kwargs):
        raise AssertionError("the model must not be called")


class RecordingLLM:
    def __init__(self, reply="Spread protein across meals [S1]."):
        self.reply, self.calls = reply, []

    def create_chat_completion(self, messages, max_tokens, temperature):
        self.calls.append(messages)
        return {"choices": [{"message": {"content": self.reply}}]}


def exploding_retriever(*args, **kwargs):
    raise AssertionError("retrieval must not run")


def retriever(query, **kwargs):
    return [{"content": "Protein supports recovery after resistance training.",
             "source": "fixture", "page": 1, "chunk_number": 0}]


def profile(level="unspecified"):
    return UserProfile(name="Test", goal="muscle", activity_level=level)


def request(message="Create a diet plan for me", prof=None, **extra):
    return AgentRequest(user_id="u", message=message, profile=prof, **extra)


def plan(sessions_per_week=4):
    return TrainingContext(today="rest_day", sessions_per_week=sessions_per_week)


# --- helpers --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("sedentary", "sedentary"),
    ("Lightly active", "lightly_active"),
    ("lightly_active", "lightly_active"),
    ("moderately active", "moderately_active"),
    ("moderate", "moderately_active"),
    ("Very Active", "very_active"),
    ("athlete", "athlete"),
    ("  I'd say moderately active  ", "moderately_active"),
])
def test_parse_accepts_typed_and_selected_answers(text, expected):
    assert parse_activity_level(text) == expected
    assert expected in ACTIVITY_LEVELS


@pytest.mark.parametrize("text", [
    "", "   ", None, "banana", "not very active", "I'm not active", "light and moderate", "x" * 100,
])
def test_parse_returns_none_when_unclear_instead_of_guessing(text):
    assert parse_activity_level(text) is None


@pytest.mark.parametrize("days,expected", [
    (None, None), (0, None), (1, "lightly_active"), (2, "lightly_active"),
    (3, "moderately_active"), (4, "moderately_active"),
    (5, "very_active"), (6, "very_active"), (7, "very_active"), (8, None), (True, None),
])
def test_derived_level_comes_only_from_weekly_frequency(days, expected):
    assert derive_activity_level(days) == expected


@pytest.mark.parametrize("message", [
    "Create a diet plan for me", "I want a meal plan", "Design my diet",
    "build a vegetarian diet", "give me a diet chart", "set my calorie target",
])
def test_diet_plan_requests_are_recognised(message):
    assert _DIET_PLAN_REQUEST.search(message)


@pytest.mark.parametrize("message", [
    "Explain protein in my diet", "How much protein should meals contain?", "Is creatine safe?",
    "potassium intake recommendations", "What should I eat before training?",
])
def test_general_nutrition_questions_are_not_gated(message):
    assert not _DIET_PLAN_REQUEST.search(message)


# --- agent gate -----------------------------------------------------------------------

def test_no_plan_and_no_activity_level_asks_instead_of_calling_the_model():
    result = NutritionAgent(ExplodingLLM(), exploding_retriever).run(request(prof=profile()))
    assert result.status is AgentStatus.NEEDS_CLARIFICATION
    assert "no ongoing workout plan" in result.message
    assert "activity level" in result.message
    assert result.metadata["needs"] == "activity_level"
    assert [o["value"] for o in result.metadata["options"]] == list(ACTIVITY_LEVELS)


def test_no_saved_profile_also_asks():
    result = NutritionAgent(ExplodingLLM(), exploding_retriever).run(request(prof=None))
    assert result.status is AgentStatus.NEEDS_CLARIFICATION
    assert result.metadata["needs"] == "activity_level"


@pytest.mark.parametrize("kwargs", [
    {"prof": profile("moderately_active")},
    {"prof": profile(), "training_context": plan()},
    {"prof": profile(), "fitness_context": "Here is your upper/lower workout plan."},
])
def test_gate_stays_closed_when_activity_is_known(kwargs):
    llm = RecordingLLM()
    result = NutritionAgent(llm, retriever).run(request(**kwargs))
    assert result.status is AgentStatus.OK
    assert len(llm.calls) == 1


def test_general_nutrition_question_without_a_plan_is_still_answered():
    llm = RecordingLLM()
    result = NutritionAgent(llm, retriever).run(request("What does protein do?", profile()))
    assert result.status is AgentStatus.OK


def test_activity_level_is_derived_from_the_plan_when_unspecified():
    llm = RecordingLLM()
    NutritionAgent(llm, retriever).run(request(prof=profile(), training_context=plan(4)))
    assert '"activity_level_from_workout_plan": "moderately_active"' in llm.calls[0][0]["content"]


def test_stated_activity_level_is_never_overridden_by_the_plan():
    llm = RecordingLLM()
    NutritionAgent(llm, retriever).run(request(prof=profile("sedentary"), training_context=plan(6)))
    system = llm.calls[0][0]["content"]
    assert "activity_level_from_workout_plan" not in system.split("USER PROFILE")[1]
    assert '"activity_level": "sedentary"' in system


# --- API flow -------------------------------------------------------------------------

class FakeProfileStore:
    def __init__(self, current):
        self.current, self.updates = current, []

    def get(self):
        return self.current

    def update(self, **fields):
        self.updates.append(fields)
        self.current = UserProfile(**{**self.current.model_dump(), **fields})
        return self.current


def call(monkeypatch, payload, store, path="/v1/chat", plan_context=None):
    seen = []

    class StubNutritionAgent:
        name = AgentName.NUTRITION

        def __init__(self, llm):
            pass

        def run(self, req):
            seen.append(req)
            return AgentResult(request_id=req.request_id, agent=self.name,
                               status=AgentStatus.OK, message="Your plan [S1].")

    monkeypatch.setattr(api, "stores", lambda user: {
        "profile": store, "programs": object(), "history": object(), "traces": object(),
    })
    monkeypatch.setattr(api, "close", lambda stores: None)
    monkeypatch.setattr(api, "model", lambda: object())
    monkeypatch.setattr(api, "exercises", lambda: [])
    monkeypatch.setattr("agent.nutrition.NutritionAgent", StubNutritionAgent)
    monkeypatch.setattr("agent.training_context.build_training_context", lambda programs, today: plan_context)
    client = TestClient(api.app)
    api.app.dependency_overrides[api.current_user] = lambda: "verified-user"
    try:
        response = client.post(path, json=payload)
    finally:
        api.app.dependency_overrides.clear()
    return response, seen


def test_typed_answer_is_saved_and_the_diet_request_continues(monkeypatch):
    store = FakeProfileStore(profile())
    response, seen = call(monkeypatch, {"message": "Create a diet plan for me",
                                        "activity_level": "lightly active"}, store)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert store.updates == [{"activity_level": "lightly_active"}]
    assert seen[0].profile.activity_level == "lightly_active"


def test_selected_value_is_accepted_too(monkeypatch):
    store = FakeProfileStore(profile())
    response, _ = call(monkeypatch, {"message": "Create a diet plan for me",
                                     "activity_level": "very_active"}, store)
    assert response.json()["status"] == "ok"
    assert store.updates == [{"activity_level": "very_active"}]


def test_unclear_answer_asks_again_and_saves_nothing(monkeypatch):
    store = FakeProfileStore(profile())
    response, seen = call(monkeypatch, {"message": "Create a diet plan for me",
                                        "activity_level": "banana"}, store)
    body = response.json()
    assert body["status"] == "needs_info"
    assert body["reply"] == ACTIVITY_QUESTION_RETRY
    assert len(body["metadata"]["nutrition"]["options"]) == 5
    assert store.updates == [] and seen == []


def test_answer_without_a_saved_profile_applies_to_this_request_only(monkeypatch):
    class NoProfile:
        def get(self):
            return None

    response, seen = call(monkeypatch, {"message": "Create a diet plan for me",
                                        "activity_level": "athlete"}, NoProfile())
    assert response.json()["status"] == "ok"
    assert seen[0].profile.activity_level == "athlete"


def test_invalid_activity_field_length_is_rejected(monkeypatch):
    response, _ = call(monkeypatch, {"message": "Create a diet plan", "activity_level": "x" * 61},
                       FakeProfileStore(profile()))
    assert response.status_code == 422


def test_stream_endpoint_keeps_the_options_for_the_client(monkeypatch):
    seen = []

    class RealGateAgent(NutritionAgent):
        def __init__(self, llm):
            super().__init__(ExplodingLLM(), exploding_retriever)

    monkeypatch.setattr(api, "stores", lambda user: {
        "profile": FakeProfileStore(profile()), "programs": object(),
        "history": object(), "traces": object(),
    })
    monkeypatch.setattr(api, "close", lambda stores: None)
    monkeypatch.setattr(api, "model", lambda: object())
    monkeypatch.setattr(api, "exercises", lambda: [])
    monkeypatch.setattr("agent.nutrition.NutritionAgent", RealGateAgent)
    monkeypatch.setattr("agent.training_context.build_training_context", lambda programs, today: None)
    client = TestClient(api.app)
    api.app.dependency_overrides[api.current_user] = lambda: "verified-user"
    try:
        response = client.post("/v1/chat/stream", json={"message": "Create a diet plan for me"})
    finally:
        api.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert "no ongoing workout plan" in response.text
    assert '"needs": "activity_level"' in response.text
    assert "lightly_active" in response.text
    assert seen == []
