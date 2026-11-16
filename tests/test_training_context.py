from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import api.app as api
from agent.contracts import AgentName, AgentRequest, AgentResult, AgentStatus, TrainingContext
from agent.nutrition import NutritionAgent
from agent.training_context import build_training_context

TODAY = date(2026, 11, 16)


def make_program():
    lower = SimpleNamespace(day_number=1, session_number=1, label="Lower body", est_minutes=52.4)
    upper = SimpleNamespace(day_number=2, session_number=1, label="Upper body", est_minutes=48.0)
    return SimpleNamespace(days_per_week=4, split="upper_lower",
                           weeks=[SimpleNamespace(number=1, sessions=[lower, upper])])


def row(day, planned_date, status):
    return {"week": 1, "day_number": day, "session_number": 1,
            "planned_date": planned_date, "status": status}


class FakePrograms:
    def __init__(self, rows):
        self.rows = rows

    def active_program(self):
        return {"id": 7, "program": make_program()}

    def sessions(self, program_id):
        return self.rows


def test_planned_session_today_is_reported_with_the_next_one():
    ctx = build_training_context(
        FakePrograms([row(1, "2026-11-16", "planned"), row(2, "2026-11-18", "planned")]), TODAY)
    assert ctx.today == "training_day_planned"
    assert ctx.today_session == "Lower body"
    assert ctx.today_minutes == 52
    assert ctx.sessions_per_week == 4
    assert ctx.next_session == "Upper body"
    assert ctx.next_session_date == "2026-11-18"


def test_session_finished_today_is_reported_as_done():
    ctx = build_training_context(
        FakePrograms([row(1, "2026-11-16", "done"), row(2, "2026-11-18", "planned")]), TODAY)
    assert ctx.today == "training_day_done"
    assert ctx.today_session == "Lower body"
    assert ctx.next_session == "Upper body"


def test_rest_day_has_no_session_for_today():
    ctx = build_training_context(
        FakePrograms([row(1, "2026-11-17", "planned"), row(2, "2026-11-19", "planned")]), TODAY)
    assert ctx.today == "rest_day"
    assert ctx.today_session is None
    assert ctx.today_minutes is None
    assert ctx.next_session == "Lower body"
    assert ctx.next_session_date == "2026-11-17"


def test_no_active_programme_gives_no_context():
    class NoProgram:
        def active_program(self):
            return None

    assert build_training_context(NoProgram(), TODAY) is None


def test_a_failing_store_never_breaks_the_request():
    class Broken:
        def active_program(self):
            raise RuntimeError("db down")

    assert build_training_context(Broken(), TODAY) is None


def test_context_carries_only_the_whitelisted_fields():
    ctx = build_training_context(FakePrograms([row(1, "2026-11-16", "planned")]), TODAY)
    assert set(ctx.model_dump()) == {
        "sessions_per_week", "today", "today_session", "today_minutes", "next_session", "next_session_date",
    }


def chunk():
    return {
        "chunk_id": "a" * 64,
        "content": "Dietary protein supports muscle protein synthesis.",
        "source_id": "sports-guide",
        "source": "sports_nutrition.pdf",
        "title": "Sports Nutrition",
        "page": 4,
        "chunk_number": 2,
        "rrf_score": 0.02,
    }


def run_agent(context):
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Protein helps recovery [S1]."}}]
    }
    agent = NutritionAgent(llm, retriever=lambda *a, **k: [chunk()])
    result = agent.run(AgentRequest(user_id="t", message="What should I eat before training?",
                                    training_context=context))
    return result, llm.create_chat_completion.call_args.kwargs["messages"][0]["content"]


def test_prompt_with_training_context_has_a_labelled_section_and_the_personal_marker():
    result, prompt = run_agent(TrainingContext(today="training_day_planned", today_session="Lower body"))
    assert result.status is AgentStatus.OK
    assert "TRAINING CONTEXT" in prompt
    assert "Lower body" in prompt
    assert "must never be cited" in prompt
    assert "FACTS:" in prompt


def test_prompt_without_training_context_is_unchanged():
    result, prompt = run_agent(None)
    assert result.status is AgentStatus.OK
    assert "TRAINING CONTEXT" not in prompt
    assert "FACTS:" not in prompt


class FakeProfileStore:
    def get(self):
        return None


def chat(monkeypatch, payload, handler=None):
    seen, built = [], []

    class StubNutritionAgent:
        name = AgentName.NUTRITION

        def __init__(self, llm):
            pass

        def run(self, request):
            seen.append(request)
            return AgentResult(request_id=request.request_id, agent=self.name,
                               status=AgentStatus.OK, message="Answer [S1].")

    def fake_build(programs, today):
        built.append(today)
        return TrainingContext(today="rest_day")

    monkeypatch.setattr(api, "stores", lambda user: {
        "profile": FakeProfileStore(), "programs": object(), "history": object(), "traces": FakeProfileStore(),
    })
    monkeypatch.setattr(api, "close", lambda stores: None)
    monkeypatch.setattr(api, "model", lambda: object())
    monkeypatch.setattr(api, "exercises", lambda: [])
    monkeypatch.setattr("agent.nutrition.NutritionAgent", StubNutritionAgent)
    monkeypatch.setattr("agent.training_context.build_training_context", fake_build)
    if handler:
        monkeypatch.setattr("agent.dispatcher.handle_message", handler)
    client = TestClient(api.app)
    api.app.dependency_overrides[api.current_user] = lambda: "verified-user"
    try:
        response = client.post("/v1/chat", json=payload)
    finally:
        api.app.dependency_overrides.clear()
    return response, seen, built


def test_pinned_nutrition_request_gets_the_training_context(monkeypatch):
    response, seen, built = chat(monkeypatch, {"message": "potassium intake recommendations", "agent": "nutrition"})
    assert response.status_code == 200
    assert len(built) == 1
    assert seen[0].training_context.today == "rest_day"


def test_keyword_routed_nutrition_request_gets_the_training_context(monkeypatch):
    response, seen, built = chat(monkeypatch, {"message": "How much protein should meals contain?"})
    assert response.status_code == 200
    assert len(built) == 1
    assert seen[0].training_context is not None


def test_fitness_only_request_does_not_build_the_training_context(monkeypatch):
    def handle(llm, message, exercises, profile, programs, today, history):
        return {"status": "ok", "tool": "search_exercises", "data": {}, "reply": "Workout options."}

    response, seen, built = chat(monkeypatch, {"message": "Suggest a workout"}, handler=handle)
    assert response.status_code == 200
    assert response.json()["agent"] == "fitness"
    assert built == []
