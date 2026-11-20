"""Saved diet plan: what is missing, when a plan is out of date, storage, API flow, and the chat fixes."""
import pytest
from fastapi.testclient import TestClient

import api.app as api
from agent.contracts import AgentName, AgentRequest, AgentResult, AgentStatus, TrainingContext
from agent.diet_plan import DIET_REQUEST_MESSAGE, effective_activity, fingerprint, readiness
from agent.nutrition import _DIET_PLAN_REQUEST, SMALL_TALK_REPLY, NutritionAgent
from state import db as sqlite_db
from state import pg_compat
from state.account import delete_account, export_account
from state.diet import DietPlanStore
from state.profile import UserProfile


def full_profile(**over):
    base = dict(name="T", age=28, height_cm=175, weight_kg=72, dietary_pattern="vegetarian",
                activity_level="moderately_active", goal="muscle")
    base.update(over)
    return UserProfile(**base)


def plan(days=4, split="upper_lower"):
    return TrainingContext(today="rest_day", sessions_per_week=days, split=split,
                           program_goal="muscle", program_level="beginner")


# --- readiness ------------------------------------------------------------------------

def test_no_profile_needs_everything():
    r = readiness(None, None)
    assert not r["ready"]
    assert set(r["missing"]) == {"activity_level", "dietary_pattern", "age", "height_cm", "weight_kg"}


def test_complete_profile_is_ready():
    r = readiness(full_profile(), None)
    assert r["ready"] and r["missing"] == [] and r["activity_source"] == "profile"
    assert r["has_workout_plan"] is False


def test_unspecified_activity_without_a_plan_is_missing_with_five_choices():
    r = readiness(full_profile(activity_level="unspecified"), None)
    assert r["missing"] == ["activity_level"]
    q = r["questions"][0]
    assert q["field"] == "activity_level" and q["type"] == "choice"
    assert [o["value"] for o in q["options"]] == [
        "sedentary", "lightly_active", "moderately_active", "very_active", "athlete"]


def test_workout_plan_supplies_activity_level_when_unspecified():
    r = readiness(full_profile(activity_level="unspecified"), plan(5))
    assert r["ready"] and r["activity_level"] == "very_active" and r["activity_source"] == "workout_plan"
    assert r["has_workout_plan"] is True


def test_stated_level_beats_the_plan():
    assert effective_activity(full_profile(activity_level="sedentary"), plan(6)) == ("sedentary", "profile")


def test_missing_body_and_diet_fields_are_listed_in_order():
    r = readiness(UserProfile(name="T", activity_level="athlete"), None)
    assert r["missing"] == ["dietary_pattern", "age", "height_cm", "weight_kg"]
    kinds = {q["field"]: q["type"] for q in r["questions"]}
    assert kinds["dietary_pattern"] == "choice" and kinds["age"] == "number"


# --- fingerprint ----------------------------------------------------------------------

def test_fingerprint_is_stable_and_tracks_real_changes():
    base = fingerprint(full_profile(), plan())
    assert base == fingerprint(full_profile(), plan())
    assert base != fingerprint(full_profile(dietary_pattern="vegan"), plan())
    assert base != fingerprint(full_profile(weight_kg=80), plan())
    assert base != fingerprint(full_profile(), plan(days=6))
    assert base != fingerprint(full_profile(), None)


def test_fingerprint_ignores_which_day_it_is():
    a = TrainingContext(today="rest_day", sessions_per_week=4, split="upper_lower")
    b = TrainingContext(today="training_day_planned", sessions_per_week=4, split="upper_lower",
                        next_session="Push", next_session_date="2026-11-20")
    assert fingerprint(full_profile(), a) == fingerprint(full_profile(), b)


# --- storage --------------------------------------------------------------------------

def test_store_round_trip_and_replace(tmp_path):
    store = DietPlanStore(str(tmp_path / "t.db"), user_id="u1")
    assert store.get() is None
    store.save({"text": "Plan one", "evidence": []}, "fp1")
    store.save({"text": "Plan two", "evidence": []}, "fp2")
    got = store.get()
    assert got["text"] == "Plan two" and got["fingerprint"] == "fp2" and got["created_at"]


def test_users_do_not_see_each_others_plans(tmp_path):
    path = str(tmp_path / "t.db")
    DietPlanStore(path, user_id="a").save({"text": "A"}, "x")
    assert DietPlanStore(path, user_id="b").get() is None
    assert DietPlanStore(path, user_id="a").get()["text"] == "A"


def test_account_delete_and_export_cover_the_plan(tmp_path):
    store = DietPlanStore(str(tmp_path / "t.db"), user_id="gone")
    store.save({"text": "Mine"}, "x")
    assert export_account(store.conn, "gone")["diet_plan"]["data"]
    assert delete_account(store.conn, "gone")["diet_plans"] == 1
    assert store.get() is None


def test_postgres_schema_matches_sqlite():
    assert pg_compat.SCHEMA_VERSION == sqlite_db.SCHEMA_VERSION == 12
    assert any("diet_plans" in stmt for stmt in pg_compat.DDL)


# --- nutrition agent fixes --------------------------------------------------------------

class CountingLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create_chat_completion(self, messages, max_tokens, temperature):
        self.calls.append(messages)
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


def retriever(query, **kwargs):
    return [{"content": "Protein supports recovery.", "source": "fixture", "page": 1, "chunk_number": 0}]


@pytest.mark.parametrize("text", ["hi", "Hi!", "hello", "hey", "thanks", "Thank you!", "ok"])
def test_greetings_are_answered_without_the_model(text):
    llm = CountingLLM([])
    result = NutritionAgent(llm, retriever).run(AgentRequest(user_id="u", message=text, profile=full_profile()))
    assert result.status is AgentStatus.NEEDS_CLARIFICATION and result.message == SMALL_TALK_REPLY
    assert llm.calls == []


@pytest.mark.parametrize("text", ["hi, how much protein do I need?", "hello protein", "good protein sources"])
def test_real_questions_are_not_treated_as_greetings(text):
    llm = CountingLLM(["Protein helps [S1]."])
    result = NutritionAgent(llm, retriever).run(AgentRequest(user_id="u", message=text, profile=full_profile()))
    assert result.status is AgentStatus.OK and len(llm.calls) == 1


def test_citation_retry_shows_the_model_its_earlier_answer():
    llm = CountingLLM(["Protein helps recovery.", "Protein helps recovery [S1]."])
    result = NutritionAgent(llm, retriever).run(
        AgentRequest(user_id="u", message="Tell me about protein", profile=full_profile()))
    assert result.status is AgentStatus.OK and len(llm.calls) == 2
    retry = llm.calls[1]
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2]["content"] == "Protein helps recovery."


def test_request_message_is_recognised_as_a_diet_plan_request():
    assert _DIET_PLAN_REQUEST.search(DIET_REQUEST_MESSAGE)
    assert len(DIET_REQUEST_MESSAGE) <= 1000


# --- API flow -----------------------------------------------------------------------------

class FakeProfiles:
    def __init__(self, profile):
        self.profile, self.updates = profile, []

    def get(self):
        return self.profile

    def update(self, **fields):
        self.updates.append(fields)
        self.profile = UserProfile(**{**self.profile.model_dump(), **fields})
        return self.profile


class Harness:
    def __init__(self, monkeypatch, tmp_path, profile, training=None, outcome=AgentStatus.OK):
        self.profiles = FakeProfiles(profile)
        self.training, self.outcome, self.requests = training, outcome, []
        harness = self

        class StubAgent:
            name = AgentName.NUTRITION

            def __init__(self, llm):
                pass

            def run(self, req):
                harness.requests.append(req)
                return AgentResult(request_id=req.request_id, agent=self.name, status=harness.outcome,
                                   message="Breakfast: oats [S1]." if harness.outcome is AgentStatus.OK
                                   else "Search is down.")

        monkeypatch.setattr(api, "DB", str(tmp_path / "t.db"))
        monkeypatch.setattr(api, "stores", lambda user: {"profile": self.profiles, "programs": object(),
                                                         "history": object(), "traces": object()})
        monkeypatch.setattr(api, "close", lambda stores: None)
        monkeypatch.setattr(api, "model", lambda: object())
        monkeypatch.setattr("agent.nutrition.NutritionAgent", StubAgent)
        monkeypatch.setattr("agent.training_context.build_training_context",
                            lambda programs, today: harness.training)
        self.client = TestClient(api.app)
        api.app.dependency_overrides[api.current_user] = lambda: "verified-user"

    def get(self):
        return self.client.get("/v1/diet/plan").json()

    def post(self, **body):
        return self.client.post("/v1/diet/plan", json=body).json()


@pytest.fixture
def harness(monkeypatch, tmp_path):
    made = []

    def build(profile, **kw):
        h = Harness(monkeypatch, tmp_path, profile, **kw)
        made.append(h)
        return h

    yield build
    api.app.dependency_overrides.clear()


def test_new_user_sees_no_plan_and_the_open_questions(harness):
    h = harness(full_profile(activity_level="unspecified"))
    body = h.get()
    assert body["status"] == "none" and body["plan"] is None
    assert body["readiness"]["missing"] == ["activity_level"]


def test_post_asks_instead_of_generating_when_details_are_missing(harness):
    h = harness(full_profile(activity_level="unspecified"))
    body = h.post()
    assert body["status"] == "needs_info" and body["plan"] is None
    assert body["readiness"]["questions"][0]["field"] == "activity_level"
    assert h.requests == []


def test_answering_the_question_saves_the_level_and_creates_the_plan(harness):
    h = harness(full_profile(activity_level="unspecified"))
    body = h.post(activity_level="lightly active")
    assert body["status"] == "ok"
    assert body["plan"]["text"] == "Breakfast: oats [S1]."
    assert body["plan"]["activity_level"] == "lightly_active" and body["plan"]["activity_source"] == "profile"
    assert h.profiles.updates == [{"activity_level": "lightly_active"}]
    assert h.requests[0].message == DIET_REQUEST_MESSAGE
    assert h.get()["status"] == "ok"


def test_unclear_answer_asks_again_and_saves_nothing(harness):
    h = harness(full_profile(activity_level="unspecified"))
    body = h.post(activity_level="banana")
    assert body["status"] == "needs_info" and h.profiles.updates == [] and h.requests == []


def test_workout_plan_means_no_question_and_the_plan_says_so(harness):
    h = harness(full_profile(activity_level="unspecified"), training=plan(5))
    body = h.post()
    assert body["status"] == "ok"
    assert body["plan"]["activity_level"] == "very_active" and body["plan"]["activity_source"] == "workout_plan"
    assert h.requests[0].training_context is not None


def test_same_inputs_return_the_saved_plan_without_another_model_call(harness):
    h = harness(full_profile())
    h.post()
    again = h.post()
    assert again["status"] == "ok" and len(h.requests) == 1
    h.post(refresh=True)
    assert len(h.requests) == 2


def test_changed_profile_marks_the_plan_out_of_date(harness):
    h = harness(full_profile())
    h.post()
    assert h.get()["plan"]["stale"] is False
    h.profiles.profile = full_profile(weight_kg=90)
    assert h.get()["plan"]["stale"] is True
    h.post()
    assert len(h.requests) == 2 and h.get()["plan"]["stale"] is False


def test_failed_generation_is_reported_and_nothing_is_saved(harness):
    h = harness(full_profile(), outcome=AgentStatus.UNAVAILABLE)
    body = h.post()
    assert body["status"] == "unavailable" and body["reply"] == "Search is down." and body["plan"] is None
    assert h.get()["status"] == "none"


def test_a_failed_refresh_keeps_the_old_plan(harness):
    h = harness(full_profile())
    h.post()
    h.outcome = AgentStatus.UNAVAILABLE
    h.profiles.profile = full_profile(weight_kg=90)
    body = h.post()
    assert body["status"] == "unavailable" and body["plan"]["text"] == "Breakfast: oats [S1]."
    assert body["plan"]["stale"] is True


def test_no_profile_yet_asks_to_finish_it(harness):
    h = harness(None)
    body = h.post()
    assert body["status"] == "needs_info" and body["readiness"]["ready"] is False


def test_unknown_fields_are_rejected(harness):
    h = harness(full_profile())
    assert h.client.post("/v1/diet/plan", json={"message": "hack"}).status_code == 422


def test_agent_result_accepts_a_full_length_diet_plan():
    from uuid import uuid4
    from agent.contracts import AgentResult, AgentName, AgentStatus
    r = AgentResult(request_id=uuid4(), agent=list(AgentName)[0], status=AgentStatus.OK, message="x" * 12000)
    assert len(r.message) == 12000
