from unittest.mock import MagicMock

from agent.cloud_llm import CloudOnlyLLM
from agent.contracts import AgentRequest, AgentStatus
from agent.nutrition import NutritionAgent
from state.profile import UserProfile

QUESTION = "How does protein support muscle recovery?"


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


def ask(agent, profile=None):
    return agent.run(AgentRequest(user_id="test-user", message=QUESTION, profile=profile))


def cited_reply():
    return {"choices": [{"message": {"content": "Protein supports muscle protein synthesis [S1]."}}]}


def system_prompt(llm):
    return llm.create_chat_completion.call_args.kwargs["messages"][0]["content"]


def test_prompt_with_a_saved_profile_is_marked_personal():
    llm = MagicMock()
    llm.create_chat_completion.return_value = cited_reply()
    ask(NutritionAgent(llm, retriever=lambda *a, **k: [chunk()]), UserProfile(name="Asha", allergies=["peanut"]))

    assert "FACTS:" in system_prompt(llm)
    assert "peanut" in system_prompt(llm)


def test_prompt_without_a_profile_is_not_marked_personal():
    llm = MagicMock()
    llm.create_chat_completion.return_value = cited_reply()
    ask(NutritionAgent(llm, retriever=lambda *a, **k: [chunk()]))

    assert "FACTS:" not in system_prompt(llm)


def counting_llm():
    calls = {"n": 0}

    def fake_post(url, key, body, timeout):
        calls["n"] += 1
        return {"choices": [{"message": {"content": "Protein supports muscle protein synthesis [S1]."},
                             "finish_reason": "stop"}]}

    llm = CloudOnlyLLM(post=fake_post, env={"GROQ_API_KEY": "a", "OPENROUTER_API_KEY": "b"})
    return llm, calls


def test_answers_built_from_a_saved_profile_are_not_cached():
    llm, calls = counting_llm()
    agent = NutritionAgent(llm, retriever=lambda *a, **k: [chunk()])
    profile = UserProfile(name="Asha", allergies=["peanut"])

    assert ask(agent, profile).status is AgentStatus.OK
    assert ask(agent, profile).status is AgentStatus.OK
    assert calls["n"] == 2


def test_profile_free_nutrition_answers_are_still_cached():
    llm, calls = counting_llm()
    agent = NutritionAgent(llm, retriever=lambda *a, **k: [chunk()])

    ask(agent)
    ask(agent)
    assert calls["n"] == 1
