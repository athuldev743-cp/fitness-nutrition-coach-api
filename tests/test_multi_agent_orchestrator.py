
from unittest.mock import MagicMock

from agent.contracts import (
    AgentName,
    AgentRequest,
    AgentResult,
    AgentStatus,
    TrainingContext,
)
from agent.fitness_adapter import FitnessAgentAdapter
from agent.orchestrator import MultiAgentOrchestrator, classify_request


def request(message):
    return AgentRequest(user_id="orchestration-test", message=message)


class StubAgent:
    def __init__(self, name, status=AgentStatus.OK, message="done"):
        self.name = name
        self.status = status
        self.message = message
        self.calls = 0

    def run(self, req):
        self.calls += 1
        return AgentResult(
            request_id=req.request_id,
            agent=self.name,
            status=self.status,
            message=self.message,
        )


def make_graph(fitness=None, nutrition=None):
    return MultiAgentOrchestrator(
        fitness or StubAgent(
            AgentName.FITNESS,
            message="Workout plan ready.",
        ),
        nutrition or StubAgent(
            AgentName.NUTRITION,
            message="Nutrition facts [S1].",
        ),
    )


def test_classifies_fitness_only():
    assert classify_request("Suggest a workout routine") == [
        AgentName.FITNESS
    ]


def test_classifies_progressive_overload_as_fitness():
    assert classify_request("What is progressive overload?") == [
        AgentName.FITNESS
    ]


def test_classifies_nutrition_only():
    assert classify_request("How much protein should meals contain?") == [
        AgentName.NUTRITION
    ]


def test_classifies_mixed_request_for_both_agents():
    assert classify_request(
        "Create a workout and suggest meals for recovery"
    ) == [AgentName.FITNESS, AgentName.NUTRITION]


def test_ambiguous_request_asks_for_clarification_without_calling_agents():
    fitness = StubAgent(AgentName.FITNESS)
    nutrition = StubAgent(AgentName.NUTRITION)

    result = make_graph(fitness, nutrition).invoke(request("Hello there"))

    assert result.status is AgentStatus.NEEDS_CLARIFICATION
    assert fitness.calls == nutrition.calls == 0


def test_fitness_only_routes_to_fitness_agent():
    fitness = StubAgent(AgentName.FITNESS)
    nutrition = StubAgent(AgentName.NUTRITION)

    result = make_graph(fitness, nutrition).invoke(
        request("Help with my workout")
    )

    assert result.agent is AgentName.FITNESS
    assert fitness.calls == 1
    assert nutrition.calls == 0


def test_nutrition_only_routes_to_nutrition_agent():
    fitness = StubAgent(AgentName.FITNESS)
    nutrition = StubAgent(AgentName.NUTRITION)

    result = make_graph(fitness, nutrition).invoke(
        request("Explain protein in my diet")
    )

    assert result.agent is AgentName.NUTRITION
    assert fitness.calls == 0
    assert nutrition.calls == 1


def test_mixed_request_invokes_both_and_aggregates_results():
    fitness = StubAgent(AgentName.FITNESS)
    nutrition = StubAgent(AgentName.NUTRITION)

    result = make_graph(fitness, nutrition).invoke(
        request("Build a workout and recommend meals")
    )

    assert fitness.calls == nutrition.calls == 1
    assert result.status is AgentStatus.OK
    assert "Fitness:" in result.message
    assert "Nutrition:" in result.message


def test_mixed_request_passes_fitness_and_training_context_to_nutrition():
    fitness_message = (
        "User follows an upper/lower split, training four days weekly."
    )
    fitness = StubAgent(
        AgentName.FITNESS,
        message=fitness_message,
    )

    received = {}

    def capture_nutrition_request(req):
        received["request"] = req
        return AgentResult(
            request_id=req.request_id,
            agent=AgentName.NUTRITION,
            status=AgentStatus.OK,
            message="Nutrition advice ready.",
        )

    nutrition = StubAgent(AgentName.NUTRITION)

    def nutrition_run(req):
     return capture_nutrition_request(req)

    nutrition.run = nutrition_run

    training = TrainingContext(
        program_goal="muscle",
        program_level="beginner",
        split="upper_lower",
        sessions_per_week=4,
        today="training_day_planned",
        today_session="Lower body",
    )

    req = AgentRequest(
        user_id="integration-test",
        message="Build a workout and recommend meals for recovery",
        training_context=training,
    )

    result = make_graph(fitness, nutrition).invoke(req)

    nutrition_request = received["request"]

    assert result.status is AgentStatus.OK
    assert received.get("request") is not None
    assert nutrition_request.fitness_context == fitness_message
    assert nutrition_request.training_context == training
    assert nutrition_request.training_context.split == "upper_lower"


def test_mixed_request_does_not_hide_agent_failure():
    fitness = StubAgent(AgentName.FITNESS)
    nutrition = StubAgent(
        AgentName.NUTRITION,
        status=AgentStatus.UNAVAILABLE,
        message="Nutrition search unavailable.",
    )

    result = make_graph(fitness, nutrition).invoke(
        request("Workout and nutrition advice")
    )

    assert result.status is AgentStatus.UNAVAILABLE


def test_fitness_adapter_maps_legacy_dispatcher_result():
    adapter = FitnessAgentAdapter(
        lambda req: {
            "status": "ok",
            "reply": "Workout ready.",
        }
    )

    result = adapter.run(request("workout"))

    assert result.agent is AgentName.FITNESS
    assert result.status is AgentStatus.OK
    assert result.message == "Workout ready."


def test_fitness_adapter_fails_closed_on_handler_exception():
    def fail(req):
        raise RuntimeError("private store details")

    result = FitnessAgentAdapter(fail).run(request("workout"))

    assert result.status is AgentStatus.UNAVAILABLE
    assert "private" not in result.message


def test_agent_identity_is_validated():
    import pytest

    with pytest.raises(ValueError):
        MultiAgentOrchestrator(
            StubAgent(AgentName.NUTRITION),
            StubAgent(AgentName.NUTRITION),
        )


def test_forcing_nutrition_skips_keyword_routing():
    fitness = StubAgent(AgentName.FITNESS)
    nutrition = StubAgent(AgentName.NUTRITION)

    result = make_graph(fitness, nutrition).invoke(
        request("Hello there"),
        force=AgentName.NUTRITION,
    )

    assert result.agent is AgentName.NUTRITION
    assert (fitness.calls, nutrition.calls) == (0, 1)


def test_forcing_an_agent_overrides_a_mixed_request():
    fitness = StubAgent(AgentName.FITNESS)
    nutrition = StubAgent(AgentName.NUTRITION)

    result = make_graph(fitness, nutrition).invoke(
        request("Create a workout and suggest meals for recovery"),
        force=AgentName.NUTRITION,
    )

    assert result.agent is AgentName.NUTRITION
    assert (fitness.calls, nutrition.calls) == (0, 1)


def test_forcing_fitness_ignores_nutrition_keywords():
    fitness = StubAgent(AgentName.FITNESS)
    nutrition = StubAgent(AgentName.NUTRITION)

    result = make_graph(fitness, nutrition).invoke(
        request("How much protein do I need?"),
        force=AgentName.FITNESS,
    )

    assert result.agent is AgentName.FITNESS
    assert (fitness.calls, nutrition.calls) == (1, 0)


def test_without_force_the_keyword_routing_is_unchanged():
    fitness = StubAgent(AgentName.FITNESS)
    nutrition = StubAgent(AgentName.NUTRITION)

    result = make_graph(fitness, nutrition).invoke(
        request("Hello there")
    )

    assert result.status is AgentStatus.NEEDS_CLARIFICATION
    assert (fitness.calls, nutrition.calls) == (0, 0)
