"""Validated contracts shared by the fitness and nutrition agents.

These are in-process application contracts. They are deliberately independent
of LangGraph, MCP and any transport protocol.
"""
from datetime import date
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from state.profile import UserProfile


class AgentName(str, Enum):
    FITNESS = "fitness"
    NUTRITION = "nutrition"


class AgentStatus(str, Enum):
    OK = "ok"
    NEEDS_CLARIFICATION = "needs_clarification"
    REFUSED = "refused"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class EvidenceReference(BaseModel):
    """A source reference returned by a knowledge-backed agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1, max_length=300)
    page: int | None = Field(default=None, ge=1)
    chunk_number: int | None = Field(default=None, ge=0)
    quote: str | None = Field(default=None, max_length=1200)

    @field_validator("source")
    @classmethod
    def _strip_source(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source must not be blank")
        return value


class WorkoutExerciseContext(BaseModel):
    """A bounded, whitelisted exercise prescription from a saved programme."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=120)
    sets: int = Field(ge=1, le=20)
    rep_min: int = Field(ge=1, le=300)
    rep_max: int = Field(ge=1, le=300)
    unit: str = Field(default="reps", max_length=12)
    rest_seconds: int = Field(ge=0, le=900)


class WorkoutSessionContext(BaseModel):
    """A single scheduled session, including its exercise prescriptions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    day_number: int = Field(ge=1, le=7)
    session_number: int = Field(default=1, ge=1, le=3)
    weekday: str = Field(min_length=1, max_length=12)
    label: str = Field(min_length=1, max_length=60)
    muscles: list[str] = Field(default_factory=list, max_length=12)
    estimated_minutes: float = Field(gt=0, le=300)
    goal: str | None = Field(default=None, max_length=60)
    exercises: list[WorkoutExerciseContext] = Field(default_factory=list, max_length=8)


class WorkoutWeekContext(BaseModel):
    """One programme week, including its phase and scheduled sessions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int = Field(ge=1, le=4)
    phase: str = Field(min_length=1, max_length=30)
    effort: str = Field(min_length=1, max_length=100)
    sessions: list[WorkoutSessionContext] = Field(default_factory=list, max_length=7)


class TrainingContext(BaseModel):
    """Whitelisted facts from the user's saved workout plan, built by code and never by a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    program_goal: str | None = Field(default=None, max_length=60)
    program_level: str | None = Field(default=None, max_length=20)
    split: str | None = Field(default=None, max_length=30)
    sessions_per_week: int | None = Field(default=None, ge=1, le=7)
    training_weeks: list[WorkoutWeekContext] = Field(default_factory=list, max_length=4)
    today: Literal["training_day_planned", "training_day_done", "rest_day"]
    today_session: str | None = Field(default=None, max_length=60)
    today_minutes: int | None = Field(default=None, ge=1, le=300)
    next_session: str | None = Field(default=None, max_length=60)
    next_session_date: str | None = Field(default=None, pattern=r"^\\d{4}-\\d{2}-\\d{2}$")


class AgentRequest(BaseModel):
    """Input passed from the orchestrator to one domain agent."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=1000)
    profile: UserProfile | None = None
    today: date | None = None
    training_context: TrainingContext | None = None

    @field_validator("user_id", "message")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class AgentResult(BaseModel):
    """Normalized output returned by a domain agent to the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    agent: AgentName
    status: AgentStatus
    message: str = Field(min_length=1, max_length=4000)
    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=20)
    metadata: dict = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _strip_result_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value
