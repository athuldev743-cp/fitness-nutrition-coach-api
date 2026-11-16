"""Whitelisted facts from the saved workout plan for the nutrition agent.

Built by plain code from the stored plan. No model output and no free text written by the
fitness agent is passed along, and the nutrition agent may not cite it as evidence.
"""
from __future__ import annotations

import logging
from datetime import date

from agent.contracts import TrainingContext

logger = logging.getLogger(__name__)


def _details(program) -> dict:
    return {
        (week.number, session.day_number, session.session_number): session
        for week in program.weeks
        for session in week.sessions
    }


def build_training_context(programs, today: date) -> TrainingContext | None:
    """Return today's training facts, or None. Never raises: this context is optional."""
    try:
        active = programs.active_program()
        if not active:
            return None
        program = active["program"]
        details = _details(program)
        rows = programs.sessions(active["id"])
        iso = today.isoformat()

        def session_of(row):
            return details.get((row["week"], row["day_number"], row["session_number"]))

        planned_today = next((r for r in rows if r["planned_date"] == iso and r["status"] == "planned"), None)
        done_today = next((r for r in rows if r["planned_date"] == iso and r["status"] == "done"), None)
        upcoming = next((r for r in rows if r["status"] == "planned" and r["planned_date"] > iso), None)

        today_row = planned_today or done_today
        today_session = session_of(today_row) if today_row else None
        next_session = session_of(upcoming) if upcoming else None
        days = getattr(program, "days_per_week", None)
        minutes = int(round(today_session.est_minutes)) if today_session else None

        return TrainingContext(
            sessions_per_week=days if isinstance(days, int) and 1 <= days <= 7 else None,
            today="training_day_planned" if planned_today else "training_day_done" if done_today else "rest_day",
            today_session=today_session.label[:60] if today_session else None,
            today_minutes=minutes if minutes and 1 <= minutes <= 300 else None,
            next_session=next_session.label[:60] if next_session else None,
            next_session_date=upcoming["planned_date"] if upcoming else None,
        )
    except Exception as exc:
        logger.warning("training context unavailable (%s)", type(exc).__name__)
        return None
