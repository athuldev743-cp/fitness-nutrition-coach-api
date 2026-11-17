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


def _text_list(values, *, limit: int, max_length: int = 40) -> list[str]:
    """Return only bounded, non-empty strings from a stored programme field."""
    if not isinstance(values, (list, tuple)):
        return []
    return [
        value.strip()[:max_length]
        for value in values
        if isinstance(value, str) and value.strip()
    ][:limit]


def _workout_weeks(program) -> list[dict]:
    """Project the saved Program model into a small, explicit nutrition-facing allowlist."""
    result = []
    for week in list(getattr(program, "weeks", []) or [])[:4]:
        sessions = []
        for session in list(getattr(week, "sessions", []) or [])[:7]:
            exercises = []
            for exercise in list(getattr(session, "exercises", []) or [])[:8]:
                name = getattr(exercise, "name", None)
                sets = getattr(exercise, "sets", None)
                rep_min = getattr(exercise, "rep_min", None)
                rep_max = getattr(exercise, "rep_max", None)
                aim_low = getattr(exercise, "aim_low", None)
                aim_high = getattr(exercise, "aim_high", None)
                rest_seconds = getattr(exercise, "rest_seconds", None)
                unit = getattr(exercise, "unit", "reps")
                if (
                    not isinstance(name, str) or not name.strip()
                    or not isinstance(sets, int) or not 1 <= sets <= 20
                    or not isinstance(rep_min, int) or not 1 <= rep_min <= 300
                    or not isinstance(rep_max, int) or not rep_min <= rep_max <= 300
                    or not isinstance(rest_seconds, int) or not 0 <= rest_seconds <= 900
                    or unit not in {"reps", "seconds"}
                ):
                    continue
                exercises.append({
                    "name": name.strip()[:120],
                    "sets": sets,
                    "rep_min": rep_min,
                    "rep_max": rep_max,
                    "aim_low": aim_low if isinstance(aim_low, int) else None,
                    "aim_high": aim_high if isinstance(aim_high, int) else None,
                    "unit": unit,
                    "rest_seconds": rest_seconds,
                })

            label = getattr(session, "label", None)
            weekday = getattr(session, "weekday", None)
            day_number = getattr(session, "day_number", None)
            session_number = getattr(session, "session_number", 1)
            minutes = getattr(session, "est_minutes", None)
            if (
                not isinstance(label, str) or not label.strip()
                or not isinstance(weekday, str) or not weekday.strip()
                or not isinstance(day_number, int) or not 1 <= day_number <= 7
                or not isinstance(session_number, int) or not 1 <= session_number <= 3
                or not isinstance(minutes, (int, float)) or not 0 < minutes <= 300
            ):
                continue

            sessions.append({
                "day_number": day_number,
                "session_number": session_number,
                "weekday": weekday.strip()[:12],
                "label": label.strip()[:60],
                "muscles": _text_list(getattr(session, "muscles", []), limit=12),
                "estimated_minutes": float(minutes),
                "goal": (
                    str(session.goal)[:60]
                    if isinstance(getattr(session, "goal", None), str) and session.goal.strip()
                    else None
                ),
                "exercises": exercises,
            })

        number = getattr(week, "number", None)
        phase = getattr(week, "phase", None)
        effort = getattr(week, "effort", None)
        if (
            isinstance(number, int) and 1 <= number <= 4
            and isinstance(phase, str) and phase.strip()
            and isinstance(effort, str) and effort.strip()
        ):
            result.append({
                "number": number,
                "phase": phase.strip()[:30],
                "effort": effort.strip()[:100],
                "sessions": sessions,
            })
    return result


def build_training_context(programs, today: date) -> TrainingContext | None:
    """Return saved workout facts for nutrition. Never raises: this context is optional."""
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
            program_goal=getattr(program, "goal", None),
            program_level=getattr(program, "level", None),
            split=getattr(program, "split", None),
            sessions_per_week=days if isinstance(days, int) and 1 <= days <= 7 else None,
            training_weeks=_workout_weeks(program),
            today="training_day_planned" if planned_today else "training_day_done" if done_today else "rest_day",
            today_session=today_session.label[:60] if today_session else None,
            today_minutes=minutes if minutes and 1 <= minutes <= 300 else None,
            next_session=next_session.label[:60] if next_session else None,
            next_session_date=upcoming["planned_date"] if upcoming else None,
        )
    except Exception as exc:
        logger.warning("training context unavailable (%s)", type(exc).__name__)
        return None
