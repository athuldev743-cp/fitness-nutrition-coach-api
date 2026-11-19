"""Storage for one user's current diet plan (the latest generated plan replaces the previous one)."""
from __future__ import annotations

import json
from datetime import datetime

from state.db import LOCAL_USER_ID, check_user_id, connect


class DietPlanStore:
    def __init__(self, path=":memory:", read_only=False, user_id=LOCAL_USER_ID):
        self.user_id = check_user_id(user_id)
        self.conn = connect(path, read_only=read_only)

    def get(self) -> dict | None:
        row = self.conn.execute(
            "SELECT created_at, fingerprint, data FROM diet_plans WHERE user_id = ?", (self.user_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row["data"])
        except (TypeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return {"created_at": row["created_at"], "fingerprint": row["fingerprint"], **data}

    def save(self, data: dict, fingerprint: str) -> str:
        """Replace this user's plan. `data` holds only the plan fields (text, evidence, inputs)."""
        created = datetime.now().isoformat(timespec="seconds")
        self.conn.execute(
            """INSERT OR REPLACE INTO diet_plans (user_id, created_at, fingerprint, data)
               VALUES (?, ?, ?, ?)""",
            (self.user_id, created, fingerprint, json.dumps(data, ensure_ascii=False)),
        )
        self.conn.commit()
        return created

    def close(self) -> None:
        self.conn.close()
