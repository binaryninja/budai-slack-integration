"""
Redis-backed session storage for Slack call sessions.
"""

from __future__ import annotations

from typing import Dict, Optional

from redis.asyncio import Redis

from .models import CallSession


class CallSessionStore:
    """Persist Slack call sessions in Redis."""

    def __init__(self, redis: Redis, namespace: str = "budai:slack:calls") -> None:
        self._redis = redis
        self._namespace = namespace

    def _key(self, call_id: str) -> str:
        return f"{self._namespace}:{call_id}"

    async def save(self, session: CallSession) -> None:
        await self._redis.set(self._key(session.call_id), session.model_dump_json())

    async def get(self, call_id: str) -> Optional[CallSession]:
        data = await self._redis.get(self._key(call_id))
        if not data:
            return None
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return CallSession.model_validate_json(data)

    async def delete(self, call_id: str) -> None:
        await self._redis.delete(self._key(call_id))

    async def update(self, call_id: str, updates: Dict[str, object]) -> Optional[CallSession]:
        session = await self.get(call_id)
        if not session:
            return None
        session_data = session.model_dump()
        session_data.update(updates)
        updated = CallSession.model_validate(session_data)
        await self.save(updated)
        return updated
