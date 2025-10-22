"""
Slack client abstraction for the Slack Integration service.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

logger = logging.getLogger(__name__)


class SlackClient:
    """Thin wrapper on top of slack_sdk.AsyncWebClient."""

    def __init__(self, bot_token: str) -> None:
        self._client = AsyncWebClient(token=bot_token)

    async def auth_test(self) -> Dict[str, Any]:
        return await self._client.auth_test()

    async def close(self) -> None:
        await self._client.close()

    async def create_call(
        self,
        *,
        external_id: str,
        join_url: str,
        title: str,
        users: List[str],
    ) -> Dict[str, Any]:
        """Create a Slack call via the Calls API."""
        try:
            response = await self._client.calls_add(
                external_unique_id=external_id,
                join_url=join_url,
                title=title,
                users=users,
            )
            return response["call"]
        except SlackApiError as exc:  # pragma: no cover - network path
            logger.error("Failed to create Slack call: %s", exc)
            raise

    async def end_call(self, *, external_id: str) -> None:
        """End a Slack call."""
        try:
            await self._client.calls_end(external_unique_id=external_id)
        except SlackApiError as exc:  # pragma: no cover - network path
            logger.error("Failed to end Slack call: %s", exc)
            raise

    async def post_message(
        self,
        *,
        channel: str,
        text: Optional[str],
        blocks: Optional[List[Dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Post a message to Slack."""
        payload: Dict[str, Any] = {
            "channel": channel,
            "text": text or "",
        }
        if blocks:
            payload["blocks"] = blocks
        if thread_ts:
            payload["thread_ts"] = thread_ts
        if metadata:
            payload["metadata"] = metadata

        try:
            response = await self._client.chat_postMessage(**payload)
            return response
        except SlackApiError as exc:  # pragma: no cover - network path
            logger.error("Failed to post Slack message: %s", exc)
            raise

    async def publish_app_home(
        self,
        *,
        user_id: str,
        blocks: List[Dict[str, Any]],
        private_metadata: Optional[str] = None,
    ) -> None:
        """Publish a Home tab view."""
        try:
            await self._client.views_publish(
                user_id=user_id,
                view={
                    "type": "home",
                    "blocks": blocks,
                    "private_metadata": private_metadata,
                },
            )
        except SlackApiError as exc:  # pragma: no cover - network path
            logger.error("Failed to publish App Home: %s", exc)
            raise
