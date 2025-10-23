"""
Slack Integration HTTP service.

Bridges BudAI workflows with Slack APIs for call management and messaging, and
coordinates voice session setup with the Voice Realtime service.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from shared import (
    HealthChecker,
    ServiceObservability,
    SlackIntegrationSettings,
    check_redis_connection,
    create_event_bus,
    init_observability,
    log_event,
)

from .models import (
    AppHomePublishRequest,
    CallSession,
    CallSessionResponse,
    CreateCallRequest,
    EndCallRequest,
    PostMessageRequest,
)
from .session_store import CallSessionStore
from .slack_client import SlackClient

logger = logging.getLogger(__name__)


def verify_slack_signature(
    *,
    signing_secret: str,
    timestamp: str | None,
    signature: str | None,
    body: bytes,
) -> bool:
    """Validate Slack request signature."""
    if not signing_secret or not timestamp or not signature:
        return False

    try:
        ts = int(timestamp)
    except ValueError:
        return False

    if abs(time.time() - ts) > 60 * 5:
        return False

    basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    digest = hmac.new(
        signing_secret.encode("utf-8"),
        basestring.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    expected = f"v0={digest}"
    return hmac.compare_digest(expected, signature)


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 timestamps (with or without Z suffix)."""
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        logger.debug("Failed to parse ISO timestamp: %s", value)
        return None


def _build_websocket_url(base_url: str, endpoint: Optional[str]) -> Optional[str]:
    """Build a websocket URL from the voice realtime base and relative endpoint."""
    if not endpoint:
        return None

    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc
    base_path = parsed.path.rstrip("/")
    endpoint_path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"{scheme}://{netloc}{base_path}{endpoint_path}"


class SlackIntegrationService:
    """Service encapsulating Slack operations."""

    def __init__(self) -> None:
        self.settings = SlackIntegrationSettings()
        self.observability: ServiceObservability = init_observability(
            "slack-integration", self.settings.service_version
        )
        self.health_checker = HealthChecker("slack-integration", self.settings.service_version)

        self.event_bus = create_event_bus(self.settings.redis_url)
        self.session_store = CallSessionStore(self.event_bus.redis)
        self.slack_client = SlackClient(self.settings.slack_bot_token)
        self.voice_client = httpx.AsyncClient(
            base_url=self.settings.voice_realtime_url.rstrip("/"),
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(10.0, connect=5.0),
        )

        self._register_health_checks()

    def _register_health_checks(self) -> None:
        self.health_checker.register_check("liveness", lambda: (True, "Service running"))

        async def redis_check() -> tuple[bool, str]:
            try:
                return await check_redis_connection(self.event_bus.redis)
            except Exception as exc:  # pragma: no cover
                return False, f"Redis check failed: {exc}"

        self.health_checker.register_check("redis", redis_check)

        async def slack_check() -> tuple[bool, str]:
            try:
                await self.slack_client.auth_test()
                return True, "Slack API reachable"
            except Exception as exc:  # pragma: no cover
                return False, f"Slack check failed: {exc}"

        self.health_checker.register_check("slack_api", slack_check)

    async def initialize(self) -> None:
        logger.info("Initializing Slack integration service")

    async def shutdown(self) -> None:
        logger.info("Shutting down Slack integration service")
        await self.slack_client.close()
        await self.voice_client.aclose()

    async def _fetch_voice_session(self, session: CallSession, team_id: Optional[str]) -> CallSession:
        """Create a realtime voice session for the call and attach credentials."""
        metadata = {
            **session.metadata,
            "topic": session.topic,
            "agenda": session.agenda,
            "team_id": team_id,
        }
        session.metadata = metadata

        payload: Dict[str, Any] = {
            "call_id": session.call_id,
            "slack_user_id": session.user_id,
            "channel_id": session.channel_id,
            "external_id": session.external_id,
            "metadata": metadata,
        }

        try:
            response = await self.voice_client.post("/session-token", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Voice realtime session creation failed: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="Voice realtime service unavailable",
            ) from exc

        data = response.json()
        session.session_id = data.get("session_id")
        session.voice_openai_session_id = data.get("openai_session_id")
        session.voice_client_secret = data.get("client_secret")
        session.voice_expires_at = _parse_iso8601(data.get("expires_at"))
        endpoint = data.get("websocket_endpoint")
        session.voice_websocket_url = _build_websocket_url(
            self.settings.voice_realtime_url,
            endpoint,
        )

        voice_meta = session.metadata.setdefault("voice", {})
        voice_meta.update(
            {
                "websocket_endpoint": endpoint,
                "client_secret": session.voice_client_secret,
                "openai_session_id": session.voice_openai_session_id,
                "expires_at": data.get("expires_at"),
            }
        )

        return session

    async def create_call(self, payload: CreateCallRequest) -> CallSessionResponse:
        call_id = f"call-{payload.external_id or payload.channel_id}-{int(time.time())}"
        session = CallSession(
            call_id=call_id,
            external_id=payload.external_id or call_id,
            user_id=payload.user_id,
            channel_id=payload.channel_id,
            agenda=payload.agenda,
            topic=payload.topic,
            metadata=dict(payload.metadata),
        )
        if payload.title and "title" not in session.metadata:
            session.metadata["title"] = payload.title
        session = await self._fetch_voice_session(session, payload.team_id)
        await self.session_store.save(session)

        join_url = session.metadata.get("join_url") or self.settings.default_join_url
        try:
            await self.slack_client.create_call(
                external_id=session.external_id,
                join_url=join_url,
                title=session.topic or session.metadata.get("title") or "BudAI Call",
                users=[session.user_id],
            )
        except Exception as exc:  # pragma: no cover - Slack API failure
            logger.warning("Slack call creation failed: %s", exc)

        log_event(
            "info",
            "Slack call created",
            context={
                "call_id": session.call_id,
                "channel_id": session.channel_id,
                "user_id": session.user_id,
                "voice_session_id": session.session_id,
            },
        )

        return CallSessionResponse.model_validate(session.model_dump())

    async def get_call(self, call_id: str) -> CallSessionResponse:
        session = await self.session_store.get(call_id)
        if not session:
            raise HTTPException(status_code=404, detail="Call not found")
        return CallSessionResponse.model_validate(session.model_dump())

    async def end_call(self, call_id: str, payload: EndCallRequest) -> CallSessionResponse:
        session = await self.session_store.update(
            call_id,
            {
                "status": "ended",
                "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        if not session:
            raise HTTPException(status_code=404, detail="Call not found")

        try:
            await self.slack_client.end_call(external_id=session.external_id)
        except Exception as exc:  # pragma: no cover
            logger.warning("Slack call end failed: %s", exc)

        if payload.post_summary and payload.post_channel:
            await self.slack_client.post_message(
                channel=payload.post_channel,
                text=payload.post_summary,
                blocks=None,
            )

        return CallSessionResponse.model_validate(session.model_dump())

    async def post_message(self, payload: PostMessageRequest) -> Dict[str, Any]:
        response = await self.slack_client.post_message(
            channel=payload.channel,
            text=payload.text,
            blocks=[block.model_dump(exclude_none=True) for block in payload.blocks] if payload.blocks else None,
            thread_ts=payload.thread_ts,
            metadata=payload.metadata,
        )
        return response

    async def publish_app_home(self, payload: AppHomePublishRequest) -> None:
        await self.slack_client.publish_app_home(
            user_id=payload.user_id,
            blocks=[block.model_dump(exclude_none=True) for block in payload.blocks],
            private_metadata=payload.private_metadata,
        )


service = SlackIntegrationService()
app = FastAPI(title="BudAI Slack Integration", version="1.0.0")


@app.on_event("startup")
async def on_startup() -> None:
    await service.initialize()
    logger.info("Slack integration service started")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await service.shutdown()


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": "BudAI Slack Integration",
            "version": "1.0.0",
            "status": "running",
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    report = await service.health_checker.check_health()
    liveness = next((c for c in report.checks if c.name == "liveness"), None)
    status_code = 200 if liveness and liveness.status == "healthy" else 503
    return JSONResponse(report.model_dump(), status_code=status_code)


async def _validate_slack_request(request: Request) -> None:
    if not service.settings.slack_signing_secret:
        return

    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    signature = request.headers.get("X-Slack-Signature")
    body = await request.body()
    if not verify_slack_signature(
        signing_secret=service.settings.slack_signing_secret,
        timestamp=timestamp,
        signature=signature,
        body=body,
    ):
        raise HTTPException(status_code=401, detail="Invalid Slack signature")


@app.post("/calls/create")
async def create_call_endpoint(request: Request) -> JSONResponse:
    await _validate_slack_request(request)
    payload_data = await request.json()
    try:
        payload = CreateCallRequest.model_validate(payload_data)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    session = await service.create_call(payload)
    return JSONResponse(session.model_dump())


@app.get("/calls/{call_id}")
async def get_call_endpoint(call_id: str) -> JSONResponse:
    session = await service.get_call(call_id)
    return JSONResponse(session.model_dump())


@app.post("/calls/{call_id}/end")
async def end_call_endpoint(call_id: str, payload: EndCallRequest) -> JSONResponse:
    session = await service.end_call(call_id, payload)
    return JSONResponse(session.model_dump())


@app.post("/messages")
async def post_message_endpoint(request: Request) -> JSONResponse:
    await _validate_slack_request(request)
    payload_data = await request.json()
    try:
        payload = PostMessageRequest.model_validate(payload_data)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    response = await service.post_message(payload)
    return JSONResponse(response)


@app.post("/app-home")
async def publish_app_home_endpoint(request: Request) -> JSONResponse:
    await _validate_slack_request(request)
    payload_data = await request.json()
    try:
        payload = AppHomePublishRequest.model_validate(payload_data)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    await service.publish_app_home(payload)
    return JSONResponse({"status": "ok"})


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8006"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
