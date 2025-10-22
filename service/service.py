"""\nSlack Integration HTTP service.\n\nBridges BudAI workflows with Slack APIs for call management and messaging.\n"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    async def create_call(self, payload: CreateCallRequest) -> CallSessionResponse:
        call_id = f"call-{payload.external_id or payload.channel_id}-{int(time.time())}"
        session = CallSession(
            call_id=call_id,
            external_id=payload.external_id or call_id,
            user_id=payload.user_id,
            channel_id=payload.channel_id,
            agenda=payload.agenda,
            topic=payload.topic,
            metadata=payload.metadata,
        )
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
            "slack.call.created",
            {
                "call_id": session.call_id,
                "channel_id": session.channel_id,
                "user_id": session.user_id,
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
