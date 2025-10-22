"""
Event bus contracts for inter-service communication.

Defines event schemas for asynchronous communication between services
using Redis Streams as the transport layer.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Standard event types in the system."""

    MEETING_SCHEDULED = "meeting.scheduled"
    MEETING_STARTED = "meeting.started"
    MEETING_COMPLETED = "meeting.completed"
    SUMMARY_GENERATED = "summary.generated"
    FOLLOWUP_REQUIRED = "followup.required"
    FOLLOWUP_SENT = "followup.sent"
    VOICE_CALL_STARTED = "voice.call.started"
    VOICE_CALL_ENDED = "voice.call.ended"
    AGENT_INVOKED = "agent.invoked"
    AGENT_COMPLETED = "agent.completed"
    DEPLOYMENT_STARTED = "deployment.started"
    DEPLOYMENT_COMPLETED = "deployment.completed"


class BaseEvent(BaseModel):
    """Base event model with common fields."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(..., description="Event type identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = Field(None, description="ID to correlate related events")
    source_service: Optional[str] = Field(None, description="Service that emitted the event")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize event to JSON."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> BaseEvent:
        """Deserialize event from JSON."""
        return cls.model_validate_json(data)


class MeetingScheduledEvent(BaseEvent):
    """Event emitted when a meeting is scheduled."""

    event_type: str = EventType.MEETING_SCHEDULED
    meeting_id: str
    title: str
    starts_at: datetime
    ends_at: datetime
    attendees: List[str] = Field(default_factory=list)
    conference_link: Optional[str] = None
    calendar_event_id: Optional[str] = None


class MeetingCompletedEvent(BaseEvent):
    """Event emitted when a meeting ends."""

    event_type: str = EventType.MEETING_COMPLETED
    meeting_id: str
    title: str
    transcript_url: Optional[str] = None
    fireflies_meeting_id: Optional[str] = None
    attendees: List[str] = Field(default_factory=list)
    duration_minutes: Optional[int] = None


class SummaryGeneratedEvent(BaseEvent):
    """Event emitted when a meeting summary is generated."""

    event_type: str = EventType.SUMMARY_GENERATED
    meeting_id: str
    summary: str
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    summary_metadata: Dict[str, Any] = Field(default_factory=dict)


class FollowupRequiredEvent(BaseEvent):
    """Event emitted when follow-up action is needed."""

    event_type: str = EventType.FOLLOWUP_REQUIRED
    meeting_id: str
    followup_type: str = Field(..., description="'email' or 'slack'")
    summary: Dict[str, Any]
    recipients: List[str] = Field(default_factory=list)


class FollowupSentEvent(BaseEvent):
    """Event emitted when follow-up is sent."""

    event_type: str = EventType.FOLLOWUP_SENT
    meeting_id: str
    followup_type: str
    message_id: Optional[str] = None
    recipients: List[str] = Field(default_factory=list)
    status: str = Field("sent", description="'sent', 'failed', 'pending'")


class VoiceCallStartedEvent(BaseEvent):
    """Event emitted when a voice call starts."""

    event_type: str = EventType.VOICE_CALL_STARTED
    call_id: str
    slack_user_id: str
    channel_id: str
    session_id: str


class VoiceCallEndedEvent(BaseEvent):
    """Event emitted when a voice call ends."""

    event_type: str = EventType.VOICE_CALL_ENDED
    call_id: str
    session_id: str
    duration_seconds: float
    transcript: Optional[str] = None


class AgentInvokedEvent(BaseEvent):
    """Event emitted when an agent is invoked."""

    event_type: str = EventType.AGENT_INVOKED
    agent_name: str
    task_id: str
    input_data: Dict[str, Any] = Field(default_factory=dict)


class AgentCompletedEvent(BaseEvent):
    """Event emitted when an agent completes execution."""

    event_type: str = EventType.AGENT_COMPLETED
    agent_name: str
    task_id: str
    output_data: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field("success", description="'success' or 'failed'")
    error_message: Optional[str] = None
    duration_seconds: float


class DeploymentStartedEvent(BaseEvent):
    """Event emitted when a deployment starts."""

    event_type: str = EventType.DEPLOYMENT_STARTED
    deployment_id: str
    service_name: str
    environment: str
    version: str


class DeploymentCompletedEvent(BaseEvent):
    """Event emitted when a deployment completes."""

    event_type: str = EventType.DEPLOYMENT_COMPLETED
    deployment_id: str
    service_name: str
    environment: str
    version: str
    status: str = Field("success", description="'success' or 'failed'")
    duration_seconds: float


# Event type registry for deserialization
EVENT_TYPE_REGISTRY: Dict[str, type[BaseEvent]] = {
    EventType.MEETING_SCHEDULED: MeetingScheduledEvent,
    EventType.MEETING_COMPLETED: MeetingCompletedEvent,
    EventType.SUMMARY_GENERATED: SummaryGeneratedEvent,
    EventType.FOLLOWUP_REQUIRED: FollowupRequiredEvent,
    EventType.FOLLOWUP_SENT: FollowupSentEvent,
    EventType.VOICE_CALL_STARTED: VoiceCallStartedEvent,
    EventType.VOICE_CALL_ENDED: VoiceCallEndedEvent,
    EventType.AGENT_INVOKED: AgentInvokedEvent,
    EventType.AGENT_COMPLETED: AgentCompletedEvent,
    EventType.DEPLOYMENT_STARTED: DeploymentStartedEvent,
    EventType.DEPLOYMENT_COMPLETED: DeploymentCompletedEvent,
}


class EventBus:
    """Simple event bus abstraction over Redis Streams.

    Provides publish/subscribe pattern for inter-service communication.
    Services can publish events and subscribe to event types they care about.
    """

    def __init__(self, redis_client: Any) -> None:
        """Initialize event bus with Redis client.

        Args:
            redis_client: Redis client (can be redis-py or aioredis)
        """
        self.redis = redis_client
        self.stream_name = "budai:events"
        self.consumer_group = "budai-services"
        self._handlers: Dict[str, List[Callable[[BaseEvent], None]]] = {}

    async def publish(self, event: BaseEvent) -> str:
        """Publish an event to the bus.

        Args:
            event: Event to publish

        Returns:
            Event ID (Redis stream message ID)
        """
        event_data = event.model_dump()
        event_json = json.dumps(event_data)

        # Add to Redis stream
        message_id = await self.redis.xadd(
            self.stream_name,
            {
                "event_type": event.event_type,
                "data": event_json,
                "correlation_id": event.correlation_id or "",
            },
        )

        logger.debug(
            "Published event %s (type=%s, correlation_id=%s)",
            event.event_id,
            event.event_type,
            event.correlation_id,
        )
        return str(message_id)

    def subscribe(self, event_type: str, handler: Callable[[BaseEvent], None]) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to (e.g., EventType.MEETING_COMPLETED)
            handler: Callback function to handle events
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.info("Subscribed handler to event type: %s", event_type)

    async def start_consuming(
        self, consumer_name: str, block_ms: int = 1000
    ) -> None:
        """Start consuming events from the stream.

        Args:
            consumer_name: Unique consumer name for this service instance
            block_ms: Milliseconds to block waiting for new events
        """
        # Create consumer group if it doesn't exist
        try:
            await self.redis.xgroup_create(
                self.stream_name, self.consumer_group, id="0", mkstream=True
            )
            logger.info("Created consumer group: %s", self.consumer_group)
        except Exception:
            # Group likely already exists
            pass

        logger.info("Starting event consumer: %s", consumer_name)

        while True:
            try:
                # Read new messages from stream
                messages = await self.redis.xreadgroup(
                    self.consumer_group,
                    consumer_name,
                    {self.stream_name: ">"},
                    count=10,
                    block=block_ms,
                )

                if not messages:
                    continue

                for stream, stream_messages in messages:
                    for message_id, message_data in stream_messages:
                        await self._process_message(message_id, message_data)
                        # Acknowledge message
                        await self.redis.xack(self.stream_name, self.consumer_group, message_id)

            except Exception as exc:
                logger.exception("Error consuming events: %s", exc)

    async def _process_message(self, message_id: bytes, message_data: Dict[bytes, bytes]) -> None:
        """Process a single event message.

        Args:
            message_id: Redis stream message ID
            message_data: Message data from stream
        """
        try:
            event_type = message_data.get(b"event_type", b"").decode("utf-8")
            event_json = message_data.get(b"data", b"").decode("utf-8")

            if not event_type or not event_json:
                logger.warning("Malformed event message: %s", message_id)
                return

            # Deserialize event
            event_class = EVENT_TYPE_REGISTRY.get(event_type, BaseEvent)
            event = event_class.model_validate_json(event_json)

            # Call registered handlers
            handlers = self._handlers.get(event_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as exc:
                    logger.exception("Error in event handler for %s: %s", event_type, exc)

        except Exception as exc:
            logger.exception("Error processing message %s: %s", message_id, exc)


# For backwards compatibility with synchronous code
import asyncio


def create_event_bus(redis_url: str) -> EventBus:
    """Create an event bus instance.

    Args:
        redis_url: Redis connection URL

    Returns:
        EventBus instance
    """
    import redis.asyncio as redis

    redis_client = redis.from_url(redis_url, decode_responses=False)
    return EventBus(redis_client)

