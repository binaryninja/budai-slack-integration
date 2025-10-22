"""
Pydantic models for Slack Integration service payloads.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreateCallRequest(BaseModel):
    """Incoming payload to create a Slack call session."""

    channel_id: str
    user_id: str
    topic: Optional[str] = None
    agenda: Optional[str] = None
    external_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CallSession(BaseModel):
    """Stored representation of a Slack call session."""

    call_id: str
    external_id: str
    user_id: str
    channel_id: str
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    session_id: Optional[str] = None
    agenda: Optional[str] = None
    topic: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CallSessionResponse(BaseModel):
    """Response returned to clients when inspecting a call session."""

    call_id: str
    external_id: str
    user_id: str
    channel_id: str
    status: str
    created_at: datetime
    ended_at: Optional[datetime] = None
    session_id: Optional[str] = None
    agenda: Optional[str] = None
    topic: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EndCallRequest(BaseModel):
    """Request to end a Slack call session."""

    reason: Optional[str] = None
    post_summary: Optional[str] = None
    post_channel: Optional[str] = None


class MessageBlock(BaseModel):
    """Single Slack block payload."""

    type: str
    block_id: Optional[str] = None
    text: Optional[Dict[str, Any]] = None
    fields: Optional[List[Dict[str, Any]]] = None
    elements: Optional[List[Dict[str, Any]]] = None
    accessory: Optional[Dict[str, Any]] = None


class PostMessageRequest(BaseModel):
    """Request to post a Slack message."""

    channel: str
    text: Optional[str] = None
    thread_ts: Optional[str] = None
    blocks: Optional[List[MessageBlock]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AppHomePublishRequest(BaseModel):
    """Payload to update the Slack App Home view."""

    user_id: str
    blocks: List[MessageBlock]
    private_metadata: Optional[str] = None
