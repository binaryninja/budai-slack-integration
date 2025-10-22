"""
Shared libraries for BudAI services.

Provides common functionality for all services:
- Event bus for inter-service communication
- Configuration management
- Health checks
- Observability (tracing, metrics, logging)
"""

from .config import (
    AgentServiceSettings,
    APIGatewaySettings,
    BaseServiceSettings,
    DeploymentSpec,
    OrchestratorSettings,
    SlackIntegrationSettings,
    ServiceConfig,
    ServiceResources,
    create_service_settings,
    load_deployment_spec,
)
from .events import (
    EVENT_TYPE_REGISTRY,
    AgentCompletedEvent,
    AgentInvokedEvent,
    BaseEvent,
    DeploymentCompletedEvent,
    DeploymentStartedEvent,
    EventBus,
    EventType,
    FollowupRequiredEvent,
    FollowupSentEvent,
    MeetingCompletedEvent,
    MeetingScheduledEvent,
    SummaryGeneratedEvent,
    VoiceCallEndedEvent,
    VoiceCallStartedEvent,
    create_event_bus,
)
from .health import (
    HealthCheck,
    HealthChecker,
    HealthStatus,
    ServiceHealth,
    check_http_endpoint,
    check_openai_api,
    check_redis_connection,
    create_liveness_check,
    create_readiness_check,
)
from .observability import (
    ServiceObservability,
    Span,
    emit_metric,
    get_observability,
    init_observability,
    log_event,
    trace_operation,
)

__all__ = [
    # Config
    "AgentServiceSettings",
    "APIGatewaySettings",
    "BaseServiceSettings",
    "DeploymentSpec",
    "OrchestratorSettings",
    "SlackIntegrationSettings",
    "ServiceConfig",
    "ServiceResources",
    "create_service_settings",
    "load_deployment_spec",
    # Events
    "EVENT_TYPE_REGISTRY",
    "AgentCompletedEvent",
    "AgentInvokedEvent",
    "BaseEvent",
    "DeploymentCompletedEvent",
    "DeploymentStartedEvent",
    "EventBus",
    "EventType",
    "FollowupRequiredEvent",
    "FollowupSentEvent",
    "MeetingCompletedEvent",
    "MeetingScheduledEvent",
    "SummaryGeneratedEvent",
    "VoiceCallEndedEvent",
    "VoiceCallStartedEvent",
    "create_event_bus",
    # Health
    "HealthCheck",
    "HealthChecker",
    "HealthStatus",
    "ServiceHealth",
    "check_http_endpoint",
    "check_openai_api",
    "check_redis_connection",
    "create_liveness_check",
    "create_readiness_check",
    # Observability
    "ServiceObservability",
    "Span",
    "emit_metric",
    "get_observability",
    "init_observability",
    "log_event",
    "trace_operation",
]

