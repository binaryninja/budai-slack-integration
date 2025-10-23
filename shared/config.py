"""
Configuration management for BudAI services.

Provides declarative configuration loading from environment variables,
YAML specs, and secrets management.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceResources(BaseModel):
    """Resource requirements for a service."""

    memory_mb: int = 512
    cpu_millicores: int = 500
    replicas: int = 1


class ServiceConfig(BaseModel):
    """Configuration for a single service."""

    name: Optional[str] = None
    enabled: bool = True
    resources: ServiceResources = Field(default_factory=ServiceResources)
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)


class DeploymentSpec(BaseModel):
    """Complete deployment specification.

    Loaded from YAML files in specs/ directory (development.yaml, staging.yaml, production.yaml).
    """

    environment: str = Field(..., description="Environment name: development, staging, or production")
    region: Optional[str] = Field(None, description="Cloud region")
    services: Dict[str, ServiceConfig] = Field(default_factory=dict)
    secrets: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Secret references: {secret_name: {source: ..., key: ...}}",
    )
    features: Dict[str, Any] = Field(default_factory=dict, description="Feature flags")

    @classmethod
    def from_file(cls, filepath: Path | str) -> DeploymentSpec:
        """Load deployment spec from YAML file.

        Args:
            filepath: Path to YAML spec file

        Returns:
            DeploymentSpec instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Deployment spec not found: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty or invalid YAML in {filepath}")

        return cls.model_validate(data)

    def to_file(self, filepath: Path | str) -> None:
        """Save deployment spec to YAML file.

        Args:
            filepath: Path to save YAML spec
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


class BaseServiceSettings(BaseSettings):
    """Base settings for all BudAI services.

    Provides common configuration that all services need:
    - Service identity
    - Redis connection
    - OpenAI API
    - Observability endpoints
    """

    model_config = SettingsConfigDict(
        env_prefix="BUDAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Service identity
    service_name: str = Field(..., description="Service name (e.g., 'api-gateway', 'agent-summarizer')")
    service_version: str = Field("1.0.0", description="Service version")
    environment: str = Field("development", description="Environment: development, staging, or production")

    # Redis (event bus and session storage)
    redis_url: str = Field("redis://localhost:6379/0", description="Redis connection URL")

    # OpenAI API
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_default_model: str = Field("gpt-4", description="Default OpenAI model")

    # Observability
    log_level: str = Field("INFO", description="Logging level")
    enable_tracing: bool = Field(True, description="Enable OpenTelemetry tracing")
    traces_endpoint: Optional[str] = Field(None, description="OTLP traces endpoint")
    metrics_endpoint: Optional[str] = Field(None, description="Prometheus metrics endpoint")

    # Health checks
    health_check_interval_seconds: int = Field(30, description="Health check interval")

    def load_secrets_from_railway(self) -> Dict[str, str]:
        """Load secrets from Railway environment variables.

        Railway injects secrets as environment variables. This method
        extracts them based on naming conventions.

        Returns:
            Dictionary of secret names to values
        """
        secrets = {}
        secret_prefix = f"{self.model_config.get('env_prefix', '')}SECRET_"

        for key, value in os.environ.items():
            if key.startswith(secret_prefix):
                secret_name = key[len(secret_prefix):].lower()
                secrets[secret_name] = value

        return secrets


class APIGatewaySettings(BaseServiceSettings):
    """Settings specific to API Gateway service."""

    service_name: str = "api-gateway"
    port: int = Field(8000, description="HTTP port")

    # Slack integration
    slack_signing_secret: str = Field(..., description="Slack request signing secret")
    slack_bot_token: str = Field(..., description="Slack bot OAuth token")

    # Upstream services
    orchestrator_url: str = Field("http://orchestrator:8001", description="Orchestrator service URL")
    voice_realtime_url: str = Field("http://voice-realtime:8005", description="Voice service URL")
    notion_agent_url: str = Field("http://agent-notion:8007", description="Notion agent service URL")


class OrchestratorSettings(BaseServiceSettings):
    """Settings specific to Orchestrator service."""

    service_name: str = "orchestrator"
    port: int = Field(8001, description="HTTP port")

    # Workflow configuration
    poll_interval_seconds: int = Field(30, description="Event polling interval")
    max_concurrent_workflows: int = Field(10, description="Max concurrent workflow executions")
    followup_delay_seconds: int = Field(900, description="Delay in seconds before invoking follow-up workflow")

    # Agent service URLs
    agent_summarizer_url: str = Field("http://agent-summarizer:8002")
    agent_followup_url: str = Field("http://agent-followup:8003")
    agent_communicator_url: str = Field("http://agent-communicator:8004")

    # Follow-up defaults
    followup_recipient_emails: Optional[str] = Field(
        None,
        description="Comma-separated list of default follow-up recipient emails",
    )
    followup_tone: str = Field("professional", description="Default tone for follow-up drafts")
    followup_include_slack: bool = Field(
        False,
        description="Whether to request Slack summary drafts by default",
    )
    followup_slack_channel: Optional[str] = Field(
        None,
        description="Default Slack channel for follow-up digests",
    )


class AgentServiceSettings(BaseServiceSettings):
    """Settings for agent services."""

    port: int = Field(8002, description="HTTP/gRPC port")

    # Agent-specific settings
    max_retries: int = Field(3, description="Max retries for agent invocation")
    timeout_seconds: int = Field(300, description="Agent execution timeout")
    reasoning_effort: str = Field("medium", description="Reasoning effort: low, medium, high")


class SlackIntegrationSettings(BaseServiceSettings):
    """Settings for the Slack Integration service."""

    service_name: str = "slack-integration"
    port: int = Field(8006, description="HTTP port")

    slack_signing_secret: str = Field(..., description="Slack signing secret for request verification")
    slack_bot_token: str = Field(..., description="Slack bot token for API calls")
    default_join_url: str = Field(
        "https://example.com/join",
        description="Fallback join URL for generated Slack calls",
    )
    voice_realtime_url: str = Field(
        "http://voice-realtime:8005",
        description="Base URL for the voice realtime service",
    )


def load_deployment_spec(environment: str, specs_dir: Path | str = "specs") -> DeploymentSpec:
    """Load deployment specification for an environment.

    Args:
        environment: Environment name (development, staging, production)
        specs_dir: Directory containing spec files

    Returns:
        DeploymentSpec for the environment

    Raises:
        FileNotFoundError: If spec file doesn't exist
    """
    specs_path = Path(specs_dir)
    spec_file = specs_path / f"{environment}.yaml"

    return DeploymentSpec.from_file(spec_file)


def create_service_settings(service_name: str) -> BaseServiceSettings:
    """Create appropriate settings instance for a service.

    Args:
        service_name: Name of the service

    Returns:
        Service-specific settings instance
    """
    settings_map = {
        "api-gateway": APIGatewaySettings,
        "orchestrator": OrchestratorSettings,
        "agent-summarizer": AgentServiceSettings,
        "agent-followup": AgentServiceSettings,
        "agent-communicator": AgentServiceSettings,
        "slack-integration": SlackIntegrationSettings,
    }

    settings_class = settings_map.get(service_name, BaseServiceSettings)
    return settings_class()
