"""
PRIME_DIRECTIVE compliant schemas for deployment automation.

This module defines the machine-readable contracts for Requirements,
DeploymentPlan, ApplyResult, VerificationReport, and DeploymentReport
as specified in the PRIME_DIRECTIVE design principle.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PermissionScope(BaseModel):
    """A single permission requirement."""

    provider: str = Field(..., description="Provider name (e.g., 'railway', 'openai', 'aws')")
    service: str = Field(..., description="Service within provider (e.g., 'redis', 'api')")
    action: str = Field(..., description="Action or permission (e.g., 'connect', 'read')")
    scope: Optional[str] = Field(None, description="Resource scope or ARN")


class Dependency(BaseModel):
    """A dependency on another resource or service."""

    type: str = Field(..., description="Dependency type: 'service', 'network', 'quota'")
    needs: List[str] = Field(default_factory=list, description="List of requirements")
    name: Optional[str] = Field(None, description="Dependency name (for service dependencies)")
    port: Optional[int] = Field(None, description="Port number (for service dependencies)")


class Resources(BaseModel):
    """Resource requirements for the capability."""

    memory_mb: int = Field(512, description="Memory in megabytes")
    cpu_millicores: int = Field(500, description="CPU in millicores (1000 = 1 core)")
    storage_gb: Optional[int] = Field(None, description="Storage in gigabytes")


class Requirements(BaseModel):
    """Requirements document per PRIME_DIRECTIVE section 7.2."""

    capability: str = Field(..., description="Capability identifier")
    version: str = Field(..., description="Semantic version")
    permissions: List[PermissionScope] = Field(default_factory=list)
    dependencies: List[Dependency] = Field(default_factory=list)
    resources: Resources = Field(default_factory=Resources)
    data_residency: Optional[str] = Field(None, description="Required data region")
    estimated_cost_floor_usd: float = Field(0, description="Minimum estimated monthly cost")


class ValidationStatus(str, Enum):
    """Permission validation result status."""

    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"
    UNKNOWN = "unknown"


class ValidationResult(BaseModel):
    """Result of permission validation per PRIME_DIRECTIVE."""

    status: ValidationStatus
    validated_permissions: List[PermissionScope] = Field(default_factory=list)
    missing_permissions: List[PermissionScope] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DeploymentStep(BaseModel):
    """A single step in a deployment plan."""

    id: str = Field(..., description="Unique step identifier")
    action: str = Field(..., description="Action to perform")
    params: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    retriable: bool = Field(True, description="Whether step can be retried on failure")
    timeout_seconds: Optional[int] = Field(None, description="Step timeout")
    depends_on: List[str] = Field(default_factory=list, description="Step dependencies")


class RollbackStep(BaseModel):
    """A rollback action for a deployment step."""

    on_fail_of: str = Field(..., description="Step ID this rollback applies to")
    action: str = Field(..., description="Rollback action")
    params: Dict[str, Any] = Field(default_factory=dict)


class DeploymentPlan(BaseModel):
    """Deployment plan per PRIME_DIRECTIVE section 7.2.

    A deterministic, machine-readable plan generated in dry-run mode.
    Plans are content-addressed and time-bound.
    """

    target_env: str = Field(..., description="Target environment (dev/staging/production)")
    capability: str = Field(..., description="Capability being deployed")
    version: str = Field(..., description="Version being deployed")
    steps: List[DeploymentStep] = Field(default_factory=list)
    rollback: List[RollbackStep] = Field(default_factory=list)
    invariants: List[str] = Field(default_factory=list, description="Deployment invariants")
    estimated_duration_seconds: Optional[int] = Field(None)
    checksum: str = Field(..., description="Content hash for plan verification")

    def __init__(self, **data: Any) -> None:
        """Initialize and compute checksum if not provided."""
        if "checksum" not in data:
            # Compute checksum from plan content
            plan_content = {
                k: v
                for k, v in data.items()
                if k not in ("checksum", "estimated_duration_seconds")
            }
            content_str = json.dumps(plan_content, sort_keys=True)
            data["checksum"] = hashlib.sha256(content_str.encode()).hexdigest()
        super().__init__(**data)


class ApplyStatus(str, Enum):
    """Deployment apply status."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ApplyResult(BaseModel):
    """Result of applying a deployment plan."""

    status: ApplyStatus
    applied_steps: List[str] = Field(default_factory=list, description="Successfully applied step IDs")
    failed_step: Optional[str] = Field(None, description="Step ID that failed (if any)")
    error_message: Optional[str] = Field(None)
    rollback_executed: bool = Field(False)
    duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    artifacts: Dict[str, str] = Field(default_factory=dict, description="Deployment artifacts (URLs, IDs)")


class HealthCheckResult(BaseModel):
    """Result of a single health check."""

    name: str
    status: str = Field(..., description="'healthy', 'unhealthy', or 'degraded'")
    message: Optional[str] = None
    latency_ms: Optional[float] = None


class VerificationReport(BaseModel):
    """Verification report per PRIME_DIRECTIVE section 4.

    Post-deployment health checks and SLIs.
    """

    capability: str
    environment: str
    overall_status: str = Field(..., description="'healthy', 'unhealthy', or 'degraded'")
    health_checks: List[HealthCheckResult] = Field(default_factory=list)
    slis: Dict[str, float] = Field(default_factory=dict, description="Service Level Indicators")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    errors: List[str] = Field(default_factory=list)


class DeploymentReport(BaseModel):
    """Comprehensive deployment report per PRIME_DIRECTIVE section 7.2.

    Machine + human readable final report.
    """

    id: str = Field(..., description="Unique deployment ID")
    capability: str
    version: str
    environment: str
    result: ApplyStatus
    applied_steps: List[str] = Field(default_factory=list)
    duration_sec: float
    started_at: datetime
    completed_at: datetime
    links: Dict[str, str] = Field(
        default_factory=dict, description="Links to logs, runbooks, dashboards"
    )
    post_checks: Dict[str, float] = Field(default_factory=dict, description="Post-deployment metrics")
    verification: Optional[VerificationReport] = None
    errors: List[str] = Field(default_factory=list)


class RollbackResult(BaseModel):
    """Result of executing a rollback plan."""

    status: ApplyStatus
    rolled_back_steps: List[str] = Field(default_factory=list)
    failed_rollback_step: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

