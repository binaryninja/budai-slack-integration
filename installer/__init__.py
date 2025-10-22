"""
Installer framework for PRIME_DIRECTIVE compliant self-deployment.

This package provides the base classes and providers for building
self-deploying capabilities that can explain, plan, apply, verify,
and rollback their own deployments.
"""

from .base import Installer
from .railway import RailwayAPIError, RailwayProvider
from .schemas import (
    ApplyResult,
    ApplyStatus,
    Dependency,
    DeploymentPlan,
    DeploymentReport,
    DeploymentStep,
    HealthCheckResult,
    PermissionScope,
    Requirements,
    Resources,
    RollbackResult,
    RollbackStep,
    ValidationResult,
    ValidationStatus,
    VerificationReport,
)

__all__ = [
    "Installer",
    "RailwayProvider",
    "RailwayAPIError",
    "ApplyResult",
    "ApplyStatus",
    "Dependency",
    "DeploymentPlan",
    "DeploymentReport",
    "DeploymentStep",
    "HealthCheckResult",
    "PermissionScope",
    "Requirements",
    "Resources",
    "RollbackResult",
    "RollbackStep",
    "ValidationResult",
    "ValidationStatus",
    "VerificationReport",
]

