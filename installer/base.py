"""
Base Installer class implementing PRIME_DIRECTIVE installer contract.

Every capability MUST ship with a built-in installer that implements
the Explain → Plan → Apply → Verify → Report → Rollback lifecycle.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from .schemas import (
    ApplyResult,
    ApplyStatus,
    DeploymentPlan,
    DeploymentReport,
    Requirements,
    RollbackResult,
    ValidationResult,
    VerificationReport,
)

logger = logging.getLogger(__name__)


class Installer(ABC):
    """Abstract base class for self-deploying capabilities.

    Implements PRIME_DIRECTIVE section 7.1 Installer API contract:
    - describe_requirements: Explain what's needed
    - validate_permissions: Self-test permissions
    - plan: Generate deterministic deployment plan (dry-run)
    - apply: Execute plan transactionally and idempotently
    - verify: Run health checks and validate deployment
    - rollback: Revert changes if needed
    - report: Generate machine + human readable report
    """

    def __init__(self, capability_name: str, version: str) -> None:
        """Initialize installer with capability identity."""
        self.capability_name = capability_name
        self.version = version
        self.logger = logging.getLogger(f"{__name__}.{capability_name}")
        self._last_apply_result: Optional[ApplyResult] = None
        self._last_verification: Optional[VerificationReport] = None
        self._deployment_start: Optional[datetime] = None

    @abstractmethod
    def describe_requirements(self, env: str) -> Requirements:
        """Describe requirements for deploying this capability.

        MUST return explicit, machine-readable requirements per
        PRIME_DIRECTIVE section 4 (tenet 1).

        Args:
            env: Target environment ('development', 'staging', 'production')

        Returns:
            Requirements document with permissions, dependencies, resources
        """
        pass

    @abstractmethod
    def validate_permissions(self, creds: Dict[str, Any], env: str) -> ValidationResult:
        """Validate that provided credentials have required permissions.

        MUST self-test permissions with non-mutating probes per
        PRIME_DIRECTIVE section 4 (tenet 3).

        Args:
            creds: Credentials dictionary (API keys, tokens, etc.)
            env: Target environment

        Returns:
            ValidationResult indicating which permissions are valid/missing
        """
        pass

    @abstractmethod
    def plan(self, spec: Dict[str, Any], env: str) -> DeploymentPlan:
        """Generate deterministic deployment plan (dry-run only).

        MUST be deterministic and idempotent per PRIME_DIRECTIVE section 4 (tenet 2).
        Plans are content-addressed and time-bound.

        Args:
            spec: Deployment specification (configuration, replicas, etc.)
            env: Target environment

        Returns:
            DeploymentPlan with steps, rollback actions, and invariants
        """
        pass

    @abstractmethod
    def apply(self, plan: DeploymentPlan, creds: Dict[str, Any]) -> ApplyResult:
        """Apply the deployment plan transactionally and idempotently.

        MUST execute changes transactionally per PRIME_DIRECTIVE section 4 (tenet 2).
        Should handle retries, circuit breakers, and partial failures.

        Args:
            plan: Validated deployment plan
            creds: Credentials for making changes

        Returns:
            ApplyResult with status, applied steps, and artifacts
        """
        pass

    @abstractmethod
    def verify(self, env: str) -> VerificationReport:
        """Verify deployment with health checks and SLIs.

        MUST validate the deployment is functional per PRIME_DIRECTIVE section 4 (tenet 2d).

        Args:
            env: Environment to verify

        Returns:
            VerificationReport with health checks and metrics
        """
        pass

    @abstractmethod
    def rollback(self, plan: DeploymentPlan, creds: Dict[str, Any]) -> RollbackResult:
        """Execute rollback plan to revert changes.

        MUST be able to safely rollback per PRIME_DIRECTIVE section 4 (tenet 5).

        Args:
            plan: Original deployment plan (contains rollback steps)
            creds: Credentials for making changes

        Returns:
            RollbackResult indicating rollback success/failure
        """
        pass

    def report(self) -> DeploymentReport:
        """Generate comprehensive deployment report.

        MUST provide machine + human readable report per
        PRIME_DIRECTIVE section 4 (tenet 4).

        Returns:
            DeploymentReport with all deployment metadata
        """
        if self._last_apply_result is None:
            raise RuntimeError("No deployment has been applied yet. Call apply() first.")

        deployment_id = f"deploy-{self.capability_name}-{datetime.utcnow().isoformat()}"
        completed_at = datetime.utcnow()
        started_at = self._deployment_start or completed_at

        return DeploymentReport(
            id=deployment_id,
            capability=self.capability_name,
            version=self.version,
            environment=self._last_apply_result.artifacts.get("environment", "unknown"),
            result=self._last_apply_result.status,
            applied_steps=self._last_apply_result.applied_steps,
            duration_sec=(completed_at - started_at).total_seconds(),
            started_at=started_at,
            completed_at=completed_at,
            links={
                "logs": self._last_apply_result.artifacts.get("logs_url", ""),
                "dashboard": self._last_apply_result.artifacts.get("dashboard_url", ""),
            },
            post_checks=self._last_verification.slis if self._last_verification else {},
            verification=self._last_verification,
            errors=self._last_apply_result.error_message.split("\n")
            if self._last_apply_result.error_message
            else [],
        )

    def deploy_full_lifecycle(
        self, spec: Dict[str, Any], creds: Dict[str, Any], env: str, auto_rollback: bool = True
    ) -> DeploymentReport:
        """Execute complete deployment lifecycle: plan → apply → verify → report.

        This is a convenience method that orchestrates the full PRIME_DIRECTIVE flow.

        Args:
            spec: Deployment specification
            creds: Credentials
            env: Target environment
            auto_rollback: Whether to automatically rollback on failure

        Returns:
            DeploymentReport with complete deployment results
        """
        self._deployment_start = datetime.utcnow()
        
        try:
            # Step 1: Describe requirements (Explain)
            self.logger.info("Describing requirements for %s v%s", self.capability_name, self.version)
            requirements = self.describe_requirements(env)
            self.logger.info(
                "Requirements: %d permissions, %d dependencies, estimated cost: $%.2f/mo",
                len(requirements.permissions),
                len(requirements.dependencies),
                requirements.estimated_cost_floor_usd,
            )

            # Step 2: Validate permissions
            self.logger.info("Validating permissions...")
            validation = self.validate_permissions(creds, env)
            if validation.status != "valid":
                raise RuntimeError(
                    f"Permission validation failed: {', '.join(validation.validation_errors)}"
                )
            self.logger.info("Permissions validated successfully")

            # Step 3: Generate plan (Plan)
            self.logger.info("Generating deployment plan...")
            plan = self.plan(spec, env)
            self.logger.info(
                "Plan generated: %d steps, %d rollback actions, checksum=%s",
                len(plan.steps),
                len(plan.rollback),
                plan.checksum[:8],
            )

            # Step 4: Apply changes (Apply)
            self.logger.info("Applying deployment plan...")
            self._last_apply_result = self.apply(plan, creds)

            if self._last_apply_result.status == ApplyStatus.FAILED:
                if auto_rollback:
                    self.logger.warning("Deployment failed, executing automatic rollback...")
                    rollback_result = self.rollback(plan, creds)
                    if rollback_result.status == ApplyStatus.SUCCESS:
                        self.logger.info("Rollback completed successfully")
                    else:
                        self.logger.error("Rollback failed: %s", rollback_result.error_message)
                raise RuntimeError(f"Deployment failed: {self._last_apply_result.error_message}")

            self.logger.info("Deployment applied successfully in %.2fs", self._last_apply_result.duration_seconds)

            # Step 5: Verify deployment (Verify)
            self.logger.info("Verifying deployment...")
            self._last_verification = self.verify(env)
            if self._last_verification.overall_status == "unhealthy":
                self.logger.warning("Verification failed, deployment is unhealthy")
                if auto_rollback:
                    self.logger.warning("Executing automatic rollback...")
                    rollback_result = self.rollback(plan, creds)
                    raise RuntimeError(
                        f"Deployment verification failed: {', '.join(self._last_verification.errors)}"
                    )
            self.logger.info("Verification passed: %s", self._last_verification.overall_status)

            # Step 6: Generate report (Report)
            report = self.report()
            self.logger.info("Deployment completed successfully: %s", report.id)
            return report

        except Exception as exc:
            self.logger.exception("Deployment lifecycle failed: %s", exc)
            # Try to generate a report even on failure
            if self._last_apply_result is not None:
                return self.report()
            raise

