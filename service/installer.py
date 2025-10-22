"""
Installer definition for the Slack Integration service.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from installer import (
    ApplyResult,
    ApplyStatus,
    DeploymentPlan,
    DeploymentStep,
    Installer,
    PermissionScope,
    Requirements,
    Resources,
    ValidationResult,
    ValidationStatus,
)
from installer.railway import RailwayProvider


class SlackIntegrationInstaller(Installer):
    """Installer that provisions the slack-integration service on Railway."""

    def __init__(self) -> None:
        super().__init__(capability_name="slack-integration", version="1.0.0")
        self.service_name = "budai-slack-integration"

    def describe_requirements(self, env: str) -> Requirements:
        return Requirements(
            capability="slack-integration",
            version="1.0.0",
            permissions=[
                PermissionScope(provider="railway", service="project", action="create_service", scope="project-level"),
                PermissionScope(provider="slack", service="api", action="web", scope="bot-token"),
            ],
            dependencies=[
                {"type": "network", "needs": ["egress to slack.com:443"], "name": None, "port": None},
                {"type": "service", "name": "redis", "port": 6379, "needs": []},
            ],
            resources=Resources(memory_mb=512, cpu_millicores=500),
            estimated_cost_floor_usd=10.0,
        )

    def validate_permissions(self, creds: dict, env: str) -> ValidationResult:
        validated, missing, errors = [], [], []
        if "railway_token" in creds:
            validated.append(
                PermissionScope(provider="railway", service="project", action="create_service", scope="project-level")
            )
        if creds.get("slack_bot_token") or creds.get("BUDAI_SLACK_BOT_TOKEN"):
            validated.append(
                PermissionScope(provider="slack", service="api", action="web", scope="bot-token")
            )
        else:
            missing.append(
                PermissionScope(provider="slack", service="api", action="web", scope="bot-token")
            )
            errors.append("Slack bot token not provided")
        status = ValidationStatus.VALID if not missing else ValidationStatus.INVALID
        return ValidationResult(status=status, validated_permissions=validated, missing_permissions=missing, validation_errors=errors)

    def plan(self, spec: dict, env: str) -> DeploymentPlan:
        steps = [
            DeploymentStep(
                id="railway.create_service",
                action="create_railway_service",
                params={"service_name": self.service_name, "environment": env},
                retriable=True,
            ),
            DeploymentStep(
                id="env.set_variables",
                action="set_environment_variables",
                params={
                    "variables": {
                        "BUDAI_SERVICE_NAME": "slack-integration",
                        "PORT": "8006",
                    }
                },
                retriable=True,
                depends_on=["railway.create_service"],
            ),
            DeploymentStep(
                id="railway.deploy",
                action="trigger_deployment",
                params={"service_name": self.service_name, "wait_for_health": True},
                retriable=False,
                depends_on=["env.set_variables"],
            ),
        ]
        return DeploymentPlan(
            target_env=env,
            capability="slack-integration",
            version="1.0.0",
            steps=steps,
            rollback=[],
            invariants=[],
            checksum="",
        )

    def apply(self, plan: DeploymentPlan, creds: dict) -> ApplyResult:
        import time

        start_time = time.time()
        applied_steps: List[str] = []
        artifacts: dict = {}

        try:
            railway = RailwayProvider(api_token=creds["railway_token"], project_id=creds.get("railway_project_id"))
            for step in plan.steps:
                if step.action == "create_railway_service":
                    target_env = step.params.get("environment", plan.target_env)
                    artifacts["service_id"] = railway.create_service(
                        name=step.params["service_name"],
                        source_repo=creds.get("github_repo"),
                        source_branch=creds.get("github_branch", "main"),
                        environment=target_env,
                    )
                    artifacts["environment"] = target_env
                elif step.action == "set_environment_variables":
                    variables = dict(step.params["variables"])
                    if creds.get("redis_url"):
                        variables.setdefault("REDIS_URL", creds["redis_url"])
                        variables.setdefault("BUDAI_REDIS_URL", creds["redis_url"])
                    if creds.get("slack_bot_token"):
                        variables.setdefault("SLACK_BOT_TOKEN", creds["slack_bot_token"])
                    if creds.get("slack_signing_secret"):
                        variables.setdefault("SLACK_SIGNING_SECRET", creds["slack_signing_secret"])
                    railway.set_environment_variables(artifacts["service_id"], plan.target_env, variables)
                elif step.action == "trigger_deployment":
                    artifacts["deployment_id"] = railway.deploy_service(
                        service_id=artifacts["service_id"],
                        environment=artifacts.get("environment", plan.target_env),
                    )
                    if step.params.get("wait_for_health"):
                        railway.wait_for_deployment(artifacts["deployment_id"], timeout_seconds=600)
                applied_steps.append(step.id)

            return ApplyResult(
                status=ApplyStatus.SUCCESS,
                applied_steps=applied_steps,
                duration_seconds=time.time() - start_time,
                artifacts=artifacts,
            )
        except Exception as exc:  # pragma: no cover - deployment failure path
            return ApplyResult(
                status=ApplyStatus.FAILED,
                applied_steps=applied_steps,
                error_message=str(exc),
                duration_seconds=time.time() - start_time,
                artifacts=artifacts,
            )


INSTALLER = SlackIntegrationInstaller()
