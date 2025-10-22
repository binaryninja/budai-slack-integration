"""
Health check framework for service monitoring.

Provides standardized health checks per PRIME_DIRECTIVE requirements.
Every service MUST implement /health endpoint with dependency checks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck(BaseModel):
    """Individual health check result."""

    name: str = Field(..., description="Health check name")
    status: HealthStatus = Field(..., description="Health status")
    message: Optional[str] = Field(None, description="Optional status message")
    latency_ms: Optional[float] = Field(None, description="Check execution time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional check metadata")


class ServiceHealth(BaseModel):
    """Overall service health report."""

    service_name: str
    version: str
    status: HealthStatus = Field(..., description="Overall service status")
    checks: List[HealthCheck] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)
    uptime_seconds: Optional[float] = None

    def is_healthy(self) -> bool:
        """Check if service is fully healthy."""
        return self.status == HealthStatus.HEALTHY

    def is_available(self) -> bool:
        """Check if service is available (healthy or degraded)."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


class HealthChecker:
    """Health check orchestrator for a service.

    Manages multiple health checks and provides aggregated status per
    PRIME_DIRECTIVE section 4 (verification requirements).
    """

    def __init__(self, service_name: str, version: str) -> None:
        """Initialize health checker.

        Args:
            service_name: Service name
            version: Service version
        """
        self.service_name = service_name
        self.version = version
        self.checks: Dict[str, Callable[[], Any]] = {}
        self.start_time = time.time()

    def register_check(self, name: str, check_fn: Callable[[], Any]) -> None:
        """Register a health check function.

        Args:
            name: Check name (e.g., "redis", "database", "openai_api")
            check_fn: Async or sync function that returns True if healthy, False otherwise
                     Can also return a tuple (bool, str) for status and message
        """
        self.checks[name] = check_fn
        logger.info("Registered health check: %s", name)

    async def run_check(self, name: str, check_fn: Callable[[], Any]) -> HealthCheck:
        """Run a single health check.

        Args:
            name: Check name
            check_fn: Check function

        Returns:
            HealthCheck result
        """
        start_time = time.time()
        
        try:
            # Execute check (handle both async and sync functions)
            if asyncio.iscoroutinefunction(check_fn):
                result = await check_fn()
            else:
                result = check_fn()

            latency_ms = (time.time() - start_time) * 1000

            # Parse result
            if isinstance(result, tuple):
                is_healthy, message = result
            elif isinstance(result, bool):
                is_healthy = result
                message = "OK" if is_healthy else "Check failed"
            else:
                is_healthy = bool(result)
                message = str(result) if result else "Check failed"

            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY

            return HealthCheck(
                name=name,
                status=status,
                message=message,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000
            logger.exception("Health check '%s' failed: %s", name, exc)
            
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Exception: {str(exc)}",
                latency_ms=latency_ms,
            )

    async def check_health(self) -> ServiceHealth:
        """Run all registered health checks.

        Returns:
            ServiceHealth with aggregated status
        """
        # Run all checks concurrently
        check_tasks = [
            self.run_check(name, check_fn)
            for name, check_fn in self.checks.items()
        ]
        
        check_results = await asyncio.gather(*check_tasks)

        # Determine overall status
        unhealthy_count = sum(1 for c in check_results if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in check_results if c.status == HealthStatus.DEGRADED)

        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        uptime = time.time() - self.start_time

        return ServiceHealth(
            service_name=self.service_name,
            version=self.version,
            status=overall_status,
            checks=list(check_results),
            uptime_seconds=uptime,
        )


# Common health check implementations


async def check_redis_connection(redis_client: Any) -> tuple[bool, str]:
    """Check Redis connection health.

    Args:
        redis_client: Redis client instance

    Returns:
        (is_healthy, message) tuple
    """
    try:
        await redis_client.ping()
        return True, "Redis connection OK"
    except Exception as exc:
        return False, f"Redis connection failed: {exc}"


async def check_http_endpoint(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Check HTTP endpoint health.

    Args:
        url: URL to check
        timeout: Request timeout in seconds

    Returns:
        (is_healthy, message) tuple
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            if response.status_code == 200:
                return True, f"Endpoint {url} responding"
            else:
                return False, f"Endpoint returned {response.status_code}"
    except Exception as exc:
        return False, f"Endpoint check failed: {exc}"


async def check_openai_api(api_key: str) -> tuple[bool, str]:
    """Check OpenAI API connectivity.

    Args:
        api_key: OpenAI API key

    Returns:
        (is_healthy, message) tuple
    """
    try:
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers=headers,
            )
            if response.status_code == 200:
                return True, "OpenAI API accessible"
            else:
                return False, f"OpenAI API returned {response.status_code}"
    except Exception as exc:
        return False, f"OpenAI API check failed: {exc}"


def create_liveness_check() -> Callable[[], bool]:
    """Create a simple liveness check (always healthy).

    Returns:
        Liveness check function
    """
    return lambda: True


def create_readiness_check(dependencies: List[Callable[[], Any]]) -> Callable[[], Any]:
    """Create a readiness check that validates all dependencies.

    Args:
        dependencies: List of dependency check functions

    Returns:
        Readiness check function
    """
    async def readiness_check() -> bool:
        for dep_check in dependencies:
            if asyncio.iscoroutinefunction(dep_check):
                result = await dep_check()
            else:
                result = dep_check()
            
            # Parse result
            if isinstance(result, tuple):
                is_healthy, _ = result
            else:
                is_healthy = bool(result)
            
            if not is_healthy:
                return False
        
        return True
    
    return readiness_check

