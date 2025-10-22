"""
Railway-specific deployment provider.

Implements Railway API integration for automated service creation,
configuration, and deployment per PRIME_DIRECTIVE requirements.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import secrets
import string

import httpx

logger = logging.getLogger(__name__)


class RailwayAPIError(Exception):
    """Raised when Railway API calls fail."""

    pass


class RailwayProvider:
    """Railway cloud provider for automated deployments.

    Handles Railway-specific operations:
    - Project and service creation
    - Environment variable configuration
    - Deployment triggering
    - Health check monitoring
    """

    def __init__(self, api_token: Optional[str] = None, project_id: Optional[str] = None) -> None:
        """Initialize Railway provider.

        Args:
            api_token: Railway API token (defaults to RAILWAY_TOKEN env var)
            project_id: Railway project ID (defaults to RAILWAY_PROJECT_ID env var)
        """
        self.api_token = api_token or os.getenv("RAILWAY_TOKEN")
        if not self.api_token:
            raise ValueError(
                "Railway API token required. Set RAILWAY_TOKEN env var or pass api_token."
            )

        self.project_id = project_id or os.getenv("RAILWAY_PROJECT_ID")
        self.graphql_url = "https://backboard.railway.app/graphql/v2"
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        # Basic throttling to keep Cloudflare happy
        self._min_request_interval = float(os.getenv("RAILWAY_API_MIN_INTERVAL", "0.75"))
        self._last_request_ts = 0.0

        # Simple caches to avoid redundant lookups
        self._env_cache: Dict[Tuple[str, str], str] = {}
        self._services_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()

    def _graphql_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        *,
        retries: int = 5,
    ) -> Dict[str, Any]:
        """Execute a GraphQL query against Railway API.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Query result data

        Raises:
            RailwayAPIError: If query fails
        """
        attempt = 0
        delay = 2.0

        while attempt <= retries:
            attempt += 1

            # Throttle client-side
            if self._min_request_interval > 0:
                elapsed = time.time() - self._last_request_ts
                if elapsed < self._min_request_interval:
                    time.sleep(self._min_request_interval - elapsed)

            try:
                response = self.client.post(
                    self.graphql_url,
                    json={"query": query, "variables": variables or {}},
                )
                self._last_request_ts = time.time()

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    sleep_for = float(retry_after) if retry_after else delay
                    time.sleep(sleep_for)
                    delay = min(delay * 1.5, 60.0)
                    continue

                data = response.json()

                if "errors" in data:
                    messages = [e.get("message", str(e)) for e in data["errors"]]
                    joined = "; ".join(messages)

                    if "Too Many Requests" in joined and attempt <= retries:
                        time.sleep(delay)
                        delay = min(delay * 1.5, 60.0)
                        continue

                    raise RailwayAPIError(f"GraphQL errors: {joined}")

                # Only raise for non-200 status if we didn't get JSON errors
                response.raise_for_status()

                return data.get("data", {})

            except httpx.HTTPError as exc:
                detail = ""
                status_code = None
                if getattr(exc, "response", None) is not None:
                    status_code = exc.response.status_code
                    if status_code == 429 and attempt <= retries:
                        retry_after = exc.response.headers.get("Retry-After")
                        sleep_for = float(retry_after) if retry_after else delay
                        time.sleep(sleep_for)
                        delay = min(delay * 1.5, 60.0)
                        continue
                    try:
                        detail = exc.response.text
                    except Exception:
                        detail = ""
                
                # Log error details for debugging
                logger.error("Railway API HTTP error (status %s, attempt %d/%d): %s", 
                            status_code, attempt, retries + 1, detail)
                
                # Don't retry on 400 Bad Request - these indicate invalid parameters
                if status_code == 400:
                    raise RailwayAPIError(
                        f"Railway API request failed (400 Bad Request - invalid parameters): {detail}"
                    ) from exc
                
                if attempt > retries:
                    raise RailwayAPIError(f"Railway API request failed: {exc} :: {detail}") from exc
                time.sleep(delay)
                delay = min(delay * 1.5, 60.0)

        raise RailwayAPIError("Railway API request exhausted retries")

    def create_project(self, name: str, description: Optional[str] = None) -> str:
        """Create a new Railway project.

        Args:
            name: Project name
            description: Optional project description

        Returns:
            Created project ID
        """
        query = """
        mutation CreateProject($name: String!, $description: String) {
            projectCreate(input: {
                name: $name
                description: $description
            }) {
                id
                name
            }
        }
        """
        
        result = self._graphql_query(query, {"name": name, "description": description})
        project = result.get("projectCreate", {})
        project_id = project.get("id")
        
        if not project_id:
            raise RailwayAPIError(f"Failed to create project '{name}'")
        
        logger.info("Created Railway project: %s (ID: %s)", name, project_id)
        return project_id

    def create_service(
        self,
        name: str,
        project_id: Optional[str] = None,
        source_repo: Optional[str] = None,
        source_branch: Optional[str] = None,
        source_image: Optional[str] = None,
        environment_id: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        environment: Optional[str] = None,
    ) -> str:
        """Create a new service in a Railway project.

        Args:
            name: Service name
            project_id: Project ID (uses instance default if not provided)
            source_repo: Optional GitHub repo URL
            source_branch: Optional repo branch (default: main)
            source_image: Optional container image to deploy
            environment_id: Railway environment ID
            variables: Environment variables to seed on creation
            environment: Environment name (resolved to ID if provided)

        Returns:
            Created service ID
        """
        proj_id = project_id or self.project_id
        if not proj_id:
            raise ValueError("Project ID required. Set project_id or RAILWAY_PROJECT_ID.")

        existing = self._get_service_by_name(name, proj_id)
        if existing:
            logger.info("Service %s already exists (ID: %s)", name, existing["id"])
            return existing["id"]

        query = """
        mutation CreateService(
            $projectId: String!,
            $name: String!,
            $source: ServiceSourceInput,
            $environmentId: String,
            $variables: EnvironmentVariables
        ) {
            serviceCreate(input: {
                projectId: $projectId
                name: $name
                source: $source
                environmentId: $environmentId
                variables: $variables
            }) {
                id
                name
            }
        }
        """

        env_id = environment_id
        if environment and not env_id:
            env_id = self._get_environment_id(proj_id, environment)

        processed_repo = None
        source_payload: Optional[Dict[str, Any]] = None
        if source_repo:
            repo_value = source_repo.strip()
            if repo_value.startswith("https://github.com/"):
                repo_value = repo_value.split("https://github.com/", 1)[1]
            elif repo_value.startswith("http://github.com/"):
                repo_value = repo_value.split("http://github.com/", 1)[1]
            if repo_value.endswith(".git"):
                repo_value = repo_value[:-4]
            processed_repo = repo_value
            source_payload = {"repo": repo_value}
            if source_branch:
                logger.debug(
                    "Railway GraphQL API does not support setting branch '%s' during serviceCreate; using repo default",
                    source_branch,
                )
        elif source_image:
            source_payload = {"image": source_image}

        if processed_repo and not env_id:
            logger.debug(
                "Creating repo-backed service %s without explicit environment mapping", name
            )

        result = self._graphql_query(
            query,
            {
                "projectId": proj_id,
                "name": name,
                "source": source_payload,
                "environmentId": env_id,
                "variables": variables or None,
            },
        )
        service = result.get("serviceCreate", {})
        service_id = service.get("id")

        if not service_id:
            raise RailwayAPIError(f"Failed to create service '{name}'")

        proj_cache = self._services_cache.setdefault(proj_id, {})
        proj_cache[name] = {"id": service_id, "name": name}

        logger.info("Created Railway service: %s (ID: %s)", name, service_id)
        return service_id

    def _list_services(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List services for a project."""
        proj_id = project_id or self.project_id
        if not proj_id:
            raise ValueError("Project ID required")

        cached = self._services_cache.get(proj_id)
        if cached is not None:
            return list(cached.values())

        query = """
        query ListServices($projectId: String!) {
            project(id: $projectId) {
                services(first: 100) {
                    edges {
                        node {
                            id
                            name
                            templateServiceId
                        }
                    }
                }
            }
        }
        """
        result = self._graphql_query(query, {"projectId": proj_id})
        edges = result.get("project", {}).get("services", {}).get("edges", [])
        services = [edge.get("node", {}) for edge in edges]
        self._services_cache[proj_id] = {svc.get("name"): svc for svc in services if svc.get("name")}
        return services

    def _get_service_by_name(self, name: str, project_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find a service by name."""
        proj_id = project_id or self.project_id
        if proj_id and proj_id in self._services_cache:
            cached = self._services_cache[proj_id]
            if name in cached:
                return cached[name]

        for service in self._list_services(proj_id):
            if service.get("name") == name:
                return service
        return None

    def _get_service_instance(
        self, service_id: str, environment_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single service instance for a service/environment."""
        query = """
        query GetServiceInstance($serviceId: String!) {
            service(id: $serviceId) {
                serviceInstances(first: 5) {
                    edges {
                        node {
                            id
                            environmentId
                            latestDeployment {
                                id
                                status
                            }
                        }
                    }
                }
            }
        }
        """
        result = self._graphql_query(query, {"serviceId": service_id})
        edges = result.get("service", {}).get("serviceInstances", {}).get("edges", [])
        for edge in edges:
            node = edge.get("node", {})
            if node.get("environmentId") == environment_id:
                return node
        return None

    def _connect_service_repo(
        self,
        service_id: str,
        repo: str,
        branch: Optional[str] = None,
        image: Optional[str] = None,
    ) -> None:
        """Associate a Git repository (or container image) with a service."""
        query = """
        mutation ConnectService($serviceId: String!, $input: ServiceConnectInput!) {
            serviceConnect(id: $serviceId, input: $input) {
                id
            }
        }
        """
        input_payload: Dict[str, Any] = {"repo": repo}
        if branch:
            input_payload["branch"] = branch
        if image:
            input_payload["image"] = image

        self._graphql_query(
            query,
            {
                "serviceId": service_id,
                "input": input_payload,
            },
        )

    def _wait_for_service_instance(
        self,
        service_id: str,
        environment_id: str,
        timeout_seconds: int = 600,
        poll_interval: int = 5,
    ) -> bool:
        """Wait for a service instance deployment to succeed."""
        start = time.time()
        while (time.time() - start) < timeout_seconds:
            instance = self._get_service_instance(service_id, environment_id)
            if instance:
                deployment = instance.get("latestDeployment")
                if deployment:
                    status = (deployment.get("status") or "").upper()
                    if status == "SUCCESS":
                        return True
                    if status in {"FAILED", "CRASHED", "REMOVED"}:
                        raise RailwayAPIError(
                            f"Service {service_id} deployment failed with status {status}"
                        )
            time.sleep(poll_interval)
        raise RailwayAPIError(
            f"Timed out waiting for service {service_id} deployment in environment {environment_id}"
        )

    def get_service_variables(
        self,
        project_id: str,
        environment_id: str,
        service_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Retrieve environment variables for a service/environment."""
        query = """
        query GetVariables($projectId: String!, $environmentId: String!, $serviceId: String) {
            variables(projectId: $projectId, environmentId: $environmentId, serviceId: $serviceId)
        }
        """
        result = self._graphql_query(
            query,
            {
                "projectId": project_id,
                "environmentId": environment_id,
                "serviceId": service_id,
            },
        )
        return result.get("variables", {}) or {}

    def ensure_redis_service(
        self,
        environment: str,
        project_id: Optional[str] = None,
        password_length: int = 24,
    ) -> Dict[str, str]:
        """Ensure a Redis instance exists for the environment and return connection info."""
        proj_id = project_id or self.project_id
        if not proj_id:
            raise ValueError("Project ID required")

        env_id = self._get_environment_id(proj_id, environment)
        service_name = f"budai-redis-{environment}"

        service = self._get_service_by_name(service_name, proj_id)
        created = False

        if service:
            service_id = service["id"]
        else:
            password = "".join(
                secrets.choice(string.ascii_letters + string.digits) for _ in range(password_length)
            )
            variables = {
                "REDIS_PASSWORD": password,
                "ALLOW_EMPTY_PASSWORD": "no",
            }
            service_id = self.create_service(
                name=service_name,
                project_id=proj_id,
                environment_id=env_id,
                source_image="railwayapp/redis:8.2",
                variables=variables,
            )
            created = True

        if created:
            self._wait_for_service_instance(service_id, env_id)
        else:
            # Ensure redis has required variables; if missing, set them and redeploy
            existing_vars = self.get_service_variables(proj_id, env_id, service_id)
            if "REDIS_PASSWORD" not in existing_vars:
                password = "".join(
                    secrets.choice(string.ascii_letters + string.digits) for _ in range(password_length)
                )
                self.set_environment_variables(
                    service_id=service_id,
                    environment=environment,
                    variables={
                        "REDIS_PASSWORD": password,
                        "ALLOW_EMPTY_PASSWORD": "no",
                    },
                    project_id=proj_id,
                )
                self._wait_for_service_instance(service_id, env_id)

        vars_payload = self.get_service_variables(proj_id, env_id, service_id)

        password = vars_payload.get("REDIS_PASSWORD")
        host = (
            vars_payload.get("RAILWAY_PRIVATE_DOMAIN")
            or vars_payload.get("REDIS_HOST")
            or vars_payload.get("REDISHOST")
        )
        port = (
            vars_payload.get("REDIS_PORT")
            or vars_payload.get("REDISPORT")
            or "6379"
        )
        redis_user = (
            vars_payload.get("REDISUSER")
            or vars_payload.get("REDIS_USERNAME")
            or "default"
        )

        if not password or not host:
            raise RailwayAPIError(
                f"Redis service {service_name} missing required connection variables"
            )
        port = str(port)
        redis_url = f"redis://:{password}@{host}:{port}/0"

        # Ensure all expected connection variables are present for Railway UI integrations
        desired_vars = {
            "REDIS_PASSWORD": password,
            "REDIS_HOST": host,
            "REDIS_PORT": port,
            "REDIS_URL": redis_url,
            "REDIS_USERNAME": redis_user,
            "REDISHOST": host,
            "REDISPORT": port,
            "REDISPASSWORD": password,
            "REDISUSER": redis_user,
        }
        to_update = {
            key: value
            for key, value in desired_vars.items()
            if value and vars_payload.get(key) != value
        }
        if to_update:
            self.set_environment_variables(
                service_id=service_id,
                environment=environment,
                variables=to_update,
                project_id=proj_id,
            )
            vars_payload.update(to_update)

        return {
            "service_id": service_id,
            "environment_id": env_id,
            "host": host,
            "port": port,
            "password": password,
            "redis_url": redis_url,
        }

    def set_environment_variables(
        self,
        service_id: str,
        environment: str,
        variables: Dict[str, str],
        project_id: Optional[str] = None,
    ) -> None:
        """Set environment variables for a service.

        Args:
            service_id: Service ID
            environment: Environment name (e.g., 'production')
            variables: Dictionary of variable names to values
            project_id: Project ID (uses instance default if not provided)
        """
        proj_id = project_id or self.project_id
        if not proj_id:
            raise ValueError("Project ID required")

        # Railway API uses a mutation per variable
        query = """
        mutation SetVariable($projectId: String!, $environmentId: String!, $serviceId: String!, $name: String!, $value: String!) {
            variableUpsert(input: {
                projectId: $projectId
                environmentId: $environmentId
                serviceId: $serviceId
                name: $name
                value: $value
            })
        }
        """

        # Get environment ID (simplified - in reality you'd query for it)
        env_id = self._get_environment_id(proj_id, environment)

        for name, value in variables.items():
            try:
                self._graphql_query(
                    query,
                    {
                        "projectId": proj_id,
                        "environmentId": env_id,
                        "serviceId": service_id,
                        "name": name,
                        "value": value,
                    },
                )
                logger.debug("Set variable %s for service %s", name, service_id)
            except RailwayAPIError as exc:
                logger.error("Failed to set variable %s: %s", name, exc)
                raise

        logger.info("Set %d environment variables for service %s", len(variables), service_id)

    def _find_environment_id(self, project_id: str, environment_name: str) -> Optional[str]:
        """Look up an environment ID by name (returns None if not found)."""
        cache_key = (project_id, environment_name)
        if cache_key in self._env_cache:
            return self._env_cache[cache_key]

        query = """
        query GetEnvironments($projectId: String!) {
            project(id: $projectId) {
                environments {
                    edges {
                        node {
                            id
                            name
                        }
                    }
                }
            }
        }
        """

        result = self._graphql_query(query, {"projectId": project_id})
        project = result.get("project", {})
        environments = project.get("environments", {}).get("edges", [])

        for edge in environments:
            node = edge.get("node", {})
            if node.get("name") == environment_name:
                self._env_cache[cache_key] = node["id"]
                return node["id"]

        return None

    def _get_environment_id(self, project_id: str, environment_name: str) -> str:
        """Get environment ID by name, creating it if necessary."""
        existing = self._find_environment_id(project_id, environment_name)
        if existing:
            return existing

        # If environment doesn't exist, create it
        return self._create_environment(project_id, environment_name)

    def _create_environment(self, project_id: str, name: str) -> str:
        """Create a new environment.

        Args:
            project_id: Project ID
            name: Environment name

        Returns:
            Created environment ID
        """
        query = """
        mutation CreateEnvironment($projectId: String!, $name: String!) {
            environmentCreate(input: {
                projectId: $projectId
                name: $name
            }) {
                id
                name
            }
        }
        """

        try:
            result = self._graphql_query(query, {"projectId": project_id, "name": name})
        except RailwayAPIError as exc:
            if "already exists" in str(exc).lower():
                existing = self._find_environment_id(project_id, name)
                if existing:
                    logger.info("Environment %s already exists (ID: %s)", name, existing)
                    self._env_cache[(project_id, name)] = existing
                    return existing
            raise

        environment = result.get("environmentCreate", {})
        env_id = environment.get("id")

        if not env_id:
            raise RailwayAPIError(f"Failed to create environment '{name}'")

        logger.info("Created Railway environment: %s (ID: %s)", name, env_id)
        self._env_cache[(project_id, name)] = env_id
        return env_id

    def deploy_service(
        self,
        service_id: str,
        *,
        environment: Optional[str] = None,
        environment_id: Optional[str] = None,
        latest_commit: bool = True,
        project_id: Optional[str] = None,
    ) -> str:
        """Trigger a deployment for a service.

        Args:
            service_id: Service ID to deploy
            environment: Environment name (resolved to ID if provided)
            environment_id: Optional Railway environment ID (takes precedence)
            latest_commit: Deploy latest connected commit (default: True)
            project_id: Project ID (uses instance default if not provided)

        Returns:
            Deployment ID
        """
        proj_id = project_id or self.project_id
        if not proj_id:
            raise ValueError("Project ID required")

        env_id = environment_id
        if not env_id:
            if not environment:
                raise ValueError("environment or environment_id required to deploy service")
            env_id = self._get_environment_id(proj_id, environment)

        query = """
        mutation TriggerDeploy($serviceId: String!, $environmentId: String!, $latestCommit: Boolean) {
            serviceInstanceDeploy(
                serviceId: $serviceId
                environmentId: $environmentId
                latestCommit: $latestCommit
            ) {
                id
            }
        }
        """

        variables: Dict[str, Any] = {
            "serviceId": service_id,
            "environmentId": env_id,
            "latestCommit": latest_commit,
        }

        if latest_commit is None:
            variables.pop("latestCommit")

        try:
            result = self._graphql_query(query, variables)
            deployment = result.get("serviceInstanceDeploy", {}) or {}
            deployment_id = deployment.get("id")
        except RailwayAPIError as exc:
            if "Problem processing request" in str(exc):
                logger.debug(
                    "serviceInstanceDeploy failed for %s in %s, attempting redeploy fallback",
                    service_id,
                    env_id,
                )
                fallback_query = """
                mutation Redeploy($serviceId: String!, $environmentId: String!) {
                    serviceInstanceRedeploy(serviceId: $serviceId, environmentId: $environmentId) {
                        id
                    }
                }
                """
                result = self._graphql_query(
                    fallback_query,
                    {"serviceId": service_id, "environmentId": env_id},
                )
                deployment = result.get("serviceInstanceRedeploy", {}) or {}
                deployment_id = deployment.get("id")
            else:
                raise

        if not deployment_id:
            raise RailwayAPIError(f"Failed to trigger deployment for service {service_id}")

        logger.info(
            "Triggered deployment for service %s in environment %s: %s",
            service_id,
            env_id,
            deployment_id,
        )
        return deployment_id

    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of a deployment.

        Args:
            deployment_id: Deployment ID

        Returns:
            Deployment status information
        """
        query = """
        query GetDeployment($id: String!) {
            deployment(id: $id) {
                id
                status
                createdAt
                completedAt
                meta
            }
        }
        """

        result = self._graphql_query(query, {"id": deployment_id})
        return result.get("deployment", {})

    def wait_for_deployment(
        self, deployment_id: str, timeout_seconds: int = 600, poll_interval: int = 10
    ) -> bool:
        """Wait for a deployment to complete.

        Args:
            deployment_id: Deployment ID to wait for
            timeout_seconds: Maximum time to wait
            poll_interval: Seconds between status checks

        Returns:
            True if deployment succeeded, False otherwise
        """
        start_time = time.time()
        
        while (time.time() - start_time) < timeout_seconds:
            status_info = self.get_deployment_status(deployment_id)
            status = status_info.get("status", "").upper()

            if status == "SUCCESS":
                logger.info("Deployment %s completed successfully", deployment_id)
                return True
            elif status in ("FAILED", "CRASHED", "REMOVED"):
                logger.error("Deployment %s failed with status: %s", deployment_id, status)
                return False
            
            logger.debug("Deployment %s status: %s", deployment_id, status)
            time.sleep(poll_interval)

        logger.error("Deployment %s timed out after %ds", deployment_id, timeout_seconds)
        return False

    def get_service_domain(self, service_id: str) -> Optional[str]:
        """Get the public domain for a service.

        Args:
            service_id: Service ID

        Returns:
            Service domain URL or None if not available
        """
        query = """
        query GetServiceDomain($id: String!) {
            service(id: $id) {
                id
                domains {
                    serviceDomains {
                        domain
                    }
                }
            }
        }
        """

        result = self._graphql_query(query, {"id": service_id})
        service = result.get("service", {})
        domains = service.get("domains", {}).get("serviceDomains", [])

        if domains:
            return domains[0].get("domain")
        
        return None


    def service_instance_update(
        self,
        service_id: str,
        environment_id: str,
        *,
        start_command: Optional[str] = None,
        builder: Optional[str] = None,
        root_directory: Optional[str] = None,
        dockerfile_path: Optional[str] = None,
        healthcheck_path: Optional[str] = None,
        healthcheck_timeout: Optional[int] = None,
    ) -> None:
        """Update a service instance configuration (builder, start command, root directory, etc)."""
        query = """
        mutation UpdateInstance($serviceId: String!, $environmentId: String!, $input: ServiceInstanceUpdateInput!) {
            serviceInstanceUpdate(serviceId: $serviceId, environmentId: $environmentId, input: $input) { id }
        }
        """
        input_payload: Dict[str, Any] = {}
        if start_command:
            input_payload["startCommand"] = start_command
        if builder:
            input_payload["builder"] = builder
        if root_directory:
            input_payload["rootDirectory"] = root_directory
        if dockerfile_path:
            input_payload["dockerfilePath"] = dockerfile_path
        if healthcheck_path:
            input_payload["healthcheckPath"] = healthcheck_path
        if healthcheck_timeout:
            input_payload["healthcheckTimeout"] = healthcheck_timeout

        if not input_payload:
            return

        # Add a small delay to ensure the service is ready for configuration updates
        time.sleep(2)

        try:
            self._graphql_query(
                query,
                {
                    "serviceId": service_id,
                    "environmentId": environment_id,
                    "input": input_payload,
                },
            )
        except RailwayAPIError as exc:
            if "Problem processing request" in str(exc):
                logger.warning(
                    "Service instance update failed with 'Problem processing request' - this may be due to service not being ready yet. "
                    "The service will be created but may need manual configuration of root directory."
                )
                # Don't re-raise the exception - let the deployment continue
                return
            else:
                raise

    def deploy_service_in_environment(self, service_id: str, environment_id: str) -> str:
        """Trigger a deployment for a service in a specific environment."""
        query = """
        mutation Deploy($serviceId: String!, $environmentId: String!) {
            serviceInstanceDeploy(serviceId: $serviceId, environmentId: $environmentId, latestCommit: true) { id }
        }
        """
        result = self._graphql_query(query, {"serviceId": service_id, "environmentId": environment_id})
        deployment = result.get("serviceInstanceDeploy", {})
        dep_id = deployment.get("id")
        if not dep_id:
            raise RailwayAPIError("Failed to trigger deployment")
        return dep_id
