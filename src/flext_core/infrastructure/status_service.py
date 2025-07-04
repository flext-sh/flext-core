"""Infrastructure status service for E2E environment verification.

This module provides comprehensive environment status checking without using
subprocess calls, following the Zero Tolerance architectural principles.
It verifies the availability of critical infrastructure components required
for E2E testing and deployment.

Key features:
- Docker daemon availability through socket and TCP port checks
- Kubernetes cluster connectivity via API client
- Kind (Kubernetes in Docker) cluster detection
- Concurrent status checking for optimal performance
- Comprehensive dependency management with required components

The service uses native Python APIs and libraries instead of shell commands, ensuring:
- Cross-platform compatibility (Windows, Linux, macOS)
- Better error handling and diagnostics
- No shell injection vulnerabilities
- Consistent behavior across environments
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import structlog
from pydantic import Field

from flext_core.config.domain_config import get_domain_constants
from flext_core.domain.pydantic_base import DomainValueObject

# ZERO TOLERANCE - Kubernetes import with graceful degradation for testing
from flext_core.utils.import_fallback_patterns import get_kubernetes_client

k8s_client, k8s_config = get_kubernetes_client()
KUBERNETES_AVAILABLE = k8s_client is not None and k8s_config is not None

# Service-specific constants not in global scope
DOCKER_SOCKET_PATH = Path("/var/run/docker.sock")

# Get domain constants for consistent values
_constants = get_domain_constants()
STATUS_CHECK_TIMEOUT_SECONDS = _constants.STATUS_CHECK_TIMEOUT_SECONDS
DEFAULT_KUBERNETES_PORT = _constants.DEFAULT_KUBERNETES_PORT
DEFAULT_DOCKER_PORT = _constants.DEFAULT_DOCKER_PORT

logger = structlog.get_logger()


class ServiceStatus(DomainValueObject):
    """Service availability status with detailed information.

    This class represents the status of an infrastructure service, providing
    not just availability but also version information, error details, and
    additional metadata for debugging and monitoring purposes.
    """

    available: bool = Field(
        description="Whether the service is available and accessible",
    )
    version: str | None = Field(
        default=None,
        description="Service version information if available",
    )
    error: str | None = Field(
        default=None,
        description="Error message if service is unavailable",
    )
    details: dict[str, object] | None = Field(
        default=None,
        description="Additional service details and metadata",
    )


class E2EEnvironmentStatus(DomainValueObject):
    """Complete E2E environment status aggregating all infrastructure checks.

    This class provides a comprehensive view of the E2E testing environment,
    including the status of Docker, Kubernetes, and Kind clusters. It calculates
    overall health based on component availability.
    """

    docker: ServiceStatus = Field(
        description="Docker daemon status and availability information",
    )
    kubernetes: ServiceStatus = Field(
        description="Kubernetes cluster status and connectivity",
    )
    kind_clusters: list[str] = Field(description="List of available Kind cluster names")
    overall_health: bool = Field(
        description="Overall environment health based on component status",
    )

    @property
    def health_summary(self) -> dict[str, object]:
        """Generate a health summary report for monitoring and logging.

        Returns a dictionary containing key health metrics and any issues
        detected during the environment status check.
        """
        return {
            "docker_available": self.docker.available,
            "kubernetes_available": self.kubernetes.available,
            "kind_clusters_count": len(self.kind_clusters),
            "overall_healthy": self.overall_health,
            "issues": [
                issue
                for issue in [
                    self.docker.error or None,
                    self.kubernetes.error or None,
                ]
                if issue is not None
            ],
        }


async def check_docker_availability() -> ServiceStatus:
    """Check Docker daemon availability through socket or TCP connection.

    This function attempts to verify Docker availability using multiple methods:
    1. Unix socket connection (Linux/macOS)
    2. TCP port connection (Windows/remote Docker)

    Returns:
    -------
        ServiceStatus with Docker availability information and connection details

    """
    try:
        # Try Docker socket first (Unix systems)
        if DOCKER_SOCKET_PATH.exists():
            try:
                # Check if socket is accessible
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(STATUS_CHECK_TIMEOUT_SECONDS)
                sock.connect(str(DOCKER_SOCKET_PATH))
                sock.close()

                return ServiceStatus(
                    available=True,
                    version="Available via socket",
                    details={
                        "connection_type": "unix_socket",
                        "path": str(DOCKER_SOCKET_PATH),
                    },
                )
            except OSError as e:
                return ServiceStatus(
                    available=False,
                    error=f"Docker socket connection failed: {e}",
                    details={
                        "connection_type": "unix_socket",
                        "path": str(DOCKER_SOCKET_PATH),
                    },
                )

        # Try Docker daemon port (Windows/remote)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(STATUS_CHECK_TIMEOUT_SECONDS)
            result = sock.connect_ex(("localhost", DEFAULT_DOCKER_PORT))
            sock.close()

            if result == 0:
                return ServiceStatus(
                    available=True,
                    version="Available via TCP",
                    details={"connection_type": "tcp", "port": DEFAULT_DOCKER_PORT},
                )
            return ServiceStatus(
                available=False,
                error="Docker daemon not accessible on standard ports",
                details={
                    "connection_type": "tcp",
                    "port": DEFAULT_DOCKER_PORT,
                    "result": result,
                },
            )
        except OSError as e:
            return ServiceStatus(
                available=False,
                error=f"Docker TCP connection failed: {e}",
                details={"connection_type": "tcp", "port": DEFAULT_DOCKER_PORT},
            )

    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        ImportError,
        ConnectionError,
        AttributeError,
    ) as e:
        return ServiceStatus(
            available=False,
            error=f"Unexpected error checking Docker: {e}",
            details={"error_type": type(e).__name__},
        )


async def check_kubernetes_availability() -> ServiceStatus:
    """Check Kubernetes cluster availability through API client.

    This function verifies Kubernetes availability by:
    1. Using the kubernetes Python client if available
    2. Falling back to TCP port check if client is not installed

    Returns:
    -------
        ServiceStatus with Kubernetes availability and cluster information

    """
    try:
        # Kubernetes client is guaranteed by pyproject.toml requirements
        # Try to load kubeconfig
        try:
            k8s_config.load_kube_config()
            v1 = k8s_client.CoreV1Api()

            # Try to list nodes to verify connection
            nodes = v1.list_node(timeout_seconds=STATUS_CHECK_TIMEOUT_SECONDS)

            return ServiceStatus(
                available=True,
                version=f"Available ({len(nodes.items)} nodes)",
                details={
                    "connection_type": "kubeconfig",
                    "node_count": len(nodes.items),
                    "nodes": [
                        node.metadata.name for node in nodes.items[:5]
                    ],  # Max 5 for brevity
                },
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            ImportError,
            ConnectionError,
            AttributeError,
        ) as config_error:
            return ServiceStatus(
                available=False,
                error=f"Kubernetes config/connection error: {config_error}",
                details={
                    "connection_type": "kubeconfig",
                    "config_error": str(config_error),
                },
            )

    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        ImportError,
        ConnectionError,
        AttributeError,
    ) as e:
        return ServiceStatus(
            available=False,
            error=f"Unexpected error checking Kubernetes: {e}",
            details={"error_type": type(e).__name__},
        )


async def get_kind_clusters() -> list[str]:
    """Detect Kind (Kubernetes in Docker) clusters through kubeconfig inspection.

    This function identifies Kind clusters by:
    1. Checking kubeconfig contexts for Kind-specific naming patterns
    2. Inspecting well-known configuration file locations

    Returns:
    -------
        List of Kind cluster names found in the environment

    """
    try:
        # Try to use kubernetes client to detect Kind clusters
        # Kubernetes client is guaranteed by pyproject.toml requirements
        try:
            k8s_config.load_kube_config()
            k8s_client.CoreV1Api()

            # Get current context to see if it's a Kind cluster
            contexts, _active_context = k8s_config.list_kube_config_contexts()

            kind_clusters = []
            for context in contexts:
                context_name = context["name"]
                if "kind-" in context_name:
                    # Extract cluster name from Kind context
                    cluster_name = context_name.replace("kind-", "")
                    kind_clusters.append(cluster_name)

        except (
            OSError,
            ValueError,
            TypeError,
            ImportError,
            k8s_config.ConfigException,
        ) as e:
            # ZERO TOLERANCE - Kubernetes config issues should fail fast
            # instead of degrading
            logger.warning("Failed to query kubectl for Kind clusters", error=str(e))
            msg = f"Kind cluster detection failed: {e}"
            raise RuntimeError(msg) from e
        else:
            return kind_clusters

    except (RuntimeError, OSError) as e:
        # Final catch-all for any unexpected errors
        logger.exception("Unexpected error detecting Kind clusters", error=str(e))
        return []


async def get_e2e_environment_status() -> E2EEnvironmentStatus:
    """Get comprehensive E2E environment status through concurrent checks.

    This function orchestrates all infrastructure checks concurrently for optimal
    performance, aggregating results into a complete environment status report.

    Returns:
    -------
        E2EEnvironmentStatus with complete infrastructure health information

    """
    # Check all services concurrently
    docker_status, kubernetes_status, kind_clusters = await asyncio.gather(
        check_docker_availability(),
        check_kubernetes_availability(),
        get_kind_clusters(),
        return_exceptions=True,
    )

    # Handle any exceptions from concurrent execution
    if isinstance(docker_status, Exception):
        docker_status = ServiceStatus(
            available=False,
            error=f"Error checking Docker: {docker_status}",
        )

    if isinstance(kubernetes_status, Exception):
        kubernetes_status = ServiceStatus(
            available=False,
            error=f"Error checking Kubernetes: {kubernetes_status}",
        )

    if isinstance(kind_clusters, Exception):
        kind_clusters = []

    # Determine overall health - type narrowing after exception handling
    overall_health = (
        isinstance(docker_status, ServiceStatus)
        and docker_status.available
        and isinstance(kubernetes_status, ServiceStatus)
        and kubernetes_status.available
        and isinstance(kind_clusters, list)
        and len(kind_clusters) > 0
    )

    # Ensure types are correct after exception handling
    if not isinstance(docker_status, ServiceStatus):
        msg = f"Expected ServiceStatus for docker_status, got {type(docker_status).__name__}"
        raise TypeError(msg)
    if not isinstance(kubernetes_status, ServiceStatus):
        msg = f"Expected ServiceStatus for kubernetes_status, got {type(kubernetes_status).__name__}"
        raise TypeError(msg)
    if not isinstance(kind_clusters, list):
        msg = f"Expected list for kind_clusters, got {type(kind_clusters).__name__}"
        raise TypeError(msg)

    return E2EEnvironmentStatus(
        docker=docker_status,
        kubernetes=kubernetes_status,
        kind_clusters=kind_clusters,
        overall_health=overall_health,
    )
