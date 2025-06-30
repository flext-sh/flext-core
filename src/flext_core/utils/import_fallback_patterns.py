"""Import fallback patterns for flext_core."""
import importlib
from typing import Any


class DependencyWrapper:
    """Wrapper for optional dependencies with fallback patterns."""

    def __init__(self, available: bool) -> None:
        self.available = available

    def try_import(self, module_name: str, component: str | None = None) -> Any:
        """Try to import a module or component."""
        if not self.available:
            return None

        try:
            module = importlib.import_module(module_name)
            if component:
                return getattr(module, component)
        except (ImportError, AttributeError):
            return None
        else:
            return module


# SQLAlchemy dependency fallback
try:
    import sqlalchemy  # noqa: F401
    SQLALCHEMY_DEPENDENCY = DependencyWrapper(True)
except ImportError:
    SQLALCHEMY_DEPENDENCY = DependencyWrapper(False)

# Redis dependency fallback
try:
    import redis  # noqa: F401
    REDIS_DEPENDENCY = DependencyWrapper(True)
except ImportError:
    REDIS_DEPENDENCY = DependencyWrapper(False)

# Prometheus dependency fallback
try:
    import prometheus_client  # noqa: F401
    PROMETHEUS_DEPENDENCY = DependencyWrapper(True)
except ImportError:
    PROMETHEUS_DEPENDENCY = DependencyWrapper(False)

# Kubernetes dependency fallback
try:
    import kubernetes  # noqa: F401  # type: ignore[import-not-found]
    KUBERNETES_DEPENDENCY = DependencyWrapper(True)
except ImportError:
    KUBERNETES_DEPENDENCY = DependencyWrapper(False)


def get_kubernetes_client() -> Any:
    """Get Kubernetes client if available."""
    if KUBERNETES_DEPENDENCY.available:
        client = KUBERNETES_DEPENDENCY.try_import("kubernetes.client")
        config = KUBERNETES_DEPENDENCY.try_import("kubernetes.config")
        return client, config
    return None, None
