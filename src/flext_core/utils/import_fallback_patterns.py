"""Import fallback patterns for optional dependencies.

This module provides graceful handling of optional dependencies with
fallback implementations for development and production environments.
"""

import importlib
import logging
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DependencyFallback:
    """Handles optional dependency imports with fallbacks."""

    def __init__(self, dependency_name: str, required: bool = False) -> None:
        self.dependency_name = dependency_name
        self.required = required
        self._module_cache: dict[str, Any] = {}

    def try_import(self, module_path: str, component: Optional[str] = None) -> Any:
        """Try to import a module or component with fallback.

        Args:
            module_path: The module path to import
            component: Optional specific component to import

        Returns:
            The imported module/component or a fallback implementation

        """
        cache_key = f"{module_path}.{component}" if component else module_path

        if cache_key in self._module_cache:
            return self._module_cache[cache_key]

        try:
            module = importlib.import_module(module_path)

            result = getattr(module, component) if component else module

            self._module_cache[cache_key] = result
            logger.debug(f"Successfully imported {cache_key}")
            return result

        except (ImportError, AttributeError) as e:
            if self.required:
                logger.error(
                    f"Required dependency {self.dependency_name} not found: {e}"
                )
                raise ImportError(
                    f"Required dependency '{self.dependency_name}' is not installed. "
                    f"Install it with: pip install {self.dependency_name}"
                ) from e

            # Return fallback implementation
            fallback = self._create_fallback(cache_key)
            self._module_cache[cache_key] = fallback
            logger.warning(
                f"Optional dependency {self.dependency_name} not found, using fallback for {cache_key}"
            )
            return fallback

    def _create_fallback(self, cache_key: str) -> Callable[..., Any]:
        """Create a fallback implementation for missing dependencies."""
        if "make_url" in cache_key:
            # SQLAlchemy URL fallback
            def fallback_make_url(url_string: str) -> Any:
                """Fallback URL parser for SQLAlchemy compatibility."""

                class FallbackURL:
                    def __init__(self, url: str) -> None:
                        self.url = url
                        self.drivername = "sqlite"
                        self.host = None
                        self.port = None
                        self.database = ":memory:"

                    def __str__(self) -> str:
                        return self.url

                return FallbackURL(url_string)

            return fallback_make_url

        # Generic fallback - return a no-op function
        def generic_fallback(*args: Any, **kwargs: Any) -> None:
            logger.warning(f"Using fallback implementation for {cache_key}")
            return None

        return generic_fallback


# Pre-configured dependency handlers
SQLALCHEMY_DEPENDENCY = DependencyFallback("sqlalchemy", required=False)
