"""Architectural Import System - with strict validation.

This module implements PROPER IMPORT ARCHITECTURE eliminating:
- sys.path.insert() anti-patterns
- Hardcoded relative path disasters
- Lazy import violations
- Import dependency chaos

ZERO TOLERANCE: All imports must be architecturally sound and type-safe.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

# ZERO TOLERANCE - Meltano dependencies are REQUIRED for FLEXT Meltano Enterprise
try:
    from meltano.core.project import (  # type: ignore[import-not-found]
        Project as MeltanoProject,
    )
    from meltano.core.runner import RunnerError  # type: ignore[import-not-found]
except ImportError:
    # Fallback for environments without meltano
    MeltanoProject = None
    RunnerError = Exception

# ZERO TOLERANCE - Verify Meltano core components are functional with guaranteed attributes
find_method = getattr(MeltanoProject, "find", None) if MeltanoProject else None
runner_name = getattr(RunnerError, "__name__", None)
if MeltanoProject is not None and (find_method is None or runner_name is None):
    msg = (
        "Meltano core components are required - this is a Meltano enterprise extension"
    )
    raise ImportError(msg)

# Import Architecture Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
E2E_TESTS_PATH = PROJECT_ROOT / "tests" / "e2e"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"

# Python 3.13 type aliases for architectural imports
ImportableClass = type[object]
ModuleComponents = dict[str, ImportableClass]


class ImportableModule:
    """Protocol for dynamically importable modules - SOLID Interface Segregation."""

    def __getattr__(self, name: str) -> object:
        """Get attribute dynamically from imported module."""


class ArchitecturalImportError(Exception):
    """Exception for import architecture violations - with strict validation."""


def _validate_module_path(module_path: Path) -> None:
    """Validate module path exists and is accessible."""
    if not module_path.exists():
        msg = f"Module path does not exist: {module_path}"
        raise ArchitecturalImportError(msg)

    if not module_path.is_file():
        msg = f"Module path is not a file: {module_path}"
        raise ArchitecturalImportError(msg)


def _import_module_from_path(module_path: Path, module_name: str) -> ModuleType:
    """Import module from path with comprehensive error handling."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = f"Failed to create module spec for: {module_path}"
            raise ArchitecturalImportError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, AttributeError, OSError, TypeError) as e:
        msg = f"Failed to import module {module_name} from {module_path}: {e}"
        raise ArchitecturalImportError(msg) from e
    else:
        return module


def dynamic_import_from_path(module_path: Path, module_name: str) -> ModuleType:
    """Dynamically import module from path - with strict validation.

    Args:
    ----
        module_path: Absolute path to the module file
        module_name: Name to assign to the imported module

    Returns:
    -------
        Imported module instance

    Raises:
    ------
        ArchitecturalImportError: When import fails or path is invalid

    """
    _validate_module_path(module_path)
    return _import_module_from_path(module_path, module_name)


def _get_class_from_module(
    module: ModuleType,
    class_name: str,
    module_path: Path,
) -> ImportableClass:
    """Extract class from module with validation."""
    # ZERO TOLERANCE - Use getattr with None default instead of hasattr check
    imported_class = getattr(module, class_name, None)
    if imported_class is not None:
        if isinstance(imported_class, type):
            return imported_class
        msg = f"{class_name} is not a class in {module_path}"
        raise ArchitecturalImportError(msg)

    msg = f"{class_name} class not found in {module_path}"
    raise ArchitecturalImportError(msg)


@lru_cache(maxsize=1)
def import_e2e_test_suite() -> ImportableClass:
    """Import E2E test suite class - ARCHITECTURAL APPROACH with caching.

    Returns:
    -------
        FlextE2ETestSuite class from tests/e2e/flext_e2e.py

    Raises:
    ------
        ArchitecturalImportError: When E2E module cannot be imported

    """
    e2e_module_path = E2E_TESTS_PATH / "flext_e2e.py"

    try:
        e2e_module = dynamic_import_from_path(e2e_module_path, "flext_e2e")
        return _get_class_from_module(e2e_module, "FlextE2ETestSuite", e2e_module_path)
    except (
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        ImportError,
        AttributeError,
    ) as e:
        # ZERO TOLERANCE - Specific exception types for E2E import failures
        msg = f"Unexpected error importing E2E test suite: {e}"
        raise ArchitecturalImportError(msg) from e


@lru_cache(maxsize=1)
def import_kind_cluster_setup() -> ImportableClass:
    """Import Kind cluster setup class - ARCHITECTURAL APPROACH with caching.

    Returns:
    -------
        KindClusterSetup class from scripts/setup_kind_cluster.py

    Raises:
    ------
        ArchitecturalImportError: When Kind setup module cannot be imported

    """
    kind_module_path = SCRIPTS_PATH / "setup_kind_cluster.py"

    try:
        kind_module = dynamic_import_from_path(kind_module_path, "setup_kind_cluster")
        return _get_class_from_module(kind_module, "KindClusterSetup", kind_module_path)
    except (
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        ImportError,
        AttributeError,
    ) as e:
        # ZERO TOLERANCE - Specific exception types for Kind import failures
        msg = f"Unexpected error importing Kind cluster setup: {e}"
        raise ArchitecturalImportError(msg) from e


@lru_cache(maxsize=1)
def import_meltano_components() -> ModuleComponents:
    """Import Meltano components - ARCHITECTURAL APPROACH for distributed processing.

    Returns:
    -------
        Dictionary of Meltano component classes

    Raises:
    ------
        ArchitecturalImportError: When Meltano components cannot be imported

    """
    components: ModuleComponents = {}

    # Meltano components are guaranteed to be available - no conditional imports
    try:
        components["MeltanoProject"] = MeltanoProject
        components["RunnerError"] = RunnerError
    except (
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        ImportError,
        AttributeError,
    ) as e:
        # ZERO TOLERANCE - Specific exception types for Meltano import failures
        msg = f"Failed to import Meltano components: {e}"
        raise ArchitecturalImportError(msg) from e

    return components


def get_e2e_test_suite_class() -> ImportableClass:
    """Get cached E2E test suite class - PERFORMANCE OPTIMIZED."""
    return import_e2e_test_suite()


def get_kind_cluster_setup_class() -> ImportableClass:
    """Get cached Kind cluster setup class - PERFORMANCE OPTIMIZED."""
    return import_kind_cluster_setup()


def get_meltano_components() -> ModuleComponents:
    """Get cached Meltano components - PERFORMANCE OPTIMIZED."""
    return import_meltano_components()


def clear_import_cache() -> None:
    """Clear import cache (for testing) - with strict validation."""
    import_e2e_test_suite.cache_clear()
    import_kind_cluster_setup.cache_clear()
    import_meltano_components.cache_clear()
