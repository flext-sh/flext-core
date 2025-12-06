"""Tests for flext_core.__version__ module.

Tests cover FlextVersion class methods and module-level exports,
including version string, version info tuple, version comparison,
package info dictionary, and PackageNotFoundError fallback handling.

Uses flext_tests automation (tm, u) and pytest parametrization for
comprehensive real validation of all methods and edge cases.
"""

from __future__ import annotations

import importlib
import sys
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

from flext_core import __version__, __version_info__
from flext_core.__version__ import FlextVersion
from flext_tests import tm


class TestFlextVersion:
    """Test FlextVersion class methods and functionality with real validation."""

    def test_get_version_string(self) -> None:
        """Test get_version_string returns valid version string."""
        version = FlextVersion.get_version_string()
        # Use tm.that() for comprehensive validation
        tm.that(version, is_=str, none=False, empty=False)
        # Validate version format (semantic versioning)
        tm.that(
            version,
            match=r"^\d+\.\d+\.\d+",
            msg="Version must match semantic versioning",
        )

    def test_get_version_info(self) -> None:
        """Test get_version_info returns valid version tuple."""
        version_info = FlextVersion.get_version_info()
        # Validate tuple structure
        tm.that(version_info, is_=tuple, none=False, empty=False, len=(1, 10))
        # First element must be int (major version)
        tm.that(
            version_info[0],
            is_=int,
            gt=-1,
            msg="Major version must be non-negative integer",
        )
        # If tuple has 2+ elements, minor should be int
        if len(version_info) >= 2 and isinstance(version_info[1], int):
            tm.that(
                version_info[1], gt=-1, msg="Minor version must be non-negative integer"
            )

    @pytest.mark.parametrize(
        ("major", "minor", "patch", "expected"),
        [
            (0, 0, 0, True),  # Always at least 0.0.0
            (0, 0, 1, None),  # Depends on current version
            (1, 0, 0, None),  # Depends on current version
            (999, 999, 999, False),  # Future version should be False
            (-1, 0, 0, True),  # Negative major should still pass (tuple comparison)
        ],
    )
    def test_is_version_at_least(
        self,
        major: int,
        minor: int,
        patch: int,
        expected: bool | None,
    ) -> None:
        """Test is_version_at_least with various version combinations."""
        result = FlextVersion.is_version_at_least(major, minor, patch)
        tm.that(result, is_=bool, none=False)

        if expected is not None:
            tm.that(
                result,
                eq=expected,
                msg=f"Version comparison failed for {major}.{minor}.{patch}",
            )
        else:
            # For dynamic cases, validate against current version
            current_version_info = FlextVersion.get_version_info()
            current_major = (
                current_version_info[0]
                if isinstance(current_version_info[0], int)
                else 0
            )
            if major < current_major:
                tm.that(
                    result,
                    eq=True,
                    msg=f"Should be True for major {major} < current {current_major}",
                )
            elif major > current_major:
                tm.that(
                    result,
                    eq=False,
                    msg=f"Should be False for major {major} > current {current_major}",
                )

    def test_get_package_info(self) -> None:
        """Test get_package_info returns complete package metadata dictionary."""
        info = FlextVersion.get_package_info()
        # Validate dictionary structure
        tm.that(info, is_=dict, none=False, empty=False)
        # Validate required keys
        required_keys = [
            "name",
            "version",
            "description",
            "author",
            "author_email",
            "license",
            "url",
        ]
        tm.that(
            info, has=required_keys, msg="Package info must contain all required keys"
        )
        # Validate key types (all should be strings)
        for key in required_keys:
            tm.that(
                info[key],
                is_=str,
                none=False,
                msg=f"Key {key} must be non-empty string",
            )
        # Validate specific values
        tm.that(info["name"], eq="flext-core", msg="Package name must be flext-core")
        tm.that(
            info["version"],
            match=r"^\d+\.\d+\.\d+",
            msg="Version must match semantic versioning",
        )

    def test_module_level_exports(self) -> None:
        """Test module-level version exports are valid."""
        # Validate __version__ export
        tm.that(__version__, is_=str, none=False, empty=False, match=r"^\d+\.\d+\.\d+")
        # Validate __version_info__ export
        tm.that(__version_info__, is_=tuple, none=False, empty=False, len=(1, 10))
        # Validate consistency between exports and class methods
        tm.that(
            __version__,
            eq=FlextVersion.get_version_string(),
            msg="Module export must match class method",
        )
        tm.that(
            __version_info__,
            eq=FlextVersion.get_version_info(),
            msg="Module export must match class method",
        )

    def test_package_not_found_error_fallback(self) -> None:
        """Test PackageNotFoundError fallback handling with real module reload.

        Note: Module-level exception handlers are difficult to test as they
        execute at import time. This test verifies the fallback mechanism
        by temporarily removing the module from cache and reimporting with
        mocked metadata that raises PackageNotFoundError.
        """
        # Save original module reference for restoration
        module_name = "flext_core.__version__"
        original_module = sys.modules.get(module_name)

        # Remove module from cache to allow fresh import
        if module_name in sys.modules:
            del sys.modules[module_name]

        try:
            with patch(
                "importlib.metadata.metadata",
                side_effect=PackageNotFoundError("test"),
            ):
                # Import module fresh to trigger exception handler
                version_module = importlib.import_module(module_name)

                # Verify fallback values are set using tm.that()
                tm.that(
                    version_module.__version__,
                    eq="0.0.0-dev",
                    msg="Fallback version must be 0.0.0-dev",
                )
                tm.that(
                    version_module.__title__,
                    eq="flext-core",
                    msg="Fallback title must be flext-core",
                )
        finally:
            # Restore original module state
            if module_name in sys.modules:
                del sys.modules[module_name]
            if original_module is not None:
                sys.modules[module_name] = original_module
            else:
                # Reimport if no original existed
                importlib.import_module(module_name)

    @pytest.mark.parametrize(
        "method_name",
        [
            "get_version_string",
            "get_version_info",
            "get_package_info",
            "is_version_at_least",
        ],
    )
    def test_methods_are_callable(self, method_name: str) -> None:
        """Test that all class methods are callable and return valid results."""
        method = getattr(FlextVersion, method_name)
        # Verify method is callable
        tm.that(callable(method), eq=True, msg=f"{method_name} must be callable")
        # Verify method can be called and returns non-None
        if method_name == "is_version_at_least":
            result = method(0, 0, 0)
            tm.that(result, is_=bool, none=False, msg=f"{method_name} must return bool")
        elif method_name == "get_version_string":
            result = method()
            tm.that(
                result,
                is_=str,
                none=False,
                empty=False,
                msg=f"{method_name} must return non-empty string",
            )
        elif method_name == "get_version_info":
            result = method()
            tm.that(
                result,
                is_=tuple,
                none=False,
                empty=False,
                msg=f"{method_name} must return non-empty tuple",
            )
        elif method_name == "get_package_info":
            result = method()
            tm.that(
                result,
                is_=dict,
                none=False,
                empty=False,
                msg=f"{method_name} must return non-empty dict",
            )
