"""Constants mixin for other.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Sequence,
)
from typing import ClassVar, Final


class TestsFlextCoreConstantsOther:
    class Architecture:
        """Architecture-test constants for flext-core validators."""

        VALIDATOR_METHODS: Final[tuple[str, ...]] = (
            "imports",
            "types",
            "bypass",
            "layer",
        )

    class EnforcementCatalog:
        """Shared enforcement-catalog literals for flext-core tests."""

        RULE_ID_PATTERN: Final[str] = r"^ENFORCE-\d{3}$"
        MISSING_RULE_ID: Final[str] = "ENFORCE-999"
        DUPLICATE_RULE_ID: Final[str] = "ENFORCE-900"
        INFRA_RULE_ID: Final[str] = "ENFORCE-901"
        RUFF_RULE_ID: Final[str] = "ENFORCE-902"
        INVALID_RULE_ID: Final[str] = "BAD-999"
        SAMPLE_DESCRIPTION: Final[str] = "x"
        INFRA_VIOLATION_FIELD: Final[str] = "loose_objects"
        RUNTIME_WARNING_CATEGORY: Final[str] = (
            "flext_core._constants.enforcement.FlextMroViolation"
        )
        RUFF_RULE_CODE_ANN401: Final[str] = "ANN401"
        RUFF_RULE_CODE_PGH003: Final[str] = "PGH003"
        VALIDATOR_METHODS: Final[tuple[str, ...]] = (
            "imports",
            "types",
            "bypass",
            "layer",
            "tests",
            "validate_config",
            "markdown",
        )

    class ExamplesExecution:
        """Shared public-example metadata for execution tests."""

        PUBLIC_EXAMPLES: Final[tuple[tuple[str, str, str], ...]] = (
            (
                "ex_01_flext_result",
                "examples.ex_01_flext_result",
                "ex_01_flext_result.py",
            ),
            (
                "ex_02_flext_settings",
                "examples.ex_02_flext_settings",
                "ex_02_flext_settings.py",
            ),
            (
                "ex_03_flext_logger",
                "examples.ex_03_flext_logger",
                "ex_03_flext_logger.py",
            ),
            (
                "ex_04_flext_dispatcher",
                "examples.ex_04_flext_dispatcher",
                "ex_04_flext_dispatcher.py",
            ),
            (
                "ex_11_flext_service",
                "examples.ex_11_flext_service",
                "ex_11_flext_service.py",
            ),
        )

    class LazyBenchmark:
        """Shared benchmark fixtures for lazy export tests."""

        REAL_SYMBOLS: Final[tuple[str, ...]] = (
            "FlextConstants",
            "FlextContainer",
            "FlextContext",
            "FlextModels",
            "FlextProtocols",
            "FlextService",
            "FlextSettings",
            "FlextUtilities",
            "c",
            "m",
            "p",
            "r",
            "t",
            "u",
        )
        EXTRA_INSTALL_MAPS: Final[tuple[dict[str, str], ...]] = (
            {
                "_types": ".typings:FlextTypes",
                "_models": ".models:FlextModels",
                "_utils": ".utilities:FlextUtilities",
            },
            {
                "_svc": ".service:FlextService",
                "_container": ".container:FlextContainer",
                "_context": ".context:FlextContext",
            },
            {
                "_const": ".constants:FlextConstants",
                "_proto": ".protocols:FlextProtocols",
                "_result": ".result:r",
            },
        )

    class Version:
        """Version-test constants for flext-core public API checks."""

        CORE_PACKAGE_NAME: Final[str] = "flext-core"
        PACKAGE_INFO_REQUIRED_KEYS: Final[tuple[str, ...]] = (
            "name",
            "version",
            "description",
            "author",
            "license",
            "url",
        )
        AT_LEAST_CASES: Final[tuple[tuple[int, int, int, bool], ...]] = (
            (0, 0, 0, True),
            (999, 999, 999, False),
        )
        AT_LEAST_CASE_IDS: Final[tuple[str, ...]] = (
            "zero_succeeds",
            "unreachable_future_fails",
        )

    SAFE_STRING_VALID_CASES: ClassVar[Sequence[tuple[str, str]]] = [
        ("hello", "hello"),
        ("  hello  ", "hello"),
        ("hello\t", "hello"),
        ("  olá mundo  ", "olá mundo"),
    ]
    SAFE_STRING_INVALID_CASES: ClassVar[Sequence[tuple[str | None, str]]] = [
        (None, "Text cannot be None"),
        ("", "empty or whitespace-only"),
        ("   ", "empty or whitespace-only"),
        ("\t", "empty or whitespace-only"),
    ]
    FORMAT_APP_ID_CASES: ClassVar[Sequence[tuple[str, str]]] = [
        ("MyApp", "myapp"),
        ("My App", "my-app"),
        ("my_app", "my-app"),
        ("My Application_Name", "my-application-name"),
    ]
