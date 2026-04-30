"""Constants mixin for other.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import ClassVar, Final


class TestsFlextConstantsOther:
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
            "ex_07_flext_exceptions",
            "examples.ex_07_flext_exceptions",
            "ex_07_flext_exceptions.py",
        ),
        (
            "ex_11_flext_service",
            "examples.ex_11_flext_service",
            "ex_11_flext_service.py",
        ),
    )

    VALIDATOR_METHODS: Final[tuple[str, ...]] = (
        "imports",
        "types",
        "bypass",
        "layer",
    )

    LAZY_BENCHMARK_REAL_SYMBOLS: Final[tuple[str, ...]] = (
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
    LAZY_BENCHMARK_EXTRA_INSTALL_MAPS: Final[tuple[dict[str, str], ...]] = (
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

    CORE_PACKAGE_NAME: Final[str] = "flext-core"
    PACKAGE_INFO_REQUIRED_KEYS: Final[frozenset[str]] = frozenset({
        "name",
        "version",
        "description",
        "author",
        "author_email",
        "license",
        "url",
    })
    AT_LEAST_CASES: Final[tuple[tuple[int, int, int, bool], ...]] = (
        (0, 0, 0, True),
        (999, 0, 0, False),
    )
    AT_LEAST_CASE_IDS: Final[tuple[str, ...]] = (
        "at-or-above-zero",
        "below-impossibly-high-major",
    )
    SEMVER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d+\.\d+\.\d+")

    SAFE_STRING_VALID_CASES: ClassVar[Sequence[tuple[str, str]]] = [
        ("hello", "hello"),
        ("  hello  ", "hello"),
        ("hello\t", "hello"),
        ("  ola mundo  ", "ola mundo"),
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
