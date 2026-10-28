"""Constants mixin for other.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from flext_core import FlextConstants


class TestsFlextConstantsOther:
    PUBLIC_ALIAS_NAMES: ClassVar[tuple[str, ...]] = (
        "Primitives",
        "Scalar",
        "Numeric",
        "JsonValue",
        "JsonMapping",
        "JsonList",
        "Mapping[str, str]",
        "Sequence[str]",
        "Mapping[str, Scalar]",
        "SecretValue",
        "SettingsValue",
        "IntPair",
        "StrictStr",
        "StrictInt",
        "StrictFloat",
        "StrictBytes",
        "TextOrBinaryContent",
        "RegistryBindingKey",
        "JsonValue",
        "FileContent",
        "JsonMapping",
        "JsonList",
        "ConfigurationMapping",
        "DispatchableHandler",
        "FactoryCallable",
        "HandlerCallable",
        "JsonPayload",
        "JsonValue",
        "MessageTypeSpecifier",
        "RegisterableService",
        "ResourceCallable",
        "JsonMapping",
        "ScalarOrModel",
        "ServiceMap",
        "SortableObjectType",
        "TypeHintSpecifier",
    )
    FLAT_ALIAS_NAMES: ClassVar[tuple[str, ...]] = (
        "AttributeMapping",
        "MutableAttributeMapping",
        "ConfigValueMapping",
        "OptionalStrMapping",
        "MutableOptionalStrMapping",
        "HeaderMapping",
        "FeatureFlagMapping",
        "MutableFeatureFlagMapping",
        "OptionalBoolMapping",
        "MutableOptionalBoolMapping",
    )
    LEGACY_GENERIC_NAMES: ClassVar[tuple[str, ...]] = (
        "EnumT",
        "MessageT_contra",
        "P",
        "R",
        "ResultT",
        "RootValueT",
        "T",
        "T_DomainResult",
        "T_Model",
        "T_Namespace",
        "T_Settings",
        "T_co",
        "T_contra",
        "TRuntime",
        "U",
        "TV",
        "TV_co",
    )

    EXAMPLE_CHECK_COUNT_PATTERN: ClassVar[str] = r"\((\d+) checks\)"

    DOCUMENTED_COMBINED_SUM: ClassVar[int] = 6
    DOCUMENTED_FLAT_MAP_VALUE: ClassVar[int] = 20
    DOCUMENTED_FLOW_VALUE: ClassVar[int] = 2
    DOCUMENTED_INCREMENTED_VALUE: ClassVar[int] = 4
    DOCUMENTED_RECOVERED_VALUE: ClassVar[int] = 99

    NON_GOLDEN_EXAMPLE_MODULES: ClassVar[Sequence[str]] = (
        "examples.ex_05_flext_mixins",
        "examples.ex_06_flext_context",
    )

    PUBLIC_EXAMPLES: ClassVar[tuple[tuple[str, str, str], ...]] = (
        ("ex_01_flext_result", "examples.ex_01_flext_result", "ex_01_flext_result.py"),
        (
            "ex_02_flext_settings",
            "examples.ex_02_flext_settings",
            "ex_02_flext_settings.py",
        ),
        ("ex_03_flext_logger", "examples.ex_03_flext_logger", "ex_03_flext_logger.py"),
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
            "ex_08_flext_container",
            "examples.ex_08_flext_container",
            "ex_08_flext_container.py",
        ),
        (
            "ex_11_flext_service",
            "examples.ex_11_flext_service",
            "ex_11_flext_service.py",
        ),
    )

    LAZY_BENCHMARK_REAL_SYMBOLS: ClassVar[Sequence[str]] = (
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
    LAZY_BENCHMARK_EXTRA_INSTALL_MAPS: ClassVar[tuple[dict[str, str], ...]] = (
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

    CORE_PACKAGE_NAME: ClassVar[str] = "flext-core"
    PACKAGE_INFO_REQUIRED_KEYS: ClassVar[frozenset[str]] = frozenset({
        "name",
        "version",
        "description",
        "author",
        "author_email",
        "license",
        "url",
    })
    AT_LEAST_CASES: ClassVar[tuple[tuple[int, int, int, bool], ...]] = (
        (0, 0, 0, True),
        (999, 0, 0, False),
    )
    AT_LEAST_CASE_IDS: ClassVar[Sequence[str]] = (
        "at-or-above-zero",
        "below-impossibly-high-major",
    )

    CORE_SAFE_STRING_VALID_CASES: ClassVar[tuple[tuple[str, str], ...]] = (
        ("hello", "hello"),
        ("  hello  ", "hello"),
        ("hello\t", "hello"),
        ("  ola mundo  ", "ola mundo"),
        ("invalid-name", "invalid-name"),
        ("invalid.name", "invalid.name"),
        ("invalid name", "invalid name"),
        ("123start", "123start"),
        ("invalid@symbol", "invalid@symbol"),
    )
    CORE_SAFE_STRING_INVALID_CASES: ClassVar[tuple[tuple[str | None, str], ...]] = (
        (None, FlextConstants.ERR_TEXT_NONE_NOT_ALLOWED),
        ("", FlextConstants.ERR_TEXT_EMPTY_NOT_ALLOWED),
        ("   ", FlextConstants.ERR_TEXT_EMPTY_NOT_ALLOWED),
        ("\t", FlextConstants.ERR_TEXT_EMPTY_NOT_ALLOWED),
    )
    CORE_FORMAT_APP_ID_CASES: ClassVar[tuple[tuple[str, str], ...]] = (
        ("MyApp", "myapp"),
        ("My App", "my-app"),
        ("my_app", "my-app"),
        ("My Application_Name", "my-application-name"),
        ("FlextApp", "flextapp"),
        ("flext-app", "flext-app"),
        ("FLEXT_APP", "flext-app"),
        ("flext_app_v2", "flext-app-v2"),
    )


__all__: list[str] = ["TestsFlextConstantsOther"]
