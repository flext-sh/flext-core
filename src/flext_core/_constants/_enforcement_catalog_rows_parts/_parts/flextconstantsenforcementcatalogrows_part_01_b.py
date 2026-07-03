"""Pattern/typing enforcement catalog rows."""

from __future__ import annotations

from typing import Final

INFRA_DETECTOR_ROWS_PATTERNS: Final[
    tuple[tuple[str, str, str, str, tuple[str, ...], bool, str], ...]
] = (
    (
        "ENFORCE-026",
        "HIGH",
        "bare_except_violations",
        "3-1-supreme-law",
        ("flext-patterns",),
        False,
        "Bare `except:` clause swallows all exceptions including SystemExit.",
    ),
    (
        "ENFORCE-027",
        "MEDIUM",
        "print_violations",
        "3-1-supreme-law",
        ("flext-patterns",),
        False,
        "`print()` call in source code — use structured logging.",
    ),
    (
        "ENFORCE-028",
        "HIGH",
        "breakpoint_violations",
        "3-1-supreme-law",
        ("flext-patterns",),
        False,
        "`breakpoint()` / pdb left in code.",
    ),
    (
        "ENFORCE-029",
        "MEDIUM",
        "open_encoding_violations",
        "3-1-supreme-law",
        ("flext-patterns",),
        False,
        "`open()` without explicit encoding.",
    ),
    (
        "ENFORCE-030",
        "HIGH",
        "dict_annotation_violations",
        "3-2-types-and-contracts",
        ("flext-strict-typing",),
        False,
        "`dict` in type annotation — prefer Mapping / MutableMapping / TypedDict.",
    ),
    (
        "ENFORCE-031",
        "HIGH",
        "typing_dict_attr_violations",
        "3-2-types-and-contracts",
        ("flext-strict-typing",),
        False,
        "`typing.Dict` attribute usage — use collections.abc.Mapping family.",
    ),
    (
        "ENFORCE-032",
        "HIGH",
        "typing_dict_import_violations",
        "3-2-types-and-contracts",
        ("flext-strict-typing",),
        False,
        "`from typing import Dict` — banned in favor of dict / Mapping.",
    ),
    (
        "ENFORCE-033",
        "HIGH",
        "hardcoded_version_violations",
        "3-4-tools-and-modules",
        ("flext-patterns",),
        False,
        "Hardcoded `__version__` string — use importlib.metadata.",
    ),
    (
        "ENFORCE-081",
        "HIGH",
        "inline_import_violations",
        "4-import-law",
        ("flext-import-rules", "flext-patterns"),
        False,
        "Inline or lazy import declared inside a function body or dynamic importlib.import_module call.",
    ),
    (
        "ENFORCE-082",
        "HIGH",
        "silent_failure_violations",
        "3-1-supreme-law",
        ("flext-patterns",),
        False,
        "Exception-silencing pattern (contextlib.suppress, except...: pass, broad except, unwrap_or sentinel, or sentinel return on failure branch).",
    ),
    (
        "ENFORCE-083",
        "HIGH",
        "type_ignore_violations",
        "3-1-supreme-law",
        ("flext-strict-typing",),
        True,
        "`# type: ignore` comment silences type checker — remove the bypass and fix the type error.",
    ),
    (
        "ENFORCE-084",
        "HIGH",
        "noqa_violations",
        "3-1-supreme-law",
        ("flext-patterns",),
        True,
        "`# noqa` comment silences lint — remove the bypass and fix the underlying issue.",
    ),
)


__all__ = ["INFRA_DETECTOR_ROWS_PATTERNS"]
