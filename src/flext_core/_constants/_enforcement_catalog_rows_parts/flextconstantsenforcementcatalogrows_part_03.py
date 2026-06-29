"""Static-tool enforcement catalog rows."""

from __future__ import annotations

from typing import Final


class FlextConstantsEnforcementCatalogToolRows:
    """Ruff, tests-validator, and ast-grep rows."""

    RUFF_ROWS: Final[tuple[tuple[str, str, str, tuple[str, ...], str], ...]] = (
        (
            "ENFORCE-023",
            "HIGH",
            "ANN401",
            ("flext-strict-typing",),
            "Dynamic Any usage (ruff ANN401) — enforced at lint time by `make lint`; listed here for cross-reference.",
        ),
        (
            "ENFORCE-024",
            "MEDIUM",
            "PGH003",
            ("flext-strict-typing",),
            "Missing specific rule code on pyright/pygrep suppressions (ruff PGH003).",
        ),
        (
            "ENFORCE-025",
            "MEDIUM",
            "TID252",
            ("flext-import-rules",),
            "Relative import (ruff TID252) — prefer absolute imports.",
        ),
    )

    TESTS_VALIDATOR_ROWS: Final[
        tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...], str], ...]
    ] = (
        (
            "ENFORCE-015",
            "HIGH",
            "imports",
            (
                "IMPORT-001",
                "IMPORT-002",
                "IMPORT-003",
                "IMPORT-004",
                "IMPORT-006",
            ),
            ("flext-import-rules",),
            "Import discipline violation — lazy imports, TYPE_CHECKING misuse, sys.path manipulation, or non-root flext-* imports.",
        ),
        (
            "ENFORCE-016",
            "HIGH",
            "types",
            ("TYPE-001", "TYPE-002", "TYPE-003"),
            ("flext-strict-typing", "flext-type-system"),
            "Type-system violation — Any/object/legacy typing or type: ignore bypass.",
        ),
        (
            "ENFORCE-017",
            "HIGH",
            "bypass",
            ("BYPASS-001", "BYPASS-002", "BYPASS-003"),
            ("flext-patterns",),
            "Bypass pattern — noqa, pragma: no cover (unapproved), or exception swallowing.",
        ),
        (
            "ENFORCE-018",
            "CRITICAL",
            "layer",
            ("LAYER-001",),
            ("flext-architecture-layers",),
            "Layer violation — lower layer importing an upper layer.",
        ),
        (
            "ENFORCE-019",
            "MEDIUM",
            "tests",
            ("TEST-001", "TEST-002", "TEST-003"),
            ("testing-patterns",),
            "Test pattern violation — monkeypatch, Mock/MagicMock, or @patch usage.",
        ),
        (
            "ENFORCE-020",
            "HIGH",
            "validate_config",
            ("CONFIG-001", "CONFIG-003", "CONFIG-004", "CONFIG-005"),
            ("flext-development-workflow",),
            "pyproject.toml deviation — mypy ignore_errors, unapproved ruff ignores, or incomplete type strictness.",
        ),
        (
            "ENFORCE-021",
            "MEDIUM",
            "markdown",
            ("MD-001", "MD-002", "MD-003", "MD-004"),
            ("testing-patterns",),
            "Markdown code block validation — syntax, forbidden typings, missing future annotations, object as type.",
        ),
    )

    AST_GREP_ROWS: Final[tuple[tuple[str, str, str, str, str], ...]] = (
        (
            "ENFORCE-026",
            "HIGH",
            "flext-patterns",
            "ban-bare-except",
            "Bare `except:` clause swallows all exceptions including SystemExit.",
        ),
        (
            "ENFORCE-027",
            "MEDIUM",
            "flext-patterns",
            "ban-print-in-src",
            "`print()` call in source code — use structured logging.",
        ),
        (
            "ENFORCE-028",
            "HIGH",
            "flext-patterns",
            "no-breakpoint",
            "`breakpoint()` / pdb left in code.",
        ),
        (
            "ENFORCE-029",
            "MEDIUM",
            "flext-patterns",
            "ban-open-no-encoding",
            "`open()` without explicit encoding.",
        ),
        (
            "ENFORCE-030",
            "HIGH",
            "flext-strict-typing",
            "ban-dict-type-annotation",
            "`dict` in type annotation — prefer Mapping / MutableMapping / TypedDict.",
        ),
        (
            "ENFORCE-031",
            "HIGH",
            "flext-strict-typing",
            "ban-typing-dict-attribute",
            "`typing.Dict` attribute usage — use collections.abc.Mapping family.",
        ),
        (
            "ENFORCE-032",
            "HIGH",
            "flext-strict-typing",
            "ban-typing-dict-import",
            "`from typing import Dict` — banned in favor of dict / Mapping.",
        ),
        (
            "ENFORCE-033",
            "HIGH",
            "flext-patterns",
            "no-hardcoded-version-string",
            "Hardcoded `__version__` string — use importlib.metadata.",
        ),
    )


__all__ = ["FlextConstantsEnforcementCatalogToolRows"]
