# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Docs package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from tests.infra.unit.docs.auditor import (
        TestAuditorBudgets,
        TestAuditorCore,
        TestAuditorNormalize,
    )
    from tests.infra.unit.docs.auditor_cli import (
        TestAuditorMainCli,
        TestAuditorScopeFailure,
    )
    from tests.infra.unit.docs.auditor_links import (
        TestAuditorBrokenLinks,
        TestAuditorToMarkdown,
    )
    from tests.infra.unit.docs.auditor_scope import (
        TestAuditorForbiddenTerms,
        TestAuditorScope,
    )
    from tests.infra.unit.docs.builder import TestBuilderCore
    from tests.infra.unit.docs.builder_scope import TestBuilderScope
    from tests.infra.unit.docs.fixer import TestFixerCore
    from tests.infra.unit.docs.fixer_internals import (
        TestFixerMaybeFixLink,
        TestFixerProcessFile,
        TestFixerScope,
        TestFixerToc,
    )
    from tests.infra.unit.docs.generator import TestGeneratorCore
    from tests.infra.unit.docs.generator_internals import (
        TestGeneratorHelpers,
        TestGeneratorScope,
    )
    from tests.infra.unit.docs.init import TestFlextInfraDocs
    from tests.infra.unit.docs.main import TestRunAudit, TestRunFix
    from tests.infra.unit.docs.main_commands import (
        TestRunBuild,
        TestRunGenerate,
        TestRunValidate,
    )
    from tests.infra.unit.docs.main_entry import TestMainRouting, TestMainWithFlags
    from tests.infra.unit.docs.shared import (
        TestFlextInfraDocScope,
        TestFlextInfraDocsShared,
    )
    from tests.infra.unit.docs.validator import TestFlextInfraDocValidator

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestAuditorBrokenLinks": (
        "tests.infra.unit.docs.auditor_links",
        "TestAuditorBrokenLinks",
    ),
    "TestAuditorBudgets": ("tests.infra.unit.docs.auditor", "TestAuditorBudgets"),
    "TestAuditorCore": ("tests.infra.unit.docs.auditor", "TestAuditorCore"),
    "TestAuditorForbiddenTerms": (
        "tests.infra.unit.docs.auditor_scope",
        "TestAuditorForbiddenTerms",
    ),
    "TestAuditorMainCli": ("tests.infra.unit.docs.auditor_cli", "TestAuditorMainCli"),
    "TestAuditorNormalize": ("tests.infra.unit.docs.auditor", "TestAuditorNormalize"),
    "TestAuditorScope": ("tests.infra.unit.docs.auditor_scope", "TestAuditorScope"),
    "TestAuditorScopeFailure": (
        "tests.infra.unit.docs.auditor_cli",
        "TestAuditorScopeFailure",
    ),
    "TestAuditorToMarkdown": (
        "tests.infra.unit.docs.auditor_links",
        "TestAuditorToMarkdown",
    ),
    "TestBuilderCore": ("tests.infra.unit.docs.builder", "TestBuilderCore"),
    "TestBuilderScope": ("tests.infra.unit.docs.builder_scope", "TestBuilderScope"),
    "TestFixerCore": ("tests.infra.unit.docs.fixer", "TestFixerCore"),
    "TestFixerMaybeFixLink": (
        "tests.infra.unit.docs.fixer_internals",
        "TestFixerMaybeFixLink",
    ),
    "TestFixerProcessFile": (
        "tests.infra.unit.docs.fixer_internals",
        "TestFixerProcessFile",
    ),
    "TestFixerScope": ("tests.infra.unit.docs.fixer_internals", "TestFixerScope"),
    "TestFixerToc": ("tests.infra.unit.docs.fixer_internals", "TestFixerToc"),
    "TestFlextInfraDocScope": (
        "tests.infra.unit.docs.shared",
        "TestFlextInfraDocScope",
    ),
    "TestFlextInfraDocValidator": (
        "tests.infra.unit.docs.validator",
        "TestFlextInfraDocValidator",
    ),
    "TestFlextInfraDocs": ("tests.infra.unit.docs.init", "TestFlextInfraDocs"),
    "TestFlextInfraDocsShared": (
        "tests.infra.unit.docs.shared",
        "TestFlextInfraDocsShared",
    ),
    "TestGeneratorCore": ("tests.infra.unit.docs.generator", "TestGeneratorCore"),
    "TestGeneratorHelpers": (
        "tests.infra.unit.docs.generator_internals",
        "TestGeneratorHelpers",
    ),
    "TestGeneratorScope": (
        "tests.infra.unit.docs.generator_internals",
        "TestGeneratorScope",
    ),
    "TestMainRouting": ("tests.infra.unit.docs.main_entry", "TestMainRouting"),
    "TestMainWithFlags": ("tests.infra.unit.docs.main_entry", "TestMainWithFlags"),
    "TestRunAudit": ("tests.infra.unit.docs.main", "TestRunAudit"),
    "TestRunBuild": ("tests.infra.unit.docs.main_commands", "TestRunBuild"),
    "TestRunFix": ("tests.infra.unit.docs.main", "TestRunFix"),
    "TestRunGenerate": ("tests.infra.unit.docs.main_commands", "TestRunGenerate"),
    "TestRunValidate": ("tests.infra.unit.docs.main_commands", "TestRunValidate"),
}

__all__ = [
    "TestAuditorBrokenLinks",
    "TestAuditorBudgets",
    "TestAuditorCore",
    "TestAuditorForbiddenTerms",
    "TestAuditorMainCli",
    "TestAuditorNormalize",
    "TestAuditorScope",
    "TestAuditorScopeFailure",
    "TestAuditorToMarkdown",
    "TestBuilderCore",
    "TestBuilderScope",
    "TestFixerCore",
    "TestFixerMaybeFixLink",
    "TestFixerProcessFile",
    "TestFixerScope",
    "TestFixerToc",
    "TestFlextInfraDocScope",
    "TestFlextInfraDocValidator",
    "TestFlextInfraDocs",
    "TestFlextInfraDocsShared",
    "TestGeneratorCore",
    "TestGeneratorHelpers",
    "TestGeneratorScope",
    "TestMainRouting",
    "TestMainWithFlags",
    "TestRunAudit",
    "TestRunBuild",
    "TestRunFix",
    "TestRunGenerate",
    "TestRunValidate",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
