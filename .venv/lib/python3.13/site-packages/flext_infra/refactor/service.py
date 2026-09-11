"""Refactor service composition root for flext_infra.refactor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c
from flext_infra.refactor.loader import FlextInfraRefactorRuleLoader
from flext_infra.refactor.orchestrator import FlextInfraRefactorOrchestrator
from flext_infra.refactor.safety import FlextInfraRefactorSafetyManager

if TYPE_CHECKING:
    from flext_infra import m, p, t


class FlextInfraRefactorService:
    """Composition root wiring loader, orchestrator, and safety services."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the composed refactor services."""
        self.config_path = config_path or Path(__file__).parent / "settings.yml"
        self.rule_loader = FlextInfraRefactorRuleLoader(self.config_path)
        self.orchestrator = FlextInfraRefactorOrchestrator(
            self.rule_loader, safety_manager=FlextInfraRefactorSafetyManager()
        )

    def load_config(self) -> p.Result[t.MappingKV[str, t.Infra.InfraValue]]:
        """Delegate config loading to the dedicated refactor loader."""
        return self.rule_loader.load_config()

    def load_rules(
        self,
    ) -> p.Result[
        t.Infra.LoadedRuleSelections[
            c.Infra.RefactorRuleKind, c.Infra.RefactorFileRuleKind
        ]
    ]:
        """Delegate rule loading to the dedicated refactor loader."""
        return self.rule_loader.load_rules()

    def set_rule_filters(self, filters: t.StrSequence) -> None:
        """Delegate rule-filter normalization to the dedicated loader."""
        self.rule_loader.set_rule_filters(filters)

    def list_rules(self) -> t.SequenceOf[t.FeatureFlagMapping]:
        """Delegate rules listing to the dedicated loader."""
        return self.rule_loader.list_rules()

    @staticmethod
    def print_rules_table(rows: t.SequenceOf[t.FeatureFlagMapping]) -> None:
        """Delegate table rendering to the dedicated loader."""
        FlextInfraRefactorRuleLoader.print_rules_table(rows)

    def run_analyze_violations(self, args: p.Infra.RefactorCliArgs) -> int:
        """Delegate violation analysis to the dedicated orchestrator."""
        exit_code: int = self.orchestrator.run_analyze_violations(args)
        return exit_code

    def run_refactor(self, args: p.Infra.RefactorCliArgs) -> int:
        """Delegate CLI refactor execution to the dedicated orchestrator."""
        exit_code: int = self.orchestrator.run_refactor(args)
        return exit_code

    def refactor_file(
        self,
        file_path: Path,
        *,
        dry_run: bool = False,
        gates: t.StrSequence | None = None,
    ) -> m.Infra.Result:
        """Delegate single-file refactoring to the dedicated orchestrator."""
        return self.orchestrator.refactor_file(file_path, dry_run=dry_run, gates=gates)

    def refactor_files(
        self,
        file_paths: t.SequenceOf[Path],
        *,
        dry_run: bool = False,
        gates: t.StrSequence | None = None,
    ) -> t.SequenceOf[m.Infra.Result]:
        """Delegate multi-file refactoring to the dedicated orchestrator."""
        return self.orchestrator.refactor_files(
            file_paths, dry_run=dry_run, gates=gates
        )

    def refactor_project(
        self,
        project_path: Path,
        *,
        dry_run: bool = False,
        pattern: str = c.Infra.EXT_PYTHON_GLOB,
        apply_safety: bool = True,
        gates: t.StrSequence | None = None,
    ) -> t.SequenceOf[m.Infra.Result]:
        """Delegate project refactoring to the dedicated orchestrator."""
        return self.orchestrator.refactor_project(
            project_path,
            dry_run=dry_run,
            pattern=pattern,
            apply_safety=apply_safety,
            gates=gates,
        )

    def refactor_workspace(
        self,
        repository_root: Path,
        *,
        dry_run: bool = False,
        pattern: str = c.Infra.EXT_PYTHON_GLOB,
        apply_safety: bool = True,
        gates: t.StrSequence | None = None,
    ) -> t.SequenceOf[m.Infra.Result]:
        """Delegate workspace refactoring to the dedicated orchestrator."""
        return self.orchestrator.refactor_workspace(
            repository_root,
            dry_run=dry_run,
            pattern=pattern,
            apply_safety=apply_safety,
            gates=gates,
        )


__all__: list[str] = ["FlextInfraRefactorService"]
