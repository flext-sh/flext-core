"""Canonical IO for generated initializer artifact sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, u
from flext_infra.codegen.codegen_generation import FlextInfraCodegenGeneration

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import m, t


# flext-wkii.17.26 (codex): IO owns one initializer and removes obsolete sidecars.
class FlextInfraCodegenLazyInitGenerationIOMixin:
    """Compare, write, and remove generated package initializers."""

    if TYPE_CHECKING:
        repository_root: Path
        _modified_files: t.Infra.StrSet

        def _cleanup_generated_support_files(
            self, plan: m.Infra.LazyInitPlan, *, check_only: bool = False
        ) -> int: ...

        @staticmethod
        def _read_generated_file(path: Path) -> str | None: ...

    def _read_previous_init(self, plan: m.Infra.LazyInitPlan) -> str | None:
        """Read existing initializer content when present."""
        return self._read_generated_file(plan.context.init_path)

    def _write_generated_file(
        self, path: Path, generated: str, previous: str | None
    ) -> None:
        """Atomically write one changed canonical generated Python file."""
        if previous == generated:
            return
        write_result = u.Cli.atomic_write_text_file(path, generated)
        if write_result.failure:
            message = f"writing {path}: {write_result.error}"
            raise OSError(message)
        self._modified_files.add(str(path))

    def _check_remove_init(
        self, plan: m.Infra.LazyInitPlan
    ) -> t.Infra.LazyInitWriteResult:
        """Record generated artifact removals without writing."""
        init_path = plan.context.init_path
        if init_path.is_file():
            self._modified_files.add(str(init_path))
        self._cleanup_generated_support_files(plan, check_only=True)
        return (0, dict(plan.lazy_map))

    def _remove_init(self, plan: m.Infra.LazyInitPlan) -> t.Infra.LazyInitWriteResult:
        """Remove the initializer and every obsolete generated sidecar."""
        init_path = plan.context.init_path
        try:
            if init_path.is_file():
                init_path.unlink()
                self._modified_files.add(str(init_path))
            cleanup_exit = self._cleanup_generated_support_files(plan)
        except OSError as exc:
            u.Cli.error(f"removing generated init {init_path}: {exc}")
            return (-1, dict(plan.lazy_map))
        if cleanup_exit < 0:
            return (-1, dict(plan.lazy_map))
        return (0, dict(plan.lazy_map))

    def _write_init(self, plan: m.Infra.LazyInitPlan) -> t.Infra.LazyInitWriteResult:
        """Write one initializer after removing obsolete generated sidecars."""
        init_path = plan.context.init_path
        try:
            generated = FlextInfraCodegenGeneration.render_init(plan)
            previous = self._read_previous_init(plan)
        except c.EXC_OS_VALUE as exc:
            u.Cli.error(f"generating {init_path}: {exc}")
            return (-1, dict(plan.lazy_map))
        cleanup_exit = self._cleanup_generated_support_files(plan)
        if cleanup_exit < 0:
            return (-1, dict(plan.lazy_map))
        try:
            self._write_generated_file(init_path, generated, previous)
        except OSError as exc:
            u.Cli.error(f"writing generated init {init_path}: {exc}")
            return (-1, dict(plan.lazy_map))
        return (0, dict(plan.lazy_map))

    def _check_write_init(
        self, plan: m.Infra.LazyInitPlan
    ) -> t.Infra.LazyInitWriteResult:
        """Compare the initializer and obsolete sidecars without mutating them."""
        init_path = plan.context.init_path
        try:
            generated = FlextInfraCodegenGeneration.render_init(plan)
            previous = self._read_previous_init(plan)
        except c.EXC_OS_VALUE as exc:
            u.Cli.error(f"checking generated init {init_path}: {exc}")
            return (-1, dict(plan.lazy_map))
        cleanup_exit = self._cleanup_generated_support_files(plan, check_only=True)
        if cleanup_exit < 0:
            return (-1, dict(plan.lazy_map))
        if previous != generated:
            self._modified_files.add(str(init_path))
        return (0, dict(plan.lazy_map))


__all__: list[str] = ["FlextInfraCodegenLazyInitGenerationIOMixin"]
