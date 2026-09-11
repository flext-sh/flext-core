"""Lazy-init generated support-file cleanup."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, u
from flext_infra._utilities._sort_keys import path_depth

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import m, t


class FlextInfraCodegenLazyInitGenerationRegistryMixin:
    """Remove stale generated sidecars for package initializers."""

    if TYPE_CHECKING:
        _modified_files: t.Infra.StrSet

    def _cleanup_generated_support_files(
        self, plan: m.Infra.LazyInitPlan, *, check_only: bool = False
    ) -> int:
        """Remove generated files outside the canonical package artifact set."""
        try:
            # flext-wkii.17.26 (codex): __unit__.py is obsolete on every surface.
            self._remove_obsolete_generated_files(plan, check_only=check_only)
            self._remove_obsolete_root_support(plan, check_only=check_only)
            self._remove_generated_export_sidecars(plan, check_only=check_only)
            self._remove_generated_typing_stub(plan, check_only=check_only)
        except c.EXC_OS_VALUE as exc:
            u.Cli.error(
                f"cleaning generated sidecars for {plan.context.pkg_dir}: {exc}"
            )
            return -1
        return 0

    def _remove_obsolete_root_support(
        self, plan: m.Infra.LazyInitPlan, *, check_only: bool = False
    ) -> None:
        """Remove closed, preflighted root registries superseded by inline maps."""
        context = plan.context
        if (
            context.pkg_dir.parent.name != c.Infra.DEFAULT_SRC_DIR
            or "." in context.current_pkg
        ):
            return
        module_paths = tuple(
            context.pkg_dir / f"{name}.py"
            for name in sorted(c.Infra.OBSOLETE_ROOT_SUPPORT_NAMES)
        )
        package_paths = tuple(
            context.pkg_dir / name
            for name in sorted(c.Infra.OBSOLETE_ROOT_SUPPORT_NAMES)
        )
        stale_files: t.MutableSequenceOf[Path] = []
        stale_dirs: t.MutableSequenceOf[Path] = []
        for path in (*module_paths, *package_paths):
            if path.is_symlink():
                msg = f"refusing to remove obsolete root-support symlink: {path}"
                raise OSError(msg)
            if path.is_file():
                stale_files.append(path)
                continue
            if not path.exists():
                continue
            if not path.is_dir():
                msg = f"unexpected obsolete root-support path type: {path}"
                raise OSError(msg)
            stale_dirs.append(path)
            for child in sorted(path.rglob("*")):
                if child.is_symlink():
                    msg = f"refusing to remove obsolete root-support symlink: {child}"
                    raise OSError(msg)
                if child.is_dir():
                    if child.name != "__pycache__":
                        msg = f"unexpected directory in obsolete root support: {child}"
                        raise OSError(msg)
                    continue
                if not child.is_file() or child.suffix not in {".py", ".pyi", ".pyc"}:
                    msg = f"unexpected file in obsolete root support: {child}"
                    raise OSError(msg)
                stale_files.append(child)
        for path in stale_files:
            self._modified_files.add(str(path))
        if check_only:
            return
        for path in stale_files:
            path.unlink()
        for path in sorted(
            (
                child
                for stale_dir in stale_dirs
                for child in stale_dir.rglob("*")
                if child.is_dir()
            ),
            key=path_depth,
            reverse=True,
        ):
            path.rmdir()
        for path in stale_dirs:
            path.rmdir()

    def _remove_obsolete_generated_files(
        self, plan: m.Infra.LazyInitPlan, *, check_only: bool = False
    ) -> None:
        """Remove generated artifacts retired by the inline-root contract."""
        for filename in c.Infra.OBSOLETE_GENERATED_INIT_FILES:
            path = plan.context.pkg_dir / filename
            previous = self._read_generated_file(path)
            if previous is None or not previous.startswith(c.Infra.AUTOGEN_HEADERS):
                continue
            if check_only:
                self._modified_files.add(str(path))
                continue
            path.unlink()
            self._modified_files.add(str(path))

    def _remove_generated_typing_stub(
        self, plan: m.Infra.LazyInitPlan, *, check_only: bool = False
    ) -> None:
        """Remove stale codegen-owned ``__init__.pyi`` files."""
        stub_path = plan.context.pkg_dir / c.Infra.INIT_PYI
        previous = self._read_generated_file(stub_path)
        if previous is None or not previous.startswith(c.Infra.AUTOGEN_HEADERS):
            return
        if check_only:
            self._modified_files.add(str(stub_path))
            return
        stub_path.unlink()
        self._modified_files.add(str(stub_path))

    def _remove_generated_export_sidecars(
        self, plan: m.Infra.LazyInitPlan, *, check_only: bool = False
    ) -> None:
        """Remove legacy generated export files outside the canonical owner."""
        search_dirs = {
            plan.context.pkg_dir,
            plan.context.pkg_dir / c.Infra.ROOT_EXPORTS_DIR,
        }
        for base_dir in sorted(search_dirs):
            if not base_dir.is_dir():
                continue
            stale_paths = sorted(
                path
                for path in base_dir.glob("*.py")
                if c.Infra.GENERATED_EXPORT_SIDECAR_RE.match(path.name)
            )
            for path in stale_paths:
                if not path.is_file():
                    continue
                if check_only:
                    self._modified_files.add(str(path))
                    continue
                path.unlink()
                self._modified_files.add(str(path))
        constants_dir = plan.context.pkg_dir / c.Infra.ROOT_EXPORTS_DIR
        constants_init = constants_dir / c.Infra.INIT_PY
        if not constants_init.is_file():
            return
        remaining_modules = tuple(
            path for path in constants_dir.glob("*.py") if path.name != c.Infra.INIT_PY
        )
        if remaining_modules:
            return
        content = self._read_generated_file(constants_init)
        if content is None or not content.startswith(c.Infra.AUTOGEN_HEADERS):
            return
        if check_only:
            self._modified_files.add(str(constants_init))
            return
        constants_init.unlink()
        self._modified_files.add(str(constants_init))
        if not tuple(constants_dir.iterdir()):
            constants_dir.rmdir()

    @staticmethod
    def _read_generated_file(path: Path) -> str | None:
        """Read an optional generated support file."""
        if not path.exists():
            return None
        read = u.Cli.files_read_text(path)
        if read.failure:
            message = f"reading {path}: {read.error}"
            raise OSError(message)
        content: str = read.value
        return content


__all__: list[str] = ["FlextInfraCodegenLazyInitGenerationRegistryMixin"]
