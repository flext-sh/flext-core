"""Path validation and containment for workspace cleanup utilities."""

from __future__ import annotations

from pathlib import Path

from flext_tests import c, p, r
from flext_tests._utilities.workspace_cleanup_git import (
    FlextTestsWorkspaceCleanupGitUtilitiesMixin,
)


class FlextTestsWorkspaceCleanupPathsUtilitiesMixin(
    FlextTestsWorkspaceCleanupGitUtilitiesMixin
):
    """Resolve the Git root and validate normalized contained residue paths."""

    # NOTE (multi-agent): immutable hard-deny floor; these components are never
    # development residue and must never be a cleanup candidate or ancestor.
    _PROTECTED_COMPONENTS: frozenset[str] = frozenset({
        ".git",
        ".hg",
        ".svn",
        ".beads",
        ".venv",
        "venv",
        ".env",
        ".ssh",
        ".gnupg",
        "secrets",
        "credentials",
        "config",
        "settings",
    })

    @classmethod
    def _workspace_root(
        cls, request: p.Tests.WorkspaceCleanupRequest
    ) -> p.Result[Path]:
        """Require the request root to be the exact enclosing Git worktree root."""
        try:
            root = request.workspace_root.resolve(strict=True)
        except OSError as exc:
            return r[Path].fail(f"cleanup workspace root resolution failed: {exc}")
        if not root.is_dir():
            return r[Path].fail(f"cleanup workspace root is not a directory: {root}")
        git_result = cls._git(root, ("rev-parse", "--show-toplevel"))
        if git_result.failure:
            return r[Path].fail(git_result.error)
        output = git_result.value
        if output.exit_code != c.Cli.EXIT_CODE_SUCCESS:
            return r[Path].fail(cls._command_error("git root discovery", output))
        raw_root = output.stdout.strip()
        if not raw_root:
            return r[Path].fail("git root discovery returned an empty path")
        try:
            git_root = Path(raw_root).resolve(strict=True)
        except OSError as exc:
            return r[Path].fail(f"git root resolution failed: {exc}")
        if git_root != root:
            return r[Path].fail(
                f"cleanup root must equal the Git worktree root: {root} != {git_root}"
            )
        return r[Path].ok(root)

    @staticmethod
    def _relative_path(path: Path) -> p.Result[Path]:
        """Validate one exact normalized workspace-relative path."""
        if path.is_absolute() or not path.parts:
            return r[Path].fail(f"cleanup residue must be a relative path: {path}")
        if any(part in {".", ".."} for part in path.parts):
            return r[Path].fail(f"cleanup residue is not normalized: {path}")
        normalized = Path(*path.parts)
        if normalized != path:
            return r[Path].fail(f"cleanup residue is not normalized: {path}")
        return r[Path].ok(normalized)

    @staticmethod
    def _lexical_path(root: Path, relative_path: Path) -> p.Result[Path]:
        """Resolve containment while retaining the lexical path for symlink unlinking."""
        lexical = root.joinpath(relative_path)
        try:
            resolved = lexical.resolve(strict=False)
            _ = resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            return r[Path].fail(
                f"cleanup residue escapes the workspace: {relative_path}: {exc}"
            )
        if lexical == root:
            return r[Path].fail("cleanup residue cannot be the workspace root")
        return r[Path].ok(lexical)

    @classmethod
    def _reject_protected(cls, root: Path, relative_path: Path) -> p.Result[bool]:
        """Refuse any residue that targets a protected component or the Git dir."""
        if any(part in cls._PROTECTED_COMPONENTS for part in relative_path.parts):
            return r[bool].fail(
                f"cleanup residue targets a protected path: {relative_path}"
            )
        for name in ("--git-dir", "--git-common-dir"):
            git_result = cls._git(root, ("rev-parse", name))
            if git_result.failure:
                return r[bool].fail(git_result.error)
            output = git_result.value
            if output.exit_code != c.Cli.EXIT_CODE_SUCCESS:
                return r[bool].fail(cls._command_error("git dir discovery", output))
            raw = output.stdout.strip()
            if not raw:
                continue
            try:
                git_dir = Path(raw if Path(raw).is_absolute() else root / raw).resolve(
                    strict=False
                )
                candidate = root.joinpath(relative_path).resolve(strict=False)
            except (OSError, ValueError) as exc:
                return r[bool].fail(
                    f"protected path resolution failed: {relative_path}: {exc}"
                )
            if candidate == git_dir or git_dir in candidate.parents:
                return r[bool].fail(
                    f"cleanup residue targets the protected Git directory: {relative_path}"
                )
            if candidate in git_dir.parents:
                return r[bool].fail(
                    f"cleanup residue would remove the protected Git directory: {relative_path}"
                )
        return r[bool].ok(True)

    @staticmethod
    def _reject_symlink_ancestor(root: Path, relative_path: Path) -> p.Result[bool]:
        """Refuse a residue whose own ancestor components are symbolic links."""
        current = root
        for part in relative_path.parts[:-1]:
            current /= part
            if current.is_symlink():
                return r[bool].fail(
                    f"cleanup residue has a symlink ancestor: {relative_path}"
                )
        return r[bool].ok(True)


__all__: tuple[str, ...] = ("FlextTestsWorkspaceCleanupPathsUtilitiesMixin",)
