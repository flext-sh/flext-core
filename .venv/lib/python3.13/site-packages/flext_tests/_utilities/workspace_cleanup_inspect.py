"""Git-ignore, WIP, and fingerprint inspection for workspace cleanup utilities."""

from __future__ import annotations

import stat
from pathlib import Path

from flext_cli import u
from flext_tests import c, p, r
from flext_tests._utilities.workspace_cleanup_paths import (
    FlextTestsWorkspaceCleanupPathsUtilitiesMixin,
)


class FlextTestsWorkspaceCleanupInspectUtilitiesMixin(
    FlextTestsWorkspaceCleanupPathsUtilitiesMixin
):
    """Classify residues and fingerprint their exact filesystem state."""

    @staticmethod
    def _reject_unsafe_node(lexical: Path, relative_path: Path) -> p.Result[bool]:
        """Refuse symlinks, hardlinked files, and non-regular filesystem nodes."""
        try:
            info = lexical.lstat()
        except OSError as exc:
            return r[bool].fail(
                f"cleanup residue inspection failed: {relative_path}: {exc}"
            )
        mode = info.st_mode
        if stat.S_ISLNK(mode):
            return r[bool].fail(
                f"cleanup residue is a symlink and cannot be removed safely: "
                f"{relative_path}"
            )
        if stat.S_ISDIR(mode):
            return r[bool].ok(True)
        if not stat.S_ISREG(mode):
            return r[bool].fail(
                f"cleanup residue is not a regular file: {relative_path}"
            )
        if info.st_nlink > 1:
            return r[bool].fail(
                f"cleanup residue is a hardlink shared with other paths: "
                f"{relative_path}"
            )
        return r[bool].ok(True)

    @classmethod
    def _ignored(cls, root: Path, relative_path: Path) -> p.Result[bool]:
        """Require Git to classify one exact candidate as ignored."""
        result = cls._git(
            root,
            ("check-ignore", "--no-index", "--stdin", "-z"),
            input_data=relative_path.as_posix().encode("utf-8") + b"\0",
        )
        if result.failure:
            return r[bool].fail(result.error)
        output = result.value
        if output.exit_code == c.Cli.EXIT_CODE_SUCCESS:
            return r[bool].ok(True)
        if output.exit_code == c.Cli.EXIT_CODE_FAILURE:
            return r[bool].fail(
                f"cleanup residue is not ignored by Git: {relative_path}"
            )
        return r[bool].fail(cls._command_error("git ignore check", output))

    @classmethod
    def _untracked_and_clean(cls, root: Path, relative_path: Path) -> p.Result[bool]:
        """Reject dirty/WIP state and tracked files beneath one candidate."""
        status_result = cls._git(
            root,
            (
                "--literal-pathspecs",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                relative_path.as_posix(),
            ),
        )
        if status_result.failure:
            return r[bool].fail(status_result.error)
        status = status_result.value
        if status.exit_code != c.Cli.EXIT_CODE_SUCCESS:
            return r[bool].fail(cls._command_error("git status check", status))
        if status.stdout:
            return r[bool].fail(
                f"cleanup residue contains dirty or untracked WIP: {relative_path}"
            )
        tracked_result = cls._git(
            root,
            ("--literal-pathspecs", "ls-files", "-z", "--", relative_path.as_posix()),
        )
        if tracked_result.failure:
            return r[bool].fail(tracked_result.error)
        tracked = tracked_result.value
        if tracked.exit_code != c.Cli.EXIT_CODE_SUCCESS:
            return r[bool].fail(cls._command_error("git tracked-path check", tracked))
        if tracked.stdout:
            return r[bool].fail(
                f"cleanup residue contains Git-tracked content: {relative_path}"
            )
        return r[bool].ok(True)

    @staticmethod
    def _fingerprint_entry(root: Path, entry: Path) -> p.Result[str]:
        """Encode one filesystem node without following symbolic links."""
        try:
            stat = entry.lstat()
            relative = Path() if entry == root else entry.relative_to(root)
        except (OSError, ValueError) as exc:
            return r[str].fail(f"cleanup residue fingerprint failed for {entry}: {exc}")
        if entry.is_symlink():
            kind = "symlink"
            try:
                payload = entry.readlink().as_posix()
            except OSError as exc:
                return r[str].fail(
                    f"cleanup residue fingerprint failed for {entry}: {exc}"
                )
        elif entry.is_dir():
            kind = "directory"
            payload = ""
        elif entry.is_file():
            kind = "file"
            try:
                payload = u.Cli.sha256_file(entry)
            except OSError as exc:
                return r[str].fail(
                    f"cleanup residue fingerprint failed for {entry}: {exc}"
                )
        else:
            return r[str].fail(
                f"cleanup residue contains unsupported filesystem node: {entry}"
            )
        # NOTE (multi-agent): NUL framing makes arbitrary path names unambiguous.
        return r[str].ok(
            "\0".join((
                relative.as_posix(),
                kind,
                str(stat.st_mode),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                str(stat.st_ctime_ns),
                str(stat.st_nlink),
                str(stat.st_dev),
                str(stat.st_ino),
                payload,
            ))
        )

    @classmethod
    def _path_fingerprint(cls, path: Path) -> p.Result[str]:
        """Fingerprint one residue tree without following symbolic links."""

        def relative_path_key(item: Path) -> str:
            return item.relative_to(path).as_posix()

        try:
            entries = (
                (path, *sorted(path.rglob("*"), key=relative_path_key))
                if path.is_dir() and not path.is_symlink()
                else (path,)
            )
        except (OSError, ValueError) as exc:
            return r[str].fail(f"cleanup residue fingerprint failed for {path}: {exc}")
        manifest: list[str] = []
        for entry in entries:
            entry_result = cls._fingerprint_entry(path, entry)
            if entry_result.failure:
                return r[str].fail(entry_result.error)
            manifest.append(entry_result.value)
        return r[str].ok(u.Cli.sha256_content("\0".join(manifest)))


__all__: tuple[str, ...] = ("FlextTestsWorkspaceCleanupInspectUtilitiesMixin",)
