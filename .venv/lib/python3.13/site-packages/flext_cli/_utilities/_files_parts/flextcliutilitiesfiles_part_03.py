"""Generic filesystem helpers shared through ``u.Cli``."""

from __future__ import annotations

import hashlib
import shutil
import stat
from pathlib import Path

from flext_cli import c, m, p, r, t
from flext_cli._utilities._atomic_file_publish import publish_guarded_staged_file
from flext_cli._utilities._atomic_file_snapshot import read_authenticated_state
from flext_cli._utilities._files_parts.flextcliutilitiesfiles_part_02 import (
    FlextCliUtilitiesFiles as FlextCliUtilitiesFilesPart02,
)


class FlextCliUtilitiesFiles:
    """Implementation part for FlextCliUtilitiesFiles."""

    @staticmethod
    def atomic_read_binary_file_state(
        file_path: t.Cli.TextPath, *, required: bool = False
    ) -> p.Result[m.Cli.AtomicFileState]:
        """Read exact bytes and mode from one descriptor-authenticated file."""
        path = Path(file_path)
        try:
            state, content = read_authenticated_state(path, required=required)
        except OSError as exc:
            return r[m.Cli.AtomicFileState].fail(
                c.Cli.ERR_BINARY_READ_FAILED.format(error=exc)
            )
        return r[m.Cli.AtomicFileState].ok(
            m.Cli.AtomicFileState(
                path=path,
                content=content,
                mode=None if state is None else stat.S_IMODE(state.st_mode),
                device=None if state is None else state.st_dev,
                inode=None if state is None else state.st_ino,
            )
        )

    @staticmethod
    def atomic_publish_staged_binary_file_guarded(
        destination_before: m.Cli.AtomicFileState, staged: m.Cli.AtomicFileState
    ) -> p.Result[m.Cli.AtomicFileState]:
        """Consume one authenticated staged file under the caller's lock."""
        if (
            staged.content is None
            or staged.mode is None
            or staged.device is None
            or staged.inode is None
        ):
            return r[m.Cli.AtomicFileState].fail(
                f"atomic staged file is absent: {staged.path}"
            )
        expected_identity: tuple[int, int] | None = None
        if destination_before.content is not None:
            expected_device = destination_before.device
            expected_inode = destination_before.inode
            if expected_device is None or expected_inode is None:
                return r[m.Cli.AtomicFileState].fail(
                    f"atomic destination identity is absent: {destination_before.path}"
                )
            expected_identity = (expected_device, expected_inode)
        try:
            published = publish_guarded_staged_file(
                destination_before.path,
                staged.path,
                expected_bytes=destination_before.content,
                expected_mode=destination_before.mode,
                expected_identity=expected_identity,
                staged_bytes=staged.content,
                staged_mode=staged.mode,
                staged_identity=(staged.device, staged.inode),
            )
        except OSError as exc:
            return r[m.Cli.AtomicFileState].fail(
                c.Cli.ERR_BINARY_WRITE_FAILED.format(error=exc)
            )
        return r[m.Cli.AtomicFileState].ok(
            m.Cli.AtomicFileState(
                path=destination_before.path,
                content=staged.content,
                mode=stat.S_IMODE(published.st_mode),
                device=published.st_dev,
                inode=published.st_ino,
            )
        )

    @staticmethod
    def files_list_directory_names(
        file_path: t.Cli.TextPath,
    ) -> p.Result[t.SequenceOf[str]]:
        """Return sorted child directory names for one path."""
        path = Path(file_path)
        if not path.exists():
            return r[t.SequenceOf[str]].ok(())

        def _list() -> t.SequenceOf[str]:
            names = sorted(entry.name for entry in path.iterdir() if entry.is_dir())
            return tuple(names)

        return FlextCliUtilitiesFilesPart02.files_execute(
            _list, c.Cli.ERR_TEXT_READ_FAILED
        )

    @staticmethod
    def ensure_symlink(
        target: t.Cli.TextPath, source: t.Cli.TextPath
    ) -> p.Result[bool]:
        """Ensure target points to source via directory symlink."""
        target_path = Path(target)
        source_path = Path(source).resolve()
        ensure_result = FlextCliUtilitiesFilesPart02.ensure_dir(target_path.parent)
        if ensure_result.failure:
            return r[bool].fail(
                ensure_result.error
                or c.Cli.ERR_CREATE_PARENT_DIR_FAILED.format(target_path=target_path)
            )
        if target_path.is_symlink() and target_path.resolve() == source_path:
            return r[bool].ok(True)
        FlextCliUtilitiesFiles._remove_symlink_target(target_path)
        try:
            target_path.symlink_to(source_path, target_is_directory=True)
        except OSError as exc:
            return r[bool].fail(
                c.Cli.ERR_ENSURE_SYMLINK_FAILED.format(
                    target_path=target_path, error=exc
                )
            )
        return r[bool].ok(True)

    @staticmethod
    def _remove_symlink_target(target_path: Path) -> None:
        """Remove an existing file, directory, or symlink at ``target_path``."""
        if not target_path.exists() and not target_path.is_symlink():
            return
        if target_path.is_dir() and not target_path.is_symlink():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    @staticmethod
    def sha256_content(content: str) -> str:
        """Return the SHA-256 hex digest for text content."""
        return hashlib.sha256(content.encode(c.Cli.ENCODING_DEFAULT)).hexdigest()

    @staticmethod
    def sha256_file(file_path: t.Cli.TextPath) -> str:
        """Return the SHA-256 hex digest for a file on disk."""
        path = Path(file_path)
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


__all__: list[str] = ["FlextCliUtilitiesFiles"]
