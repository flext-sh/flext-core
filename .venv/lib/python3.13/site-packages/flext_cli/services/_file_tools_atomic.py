"""Atomic filesystem service surface for FLEXT CLI."""

from __future__ import annotations

from flext_cli import m, p, t, u


class FlextCliFileToolsAtomicMixin:
    """Delegate atomic file operations to their canonical utility owner."""

    @staticmethod
    def atomic_write_text_file(
        file_path: t.Cli.TextPath, content: str
    ) -> p.Result[bool]:
        """Write text atomically through the canonical utility surface."""
        return u.Cli.atomic_write_text_file(file_path, content)

    @staticmethod
    def atomic_write_text_file_guarded(
        file_path: t.Cli.TextPath, content: str, *, expected_bytes: bytes | None
    ) -> p.Result[bool]:
        """Publish text after an exact raw-byte precondition."""
        return u.Cli.atomic_write_text_file_guarded(
            file_path, content, expected_bytes=expected_bytes
        )

    @staticmethod
    def atomic_write_binary_file_guarded(
        file_path: t.Cli.TextPath,
        data: bytes,
        *,
        expected_bytes: bytes | None,
        expected_mode: int | None,
        permission_mode: int,
    ) -> p.Result[bool]:
        """Publish exact bytes and mode under the caller's lock."""
        return u.Cli.atomic_write_binary_file_guarded(
            file_path,
            data,
            expected_bytes=expected_bytes,
            expected_mode=expected_mode,
            permission_mode=permission_mode,
        )

    @staticmethod
    def atomic_delete_binary_file_guarded(
        file_path: t.Cli.TextPath, *, expected_bytes: bytes, expected_mode: int
    ) -> p.Result[bool]:
        """Delete one exact byte-and-mode version under the caller's lock."""
        return u.Cli.atomic_delete_binary_file_guarded(
            file_path, expected_bytes=expected_bytes, expected_mode=expected_mode
        )

    @staticmethod
    def atomic_publish_staged_binary_file_guarded(
        destination_before: m.Cli.AtomicFileState, staged: m.Cli.AtomicFileState
    ) -> p.Result[m.Cli.AtomicFileState]:
        """Consume one exact staged file through the utility owner."""
        return u.Cli.atomic_publish_staged_binary_file_guarded(
            destination_before, staged
        )

    @staticmethod
    def atomic_read_binary_file_state(
        file_path: t.Cli.TextPath, *, required: bool = False
    ) -> p.Result[m.Cli.AtomicFileState]:
        """Read an exact descriptor-authenticated byte-and-mode snapshot."""
        return u.Cli.atomic_read_binary_file_state(file_path, required=required)


__all__: list[str] = ["FlextCliFileToolsAtomicMixin"]
