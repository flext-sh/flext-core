"""Content-addressed Git worktree fingerprints for validation integrity."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra._utilities.git import FlextInfraUtilitiesGit
from flext_infra.constants import c
from flext_infra.models import m

if TYPE_CHECKING:
    from flext_infra.protocols import p
    from flext_infra.typings import t


class FlextInfraUtilitiesWorkspaceFingerprint:
    """Fingerprint HEAD, index entries, tracked content, and untracked content."""

    @staticmethod
    def _excluded(path: Path, exclusions: frozenset[Path]) -> bool:
        """Return whether a path is an explicit excluded artifact or descendant."""
        return any(
            path == excluded or path.is_relative_to(excluded) for excluded in exclusions
        )

    @staticmethod
    def _read_content_digest(path: Path) -> bytes:
        """Hash one path without following symlinks."""
        digest = hashlib.sha256()
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            digest.update(b"missing\0")
            return digest.digest()
        digest.update(str(stat.S_IMODE(metadata.st_mode)).encode())
        if stat.S_ISLNK(metadata.st_mode):
            digest.update(b"symlink\0")
            digest.update(os.fsencode(path.readlink()))
        elif stat.S_ISREG(metadata.st_mode):
            digest.update(b"file\0")
            with path.open("rb") as stream:
                while chunk := stream.read(
                    c.Infra.WORKSPACE_FINGERPRINT_READ_CHUNK_BYTES
                ):
                    digest.update(chunk)
        elif stat.S_ISDIR(metadata.st_mode):
            digest.update(b"directory\0")
        else:
            digest.update(f"special:{stat.S_IFMT(metadata.st_mode)}".encode())
        return digest.digest()

    @classmethod
    def _file_content_digest(cls, path: Path) -> p.Result[bytes]:
        """Return a typed content digest or one precise read failure."""
        try:
            return r[bytes].ok(cls._read_content_digest(path))
        except OSError as exc:
            return r[bytes].fail(f"workspace fingerprint read failed for {path}: {exc}")

    @classmethod
    def workspace_fingerprint(
        cls, checkout: Path, *, excluded_paths: t.SequenceOf[Path] = ()
    ) -> p.Result[m.Infra.WorkspaceFingerprint]:
        """Capture a content-addressed snapshot of one Git checkout."""
        root = checkout.resolve()
        inputs = FlextInfraUtilitiesGit.git_fingerprint_inputs(
            m.Infra.GitRepoRequest(repo_root=root)
        )
        if inputs.failure:
            return r[m.Infra.WorkspaceFingerprint].fail(
                inputs.error or f"not a Git worktree: {root}"
            )
        paths_result_value = inputs.value.paths_z
        index_result_value = inputs.value.index_z
        head = inputs.value.head

        index_entries: dict[bytes, list[bytes]] = {}
        for record in index_result_value.split(b"\0"):
            if not record:
                continue
            try:
                metadata, raw_path = record.split(b"\t", maxsplit=1)
            except ValueError:
                return r[m.Infra.WorkspaceFingerprint].fail(
                    "invalid NUL-delimited git index entry"
                )
            index_entries.setdefault(raw_path, []).append(metadata)

        exclusions = frozenset(excluded_paths)
        entries: list[m.Infra.WorkspaceFingerprintEntry] = []
        for raw_path in sorted(filter(None, paths_result_value.split(b"\0"))):
            relative = Path(os.fsdecode(raw_path))
            if relative.is_absolute() or ".." in relative.parts:
                return r[m.Infra.WorkspaceFingerprint].fail(
                    f"unsafe repository path in fingerprint: {relative}"
                )
            if cls._excluded(relative, exclusions):
                continue
            content_result = cls._file_content_digest(root / relative)
            if content_result.failure:
                return r[m.Infra.WorkspaceFingerprint].from_failure(content_result)
            entry_digest = hashlib.sha256()
            entry_digest.update(raw_path)
            entry_digest.update(b"\0")
            for index_metadata in sorted(index_entries.get(raw_path, ())):
                entry_digest.update(index_metadata)
                entry_digest.update(b"\0")
            entry_digest.update(content_result.value)
            entries.append(
                m.Infra.WorkspaceFingerprintEntry(
                    path=relative.as_posix(), digest=entry_digest.hexdigest()
                )
            )

        aggregate = hashlib.sha256(head)
        for entry in entries:
            aggregate.update(entry.path.encode())
            aggregate.update(b"\0")
            aggregate.update(entry.digest.encode())
            aggregate.update(b"\0")
        return r[m.Infra.WorkspaceFingerprint].ok(
            m.Infra.WorkspaceFingerprint(
                digest=aggregate.hexdigest(), entries=tuple(entries)
            )
        )

    @staticmethod
    def workspace_fingerprint_changes(
        before: m.Infra.WorkspaceFingerprint, after: m.Infra.WorkspaceFingerprint
    ) -> t.StrSequence:
        """Return repository paths whose content or index state changed."""
        before_entries = {entry.path: entry.digest for entry in before.entries}
        after_entries = {entry.path: entry.digest for entry in after.entries}
        return tuple(
            path
            for path in sorted(before_entries.keys() | after_entries.keys())
            if before_entries.get(path) != after_entries.get(path)
        )


__all__: list[str] = ["FlextInfraUtilitiesWorkspaceFingerprint"]
