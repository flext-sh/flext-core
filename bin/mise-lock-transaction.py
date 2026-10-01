# Copyright 2026 FLEXT
"""Publish a Mise lock with its native sidecars from one physical stage.

This bootstrap runs with the Python selected by the staged Mise lock, before
the project's virtual environment exists. It intentionally uses only stdlib.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import tomllib
from pathlib import Path, PurePosixPath


class MiseLockTransaction:
    """Keep the old lock usable until every new sidecar is published."""

    JOURNAL = "transaction.json"
    NEW_LOCK = "new.lock"
    OLD_LOCK = "old.lock"

    @staticmethod
    def _physical_directory(path: Path) -> None:
        observed = path.lstat()
        if not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"transaction directory is not physical: {path}")

    @staticmethod
    def _bytes(path: Path) -> bytes | None:
        try:
            observed = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise ValueError(f"transaction file is not physical: {path}")
        return path.read_bytes()

    @staticmethod
    def _digest(content: bytes | None) -> str | None:
        return None if content is None else hashlib.sha256(content).hexdigest()

    @staticmethod
    def _sidecar_selector(relative: str) -> PurePosixPath:
        selector = PurePosixPath(relative)
        if (
            selector.is_absolute()
            or selector.as_posix() != relative
            or len(selector.parts) < 4
            or selector.parts[:2] != (".mise", "locks")
            or ".." in selector.parts
        ):
            raise ValueError(f"unsafe mise.lock sidecar: {relative}")
        return selector

    @classmethod
    def _sidecars(cls, content: bytes | None, root: Path) -> dict[str, str]:
        if content is None:
            return {}
        payload = tomllib.loads(content.decode("utf-8"))
        tools = payload.get("tools")
        if not isinstance(tools, dict):
            raise ValueError("mise.lock has no tools table")
        result: dict[str, str] = {}
        for entries in tools.values():
            for entry in entries if isinstance(entries, list) else (entries,):
                if not isinstance(entry, dict):
                    raise ValueError("mise.lock tool entry is not a table")
                for graph, filename in (("aube", "aube-lock.yaml"), ("uv", "uv.lock")):
                    annotation = entry.get(graph)
                    if annotation is None:
                        continue
                    if not isinstance(annotation, dict):
                        raise ValueError(f"mise.lock {graph} annotation is not a table")
                    relative = annotation.get("path")
                    digest = annotation.get("digest")
                    if not isinstance(relative, str) or not isinstance(digest, str):
                        raise ValueError(f"mise.lock {graph} annotation is incomplete")
                    selector = cls._sidecar_selector(relative)
                    if not digest.startswith("sha256:"):
                        raise ValueError(f"invalid mise.lock sidecar digest: {relative}")
                    cls._reject_symlink_path(root, relative)
                    sidecar = root.joinpath(*selector.parts)
                    cls._physical_directory(sidecar)
                    source = cls._bytes(sidecar / filename)
                    if source is None:
                        raise ValueError(f"mise.lock sidecar is absent: {sidecar / filename}")
                    actual = hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest()
                    if actual != digest.removeprefix("sha256:"):
                        raise ValueError(f"mise.lock sidecar digest differs: {sidecar / filename}")
                    result[relative] = cls._tree_digest(sidecar)
        return result

    @classmethod
    def _tree_digest(cls, root: Path) -> str:
        cls._physical_directory(root)
        checksum = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            observed = path.lstat()
            relative = path.relative_to(root).as_posix().encode()
            if stat.S_ISDIR(observed.st_mode):
                checksum.update(b"D\0" + relative + b"\0")
            elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
                checksum.update(b"F\0" + relative + b"\0" + path.read_bytes())
            else:
                raise ValueError(f"nonphysical mise sidecar entry: {path}")
        return checksum.hexdigest()

    @staticmethod
    def _sync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _sync_tree(cls, root: Path) -> None:
        """Persist staged payload bytes before publishing the journal."""
        cls._physical_directory(root)
        for path in sorted(root.rglob("*"), reverse=True):
            observed = path.lstat()
            if stat.S_ISDIR(observed.st_mode):
                cls._sync_directory(path)
            elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            else:
                raise ValueError(f"nonphysical Mise stage entry: {path}")
        cls._sync_directory(root)

    @classmethod
    def _write_journal(cls, stage: Path, journal: dict[str, str]) -> None:
        candidate = stage / "transaction.json.new"
        with candidate.open("x", encoding="utf-8") as stream:
            json.dump(journal, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(candidate, stage / cls.JOURNAL)
        cls._sync_directory(stage)

    @classmethod
    def _read_journal(cls, stage: Path) -> dict[str, str] | None:
        content = cls._bytes(stage / cls.JOURNAL)
        if content is None:
            return None
        payload = json.loads(content)
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in payload.items()
        ):
            raise ValueError(f"invalid Mise lock transaction journal: {stage}")
        return payload

    @staticmethod
    def _journal_refs(journal: dict[str, str], name: str) -> dict[str, str]:
        raw = journal.get(name)
        if raw is None:
            raise ValueError(f"Mise lock journal lacks {name}")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in payload.items()
        ):
            raise ValueError(f"Mise lock journal has invalid {name}")
        for relative in payload:
            MiseLockTransaction._sidecar_selector(relative)
        return payload

    @classmethod
    def _require_roots(cls, project: Path, stage: Path) -> None:
        cls._physical_directory(project)
        cls._physical_directory(stage)
        if stage.parent != project.parent or stage == project:
            raise ValueError("Mise lock stage must be a sibling of its destination")
        if stage.stat().st_dev != project.stat().st_dev:
            raise ValueError("Mise lock stage is not on the destination filesystem")
        if not stage.name.startswith(f".{project.name}.mise-lock-stage."):
            raise ValueError(f"unexpected Mise lock transaction stage: {stage}")

    @staticmethod
    def _reject_symlink_path(project: Path, relative: str) -> None:
        cursor = project
        for part in PurePosixPath(relative).parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f"Mise sidecar path contains a symlink: {cursor}")

    @classmethod
    def _ensure_parent(cls, root: Path, target: Path) -> None:
        """Materialize physical parents and durably record each directory entry."""
        if not target.is_relative_to(root):
            raise ValueError(f"Mise sidecar escapes transaction root: {target}")
        missing: list[Path] = []
        cursor = target.parent
        while cursor != root:
            missing.append(cursor)
            cursor = cursor.parent
        for directory in reversed(missing):
            if directory.exists():
                cls._physical_directory(directory)
            else:
                directory.mkdir()
                cls._sync_directory(directory.parent)

    @classmethod
    def _retire_stage(cls, stage: Path) -> None:
        """Move a completed journal out of the recovery scan before deleting it."""
        retired = stage.with_name(
            stage.name.replace(".mise-lock-stage.", ".mise-lock-cleanup.", 1),
        )
        if retired.exists() or retired.is_symlink():
            raise ValueError(f"Mise cleanup target already exists: {retired}")
        os.rename(stage, retired)
        cls._sync_directory(stage.parent)
        shutil.rmtree(retired)

    @classmethod
    def recover(cls, project: Path, stage: Path) -> None:
        """Finish or undo a prior interrupted publication by its lock commit point."""
        cls._require_roots(project, stage)
        journal = cls._read_journal(stage)
        if journal is None:
            cls._retire_stage(stage)
            return
        if journal.get("project") != str(project):
            raise ValueError(f"Mise lock journal belongs to another project: {stage}")
        old = cls._bytes(stage / cls.OLD_LOCK)
        new = cls._bytes(stage / cls.NEW_LOCK)
        if new is None:
            raise ValueError(f"Mise lock journal lost new lock: {stage}")
        if cls._digest(old) != (journal.get("old") or None) or cls._digest(new) != journal.get("new"):
            raise ValueError(f"Mise lock journal digest changed: {stage}")
        old_refs = cls._journal_refs(journal, "old_refs")
        new_refs = cls._journal_refs(journal, "new_refs")
        for relative in old_refs | new_refs:
            cls._reject_symlink_path(project, relative)
        live = cls._bytes(project / "mise.lock")
        if live == old:
            for relative, expected in new_refs.items():
                destination = project / relative
                backup = stage / "old-sidecars" / relative
                abandoned = stage / "abandoned-sidecars" / relative
                if destination.exists() and cls._tree_digest(destination) == expected and expected != old_refs.get(relative):
                    cls._ensure_parent(stage, abandoned)
                    os.rename(destination, abandoned)
                    cls._sync_directory(destination.parent)
                    cls._sync_directory(abandoned.parent)
                if backup.exists():
                    if destination.exists() or cls._tree_digest(backup) != old_refs[relative]:
                        raise ValueError(f"old Mise sidecar changed during recovery: {backup}")
                    cls._ensure_parent(project, destination)
                    os.rename(backup, destination)
                    cls._sync_directory(backup.parent)
                    cls._sync_directory(destination.parent)
                elif relative in old_refs:
                    if not destination.exists() or cls._tree_digest(destination) != old_refs[relative]:
                        raise ValueError(f"old Mise sidecar missing during recovery: {destination}")
                elif destination.exists():
                    raise ValueError(f"unowned Mise sidecar changed during recovery: {destination}")
            cls._retire_stage(stage)
            return
        if live != new:
            raise ValueError(f"Mise lock changed outside transaction: {project / 'mise.lock'}")
        for relative, expected in new_refs.items():
            destination = project / relative
            if not destination.exists() or cls._tree_digest(destination) != expected:
                raise ValueError(f"committed Mise sidecar differs: {destination}")
        for relative, expected in old_refs.items():
            if relative in new_refs:
                continue
            destination = project / relative
            retired = stage / "retired-sidecars" / relative
            if destination.exists():
                if cls._tree_digest(destination) != expected:
                    raise ValueError(f"stale sidecar changed during recovery: {destination}")
                cls._ensure_parent(stage, retired)
                os.rename(destination, retired)
                cls._sync_directory(destination.parent)
                cls._sync_directory(retired.parent)
            elif not retired.exists():
                raise ValueError(f"stale sidecar disappeared during recovery: {destination}")
        cls._retire_stage(stage)

    @classmethod
    def publish(cls, project: Path, stage: Path) -> None:
        """Publish sidecars first and make the lock rename the commit point."""
        cls._require_roots(project, stage)
        for prior in sorted(project.parent.glob(f".{project.name}.mise-lock-stage.*")):
            if prior != stage:
                cls.recover(project, prior)
        for retired in sorted(project.parent.glob(f".{project.name}.mise-lock-cleanup.*")):
            cls._physical_directory(retired)
            shutil.rmtree(retired)
        old = cls._bytes(project / "mise.lock")
        new = cls._bytes(stage / "mise.lock")
        if new is None:
            raise ValueError(f"staged mise.lock is absent: {stage}")
        old_refs = cls._sidecars(old, project)
        new_refs = cls._sidecars(new, stage)
        cls._sync_tree(stage)
        for relative in old_refs | new_refs:
            cls._reject_symlink_path(project, relative)
        for relative, expected in new_refs.items():
            destination = project / relative
            if destination.exists():
                if relative not in old_refs:
                    raise ValueError(f"unowned Mise sidecar occupies target: {destination}")
                actual = cls._tree_digest(destination)
                if actual != expected and actual != old_refs.get(relative):
                    raise ValueError(f"Mise sidecar changed outside transaction: {destination}")
        if old is not None:
            with (stage / cls.OLD_LOCK).open("xb") as stream:
                stream.write(old)
                stream.flush()
                os.fsync(stream.fileno())
        with (stage / cls.NEW_LOCK).open("xb") as stream:
            stream.write(new)
            stream.flush()
            os.fsync(stream.fileno())
        cls._sync_directory(stage)
        cls._write_journal(stage, {
            "project": str(project),
            "old": cls._digest(old) or "",
            "new": cls._digest(new) or "",
            "old_refs": json.dumps(old_refs, sort_keys=True),
            "new_refs": json.dumps(new_refs, sort_keys=True),
        })
        for relative, expected in new_refs.items():
            destination = project / relative
            if destination.exists():
                if cls._tree_digest(destination) == expected:
                    continue
                backup = stage / "old-sidecars" / relative
                cls._ensure_parent(stage, backup)
                os.rename(destination, backup)
                cls._sync_directory(destination.parent)
                cls._sync_directory(backup.parent)
            cls._ensure_parent(project, destination)
            os.rename(stage / relative, destination)
            cls._sync_directory(destination.parent)
            cls._sync_directory((stage / relative).parent)
        os.replace(stage / "mise.lock", project / "mise.lock")
        cls._sync_directory(project)
        cls.recover(project, stage)

    @classmethod
    def main(cls, arguments: list[str]) -> int:
        if len(arguments) != 3 or arguments[0] != "publish":
            raise ValueError("usage: mise-lock-transaction.py publish PROJECT STAGE")
        project = Path(arguments[1]).absolute()
        stage = Path(arguments[2]).absolute()
        cls.publish(project, stage)
        return 0


if __name__ == "__main__":
    raise SystemExit(MiseLockTransaction.main(sys.argv[1:]))
