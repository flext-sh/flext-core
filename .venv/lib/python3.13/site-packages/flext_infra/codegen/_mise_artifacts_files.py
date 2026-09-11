"""Exact filesystem-state primitives for Mise artifact transactions."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final

from flext_core import r
from flext_infra import c, m, u

if TYPE_CHECKING:
    from flext_infra import p

ARTIFACT_SPECS: Final[tuple[tuple[str, int], ...]] = (
    ("bin/mise", 0o755),
    ("bin/mise.cmd", 0o644),
    ("mise.lock", 0o644),
)
CONFIG_SPEC: Final[tuple[str, int]] = (c.Infra.MISE_TOML_FILENAME, 0o644)
PUBLICATION_SPECS: Final[tuple[tuple[str, int], ...]] = (CONFIG_SPEC, *ARTIFACT_SPECS)
ARTIFACT_NAMES: Final[tuple[str, ...]] = tuple(name for name, _mode in ARTIFACT_SPECS)
JOURNAL_NAME: Final[str] = "journal.json"
JOURNAL_MODE: Final[int] = 0o600
LOCK_NAME: Final[str] = "publication.lock"
STATE_DIRECTORY: Final[Path] = Path(".state") / "mise-artifacts"
TRANSACTION_DIR_NAME: Final[str] = "transaction"


def digest(content: bytes) -> str:
    """Return the exact lowercase SHA-256 identity for raw bytes."""
    return sha256(content).hexdigest()


def read_state(path: Path, *, required: bool) -> p.Result[m.Cli.AtomicFileState]:
    """Read exact state through the canonical descriptor-authenticated owner.

    The descriptor-bound reader authenticates a file through its parent
    directory, so a repository that has never carried ``bin/`` fails there
    before it can report the file itself as absent. An optional read treats a
    missing parent as exactly what it is -- an absent artifact -- while a
    required read still fails loud with the original cause.
    """
    if not required and not path.parent.is_dir():
        return r[m.Cli.AtomicFileState].ok(m.Cli.AtomicFileState(path=path))
    return u.Cli.atomic_read_binary_file_state(path, required=required)


def write_publication(
    publication: m.Infra.MiseToolchainPublication,
) -> p.Result[m.Cli.AtomicFileState]:
    """Consume one staged state through the canonical guarded CLI owner."""
    before = publication.before
    replacement = publication.replacement
    if replacement.content is None or replacement.mode is None:
        return r[m.Cli.AtomicFileState].fail(
            f"Mise publication replacement is absent: {replacement.path}"
        )
    return u.Cli.atomic_publish_staged_binary_file_guarded(before, replacement)


def delete_state(state: m.Cli.AtomicFileState) -> p.Result[bool]:
    """Delete one exact existing state through the CLI owner."""
    if state.content is None or state.mode is None:
        return r[bool].fail(f"cannot delete absent Mise file state: {state.path}")
    return u.Cli.atomic_delete_binary_file_guarded(
        state.path, expected_bytes=state.content, expected_mode=state.mode
    )


def workspace_relative(root: Path, path: Path) -> p.Result[str]:
    """Return a canonical lexical workspace-relative path selector."""
    absolute_root = root.absolute()
    try:
        relative = path.absolute().relative_to(absolute_root)
    except ValueError as exc:
        return r[str].fail_op(f"resolve workspace artifact {path}", exc)
    selector = relative.as_posix()
    if not selector or selector == "." or ".." in relative.parts:
        return r[str].fail(f"invalid workspace artifact path: {path}")
    return r[str].ok(selector)


def resolve_relative(root: Path, selector: str, *, purpose: str) -> p.Result[Path]:
    """Resolve a lexical relative selector without dereferencing its leaf."""
    relative = Path(selector)
    if (
        relative.is_absolute()
        or relative.as_posix() != selector
        or not relative.parts
        or ".." in relative.parts
    ):
        return r[Path].fail(f"unsafe {purpose} path: {selector}")
    absolute_root = root.absolute()
    candidate = (absolute_root / relative).absolute()
    if not candidate.is_relative_to(absolute_root):
        return r[Path].fail(f"{purpose} path escapes its root: {selector}")
    return r[Path].ok(candidate)


__all__: list[str] = [
    "ARTIFACT_NAMES",
    "ARTIFACT_SPECS",
    "CONFIG_SPEC",
    "JOURNAL_MODE",
    "JOURNAL_NAME",
    "LOCK_NAME",
    "PUBLICATION_SPECS",
    "STATE_DIRECTORY",
    "TRANSACTION_DIR_NAME",
    "delete_state",
    "digest",
    "read_state",
    "resolve_relative",
    "workspace_relative",
    "write_publication",
]
