"""Typed boundary for exact pytest FILE, MATCH, and WHAT selectors."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Self, override

from flext_core import r
from flext_infra import m, u
from flext_infra.base import s

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraPytestSelectorValidator(s[bool]):
    """Validate structured pytest selectors before Make materializes argv."""

    file: Annotated[
        str | None,
        m.Field(
            default=None,
            min_length=1,
            description="Exact repository-relative pytest path or nodeid.",
        ),
    ] = None
    match: Annotated[
        str | None,
        m.Field(default=None, min_length=1, description="Exact pytest -k expression."),
    ] = None
    what: Annotated[
        str | None,
        m.Field(default=None, min_length=1, description="Exact pytest submode."),
    ] = None

    @staticmethod
    def _file_path_violation(file: str | None) -> str | None:
        """Return the normalized-path violation independently of mode syntax."""
        if file is None:
            return None
        path_prefix = file.split("::", maxsplit=1)[0]
        parts = path_prefix.split("/")
        if (
            not path_prefix
            or path_prefix.startswith(("-", "/"))
            or "\\" in path_prefix
            or any(part in {"", ".", ".."} for part in parts)
        ):
            return "file must have a normalized repository-relative path prefix"
        return None

    @staticmethod
    def syntax_violation(
        *, file: str | None, match: str | None, what: str | None
    ) -> str | None:
        """Return one deterministic selector syntax violation, if present."""
        for field_name, value in (("file", file), ("match", match), ("what", what)):
            if value is not None and any(character in value for character in "\0\r\n"):
                return f"{field_name} must not contain control separators"
        allowed = {
            None,
            "all",
            "full",
            "profile",
            "cache-status",
            "cache-clear",
            "cache-checkpoint",
        }
        if what not in allowed:
            return (
                "what must be: all, full, profile, cache-status, cache-clear, "
                "or cache-checkpoint"
            )
        if what == "profile" and file is None and match is None:
            return "profile requires FILE or MATCH"
        if what in {"cache-status", "cache-clear", "cache-checkpoint"} and (
            file is not None or match is not None
        ):
            return f"{what} rejects FILE and MATCH"
        if what == "full" and (file is not None or match is not None):
            return "full rejects FILE and MATCH"
        return FlextInfraPytestSelectorValidator._file_path_violation(file)

    @staticmethod
    def resolve_file(root: Path, file: str) -> p.Result[Path]:
        """Resolve a FILE path prefix under root without following symlink hops."""
        path_prefix = file.split("::", maxsplit=1)[0]
        unresolved = root / Path(path_prefix)
        cursor = root
        for part in Path(path_prefix).parts:
            cursor /= part
            if cursor.is_symlink():
                return r[Path].fail(
                    f"file path must not traverse a symlink: {path_prefix}"
                )
        resolved_root = root.resolve()
        resolved_target = unresolved.resolve()
        if not resolved_target.is_relative_to(resolved_root):
            return r[Path].fail(f"file path escapes workspace: {path_prefix}")
        if not resolved_target.exists():
            return r[Path].fail(f"file path does not exist: {path_prefix}")
        return r[Path].ok(resolved_target)

    @u.model_validator(mode="after")
    def _validate_selector_syntax(self) -> Self:
        """Reject control separators and non-normalized FILE path prefixes."""
        if msg := self.syntax_violation(
            file=self.file, match=self.match, what=self.what
        ):
            raise ValueError(msg)
        return self

    @override
    def execute(self) -> p.Result[bool]:
        """Prove FILE containment and existence without reparsing selector text."""
        if self.file is None:
            return r[bool].ok(True)
        resolved = self.resolve_file(self.root, self.file)
        if resolved.failure:
            return r[bool].fail(resolved.error or "pytest FILE resolution failed")
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraPytestSelectorValidator"]
