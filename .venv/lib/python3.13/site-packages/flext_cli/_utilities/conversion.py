"""CLI conversion helpers shared through ``u.Cli``."""

from __future__ import annotations

from pathlib import Path

from flext_cli import c, m, p, r, t
from flext_core import u


class FlextCliUtilitiesConversion:
    """Conversion methods exposed directly on ``u.Cli``."""

    @staticmethod
    def default_for_type_kind(
        type_kind: t.Cli.TypeKind, default: t.JsonValue | None
    ) -> t.Cli.TypedExtractValue:
        """Return a canonical default for one type kind."""
        return m.Cli.TypedExtract(type_kind=type_kind, default=default).resolved

    @staticmethod
    def cli_args_to_model[M: t.Cli.ModelLike](
        model_class: t.ModelClass[M], cli_args: t.JsonMapping
    ) -> p.Result[M]:
        """Convert a CLI args mapping into a validated Pydantic model."""
        # NOTE (multi-agent): Keep one model-class protocol across CLI utilities.
        try:
            instance: M = model_class.model_validate(cli_args)
            return r[M].ok(instance)
        except c.ValidationError as exc:
            return r[M].fail(f"Validation error for {model_class.__name__}: {exc}")

    @staticmethod
    def resolve_optional_path(value: t.Cli.TextPath | None, *, default: Path) -> Path:
        """Resolve an optional text/path value while preserving a default path."""
        if isinstance(value, Path):
            return value
        normalized = FlextCliUtilitiesConversion.normalize_optional_text(value)
        if normalized is not None:
            return Path(normalized)
        return default

    @staticmethod
    def normalize_optional_text(value: t.JsonValue | Path) -> str | None:
        """Normalize optional text-like values, preserving ``None`` for empties."""
        if isinstance(value, Path):
            return str(value)
        normalized = u.norm_str(value, default="").strip()
        return normalized or None

    @staticmethod
    def normalize_required_text(value: t.JsonValue | Path, *, default: str) -> str:
        """Normalize required text-like values, falling back to ``default``."""
        normalized = FlextCliUtilitiesConversion.normalize_optional_text(value)
        return normalized if normalized is not None else default


__all__: t.MutableSequenceOf[str] = ["FlextCliUtilitiesConversion"]
