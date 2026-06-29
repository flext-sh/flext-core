"""Skill pointer enforcement catalog rows."""

from __future__ import annotations

from typing import Final


class FlextConstantsEnforcementCatalogSkillRows:
    """Skill pointer rows for the enforcement catalog."""

    SKILL_POINTER_ROWS: Final[
        tuple[tuple[str, str, str, str, str, tuple[str, ...], str], ...]
    ] = (
        (
            "ENFORCE-034",
            "HIGH",
            "flext-patterns",
            "no-accessor-methods",
            "3-code-law",
            ("flext-patterns",),
            "Accessor method (get_*, set_*) forbidden — expose as field or @u.computed_field (AGENTS.md §3.1).",
        ),
        (
            "ENFORCE-035",
            "HIGH",
            "lib-pydantic-settings",
            "settings-baseline",
            "2-architecture-law",
            ("lib-pydantic-settings",),
            "Settings models must inherit FlextSettings, not BaseModel or BaseSettings (AGENTS.md §2.6).",
        ),
        (
            "ENFORCE-036",
            "MEDIUM",
            "pydantic-v2-governance",
            "",
            "0-quick-reference-must-read",
            ("pydantic-v2-governance",),
            "Never call `model_rebuild()` as a fix strategy — resolve forward refs via proper imports/annotations.",
        ),
        (
            "ENFORCE-037",
            "HIGH",
            "lib-pydantic-settings",
            "",
            "0-quick-reference-must-read",
            ("lib-pydantic-settings", "flext-constants-discipline"),
            "No `os.environ` / `os.getenv` in src/ — use settings + constants contracts.",
        ),
        (
            "ENFORCE-038",
            "HIGH",
            "flext-mro-namespace-rules",
            "",
            "0-quick-reference-must-read",
            ("flext-mro-namespace-rules",),
            "Never flatten organic namespace paths — preserve `m.TargetOracle.ExecuteResult` etc., don't rebind to `m.ExecuteResult`.",
        ),
        (
            "ENFORCE-056",
            "HIGH",
            "flext-mro-namespace-rules",
            "facade-purity",
            "2-5-services-pattern",
            ("flext-mro-namespace-rules", "flext-strict-refactoring"),
            "api.py must contain exactly one ClassDef whose body is Pass-only and exactly one eager alias assignment. Logic in api.py or module-level mutable instance (e.g., `api = FlextApi()`) violates AGENTS.md §2.5 (Services Pattern facade purity).",
        ),
        (
            "ENFORCE-057",
            "HIGH",
            "flext-mro-namespace-rules",
            "service-mixin-inheritance",
            "2-5-services-pattern",
            ("flext-mro-namespace-rules", "flext-patterns"),
            "Classes in services/*.py must end MRO at the project's Flext<X>ServiceBase (which itself ends at FlextService/s[T]). Plain classes lacking the canonical service base violate AGENTS.md §2.5 (Services Pattern services/* shape).",
        ),
        (
            "ENFORCE-058",
            "HIGH",
            "pydantic-v2-governance",
            "models-have-no-helpers",
            "3-1-supreme-law",
            ("pydantic-v2-governance", "pydantic-v2-patterns"),
            "Pydantic 2 data models must contain only fields, model_config, field_validator/model_validator/computed_field, and essential dunders. Methods named get_*/to_*/from_*/is_*/with_* on data models violate AGENTS.md §3.1 (Pydantic v2 Mastery) — relocate to a service mixin or @computed_field. Carve-out: infrastructure base classes (FlextSettingsBase, FlextService, Flext<X>ServiceBase) may expose singleton/clone/factory kernel methods. Extends ENFORCE-034 (get_/set_) to to_/from_/is_/with_ family.",
        ),
        (
            "ENFORCE-059",
            "HIGH",
            "pydantic-v2-governance",
            "data-boundaries-are-models",
            "3-1-supreme-law",
            ("pydantic-v2-governance", "flext-patterns"),
            "Public method parameters and return types in services/*.py and api.py must be Pydantic 2 models, p.* Protocols, r[T] of those, or PEP 604 unions thereof. Bare `dict`, `list[primitive]`, `tuple[primitive...]`, `set`, `TypedDict`, `Mapping[str, Any]`, `Sequence[primitive]` are loose data crossings and violate AGENTS.md §3.1.",
        ),
        (
            "ENFORCE-060",
            "HIGH",
            "pydantic-v2-patterns",
            "no-round-trip-validation",
            "3-1-supreme-law",
            ("pydantic-v2-patterns",),
            "Sequence `model_dump()` -> dict ops -> `model_validate(...)` in the same scope is round-trip validation and violates AGENTS.md §3.1. Use `model_copy(update={...})` directly. JSON roundtrips: use `model_dump(mode='json')` / `model_dump_json()` / `model_validate_json()` — never `json.loads(model.model_dump_json())`.",
        ),
        (
            "ENFORCE-061",
            "MEDIUM",
            "flext-strict-typing",
            "override-decorator-required",
            "3-1-supreme-law",
            ("flext-strict-typing",),
            "Methods overriding a parent method must carry `@typing.override` (PEP 698). Missing decorator weakens static refactoring guarantees and violates AGENTS.md §3.1 (Python 3.13 idioms).",
        ),
        (
            "ENFORCE-062",
            "HIGH",
            "flext-strict-typing",
            "match-exhaustiveness-assert-never",
            "3-1-supreme-law",
            ("flext-strict-typing", "pydantic-v2-patterns"),
            "`match value:` over a discriminated union or finite type set must include `case _: assert_never(value)` as default. Missing exhaustiveness guard violates AGENTS.md §3.1 (Python 3.13 idioms — typing.assert_never).",
        ),
        (
            "ENFORCE-063",
            "HIGH",
            "flext-mro-namespace-rules",
            "constants-chain-inheritance",
            "2-3-mro-inheritance-namespace-composition",
            ("flext-mro-namespace-rules", "flext-architecture-layers"),
            "Flext<X>{Constants,Models,Protocols,Utilities} must extend the immediately-preceding project's facade in the dependency chain (flext-core -> flext-cli -> flext-infra -> flext-{ldap,ldif,...} -> flext-{tap,target,dbt}-* -> end-user), not skip to FlextX root when an intermediate parent project is depended on. Violates AGENTS.md §2.3 (MRO Cascade) by redeclaring symbols already inheritable.",
        ),
    )


__all__ = ["FlextConstantsEnforcementCatalogSkillRows"]
