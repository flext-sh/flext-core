"""Real module-bound aliases for runtime and static-only inspection contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAliasType

if TYPE_CHECKING:
    from pathlib import PurePath as DeferredPath


class TestsTypeAliasDeclarations:
    """Importable declarations keep runtime identities tied to their source."""

    class LegalTypes:
        """Aliases with either runtime values or a declared reverse import."""

        type Deferred = DeferredPath | str
        type Resolved = int | str
        type Nested = tuple[Deferred, ...]

    class OtherScope:
        """An unrelated guarded import cannot prove another class's deferral."""

        if TYPE_CHECKING:
            from pathlib import PurePath as Absent

    class InvalidTypes:
        """A missing name without a guarded import retains its NameError."""

        if TYPE_CHECKING:
            type Absent = str

        type Invalid = Absent | str

    class MissingAttribute:
        """A static-only declaration is absent at the runtime attribute boundary."""

        if TYPE_CHECKING:
            type Value = str

    class InvalidAttributeTypes:
        """AttributeError remains a causal failure of the alias evaluator."""

        type Invalid = TestsTypeAliasDeclarations.MissingAttribute.Value

    @staticmethod
    def unbound() -> TypeAliasType:
        """A local alias has no module-bound declaration that proves deferral."""

        class Host:
            type Local = DeferredPath | str

        return Host.Local
