"""Surgical beartype patches for ``typing_extensions`` PEP 695 aliases.

beartype 0.23 (git pin ``ee481e0c``) mishandles pydantic/``typing_extensions``
PEP 695 type aliases in two independent ways. Both are patched once and
idempotently:

1. **Recognition.** beartype only recognises the stdlib ``typing.TypeAliasType``
   (Python >= 3.12) as a PEP-695 alias. Pydantic builds aliases such as
   ``pydantic.JsonValue`` via ``typing_extensions.TypeAliasType``, so beartype's
   hint reduction asserts ``not PEP 695-compliant unsubscripted type alias``. We
   extend beartype's ``HintPep695TypeAlias`` cave tuple to also accept the
   ``typing_extensions`` type.

2. **Forward-ref module scope.** A PEP-695 alias value is evaluated lazily in the
   *alias's own* defining module. ``pydantic.JsonValue`` is recursive: its
   ``__value__`` embeds stringified self-references (``list['JsonValue']``) that
   carry no module. beartype resolves those bare strings against the *decorated
   callable's* module instead of ``pydantic.types`` and raises
   ``BeartypeCallHintPep484ForwardRefStrException``. We wrap beartype's
   ``get_hint_pep695_unsubbed_alias`` getter so the reduced value's stringified
   refs are tagged with the alias's ``__module__`` (``typing.ForwardRef``
   ``module=``); beartype's existing module-scoped resolver
   (``redpep484ref``, which reads ``__forward_module__``) then resolves them.

Both patches are independent; each is required. They must be installed before
``beartype.claw`` activates.
"""

from __future__ import annotations

import importlib
import sys as _sys
from collections.abc import Callable
from typing import Annotated, ClassVar, ForwardRef, cast, get_args, get_origin

import typing_extensions as _typing_extensions

from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesBeartypeTypingExtPatch:
    """Idempotent beartype patches for ``typing_extensions`` PEP 695 aliases."""

    type _TypeHintSpecifier = t.TypeHintSpecifier | _typing_extensions.TypeAliasType
    type _HintPep695AliasValue = type | tuple[type, ...]
    type _Pep695Getter = Callable[[_TypeHintSpecifier, str], _TypeHintSpecifier]

    _applied: ClassVar[bool] = False

    @classmethod
    def apply(cls) -> None:
        """Install both patches once; subsequent calls are a no-op."""
        if cls._applied:
            return
        cls._patch_alias_recognition()
        cls._patch_forwardref_module_scope()
        cls._applied = True

    @classmethod
    def _patch_alias_recognition(cls) -> None:
        """Extend ``HintPep695TypeAlias`` with ``typing_extensions.TypeAliasType``."""
        cavefast = importlib.import_module("beartype._cave._cavefast")
        raw_current = cavefast.__dict__.get("HintPep695TypeAlias")
        if isinstance(raw_current, type):
            current: cls._HintPep695AliasValue = raw_current
        elif isinstance(raw_current, tuple):
            current_types = tuple(
                item for item in raw_current if isinstance(item, type)
            )
            if len(current_types) != len(raw_current):
                return
            current = current_types
        else:
            return

        new_value = current
        te = _typing_extensions.TypeAliasType

        if isinstance(current, tuple):
            if te not in current:
                new_value = (*current, te)
        elif current is te:
            return
        else:
            new_value = (current, te)

        if new_value is not current:
            cavefast.__dict__["HintPep695TypeAlias"] = new_value
            # Modules that did ``from _cavefast import HintPep695TypeAlias`` hold
            # a stale reference; patch only loaded beartype modules and avoid
            # module-level __getattr__ side effects during import-time patching.
            for mod_name, mod in tuple(_sys.modules.items()):
                if not mod_name.startswith("beartype"):
                    continue
                if mod.__dict__.get("HintPep695TypeAlias") is current:
                    mod.__dict__["HintPep695TypeAlias"] = new_value

    @classmethod
    def _patch_forwardref_module_scope(cls) -> None:
        """Tag a reduced alias's stringified refs with the alias's module."""
        pep695 = importlib.import_module(
            "beartype._util.hint.pep.proposal.pep695",
        )
        raw_original = pep695.__dict__.get("get_hint_pep695_unsubbed_alias")
        if not callable(raw_original):
            return

        # Dynamic module dictionaries expose plugin callables as object; callable()
        # above validates the runtime boundary before narrowing the contract.
        original = cast("cls._Pep695Getter", raw_original)

        def tagged(
            hint: cls._TypeHintSpecifier,
            exception_prefix: str = "",
        ) -> cls._TypeHintSpecifier:
            reduced = original(hint, exception_prefix)
            module_name = getattr(hint, "__module__", None)
            if isinstance(module_name, str) and module_name:
                return cls._tag_forward_refs(reduced, module_name)
            return reduced

        # Replace the getter across every beartype module that imported it by
        # name (``redpep695``) as well as its defining module.
        for mod_name, mod in tuple(_sys.modules.items()):
            if not mod_name.startswith("beartype"):
                continue
            if mod.__dict__.get("get_hint_pep695_unsubbed_alias") is original:
                mod.__dict__["get_hint_pep695_unsubbed_alias"] = tagged

    @staticmethod
    def _tag_forward_refs(
        hint: _TypeHintSpecifier,
        module_name: str,
    ) -> _TypeHintSpecifier:
        """Rebind bare stringified forward refs in ``hint`` to ``module_name``."""
        tag = FlextUtilitiesBeartypeTypingExtPatch._tag_forward_refs
        if isinstance(hint, str):
            return ForwardRef(hint, module=module_name)
        if isinstance(hint, ForwardRef):
            if hint.__forward_module__:
                return hint
            return ForwardRef(hint.__forward_arg__, module=module_name)
        origin = get_origin(hint)
        args = get_args(hint)
        if origin is None or not args:
            return hint
        if origin is Annotated:
            new_args = (tag(args[0], module_name), *args[1:])
        else:
            new_args = tuple(tag(child, module_name) for child in args)
        if new_args == args:
            return hint
        if len(new_args) == 1:
            return cast(
                "FlextUtilitiesBeartypeTypingExtPatch._TypeHintSpecifier",
                origin[new_args[0]],
            )
        return cast(
            "FlextUtilitiesBeartypeTypingExtPatch._TypeHintSpecifier",
            origin[new_args],
        )


# Apply immediately so any subsequent beartype.claw usage sees both fixes.
FlextUtilitiesBeartypeTypingExtPatch.apply()

__all__: list[str] = ["FlextUtilitiesBeartypeTypingExtPatch"]
