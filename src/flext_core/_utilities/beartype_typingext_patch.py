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
import typing as _typing
from typing import Annotated, ClassVar, ForwardRef, get_args, get_origin

import typing_extensions as _typing_extensions

if _typing.TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesBeartypeTypingExtPatch:
    """Idempotent beartype patches for ``typing_extensions`` PEP 695 aliases."""

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
        cf = importlib.import_module("beartype._cave._cavefast")

        # Already patched or not applicable on this interpreter.
        if not hasattr(cf, "HintPep695TypeAlias"):
            return

        current = cf.HintPep695TypeAlias
        new_value = current
        te = _typing_extensions.TypeAliasType
        std = _typing.TypeAliasType

        if isinstance(current, tuple):
            if te not in current:
                new_value = (*current, te)
        elif current is std:
            if len({std, te}) > 1:
                new_value = (std, te)
        elif current is not te:
            new_value = (current, te)

        if new_value is not current:
            cf.HintPep695TypeAlias = new_value
            # Modules that did ``from _cavefast import HintPep695TypeAlias`` hold
            # a stale reference; patch only loaded beartype modules and avoid
            # module-level __getattr__ side effects during import-time patching.
            for mod_name, mod in tuple(_sys.modules.items()):
                if not mod_name.startswith("beartype"):
                    continue
                if mod.__dict__.get("HintPep695TypeAlias") is current:
                    mod.HintPep695TypeAlias = new_value

    @classmethod
    def _patch_forwardref_module_scope(cls) -> None:
        """Tag a reduced alias's stringified refs with the alias's module."""
        pep695 = importlib.import_module(
            "beartype._util.hint.pep.proposal.pep695",
        )
        original = pep695.get_hint_pep695_unsubbed_alias

        def tagged(
            hint: t.TypeHintSpecifier,
            exception_prefix: str = "",
        ) -> t.TypeHintSpecifier:
            reduced = original(hint, exception_prefix)
            module_name = getattr(hint, "__module__", None)
            if module_name:
                return cls._tag_forward_refs(reduced, module_name)
            return _typing.cast("t.TypeHintSpecifier", reduced)

        # Replace the getter across every beartype module that imported it by
        # name (``redpep695``) as well as its defining module.
        for mod_name, mod in tuple(_sys.modules.items()):
            if not mod_name.startswith("beartype"):
                continue
            if mod.__dict__.get("get_hint_pep695_unsubbed_alias") is original:
                mod.get_hint_pep695_unsubbed_alias = tagged

    @staticmethod
    def _tag_forward_refs(
        hint: t.TypeHintSpecifier,
        module_name: str,
    ) -> t.TypeHintSpecifier:
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
            return _typing.cast("t.TypeHintSpecifier", origin[new_args[0]])
        return _typing.cast("t.TypeHintSpecifier", origin[new_args])


# Apply immediately so any subsequent beartype.claw usage sees both fixes.
FlextUtilitiesBeartypeTypingExtPatch.apply()

__all__: list[str] = ["FlextUtilitiesBeartypeTypingExtPatch"]
