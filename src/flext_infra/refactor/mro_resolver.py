"""Resolve and validate facade-family MRO chains."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from flext_infra.refactor.project_classifier import ProjectClassification

type FlextInfraRefactorFacadeFamily = Literal["c", "t", "p", "m", "u"]
type FlextInfraRefactorExpectedBase = type | str


class FlextInfraRefactorMROError(RuntimeError):
    """Raised when MRO resolution violates policy constraints."""

    pass


@dataclass(frozen=True, slots=True)
class FlextInfraRefactorFamilyMROResolution:
    """Resolution payload for one facade family."""

    family: FlextInfraRefactorFacadeFamily
    expected_bases: tuple[str, ...]
    resolved_mro: tuple[str, ...]
    accessible_namespaces: tuple[str, ...]


class FlextInfraRefactorMROResolver:
    """MRO resolver for c/t/p/m/u facade families."""

    _FAMILY_SUFFIXES: Final[MappingProxyType[FlextInfraRefactorFacadeFamily, str]] = (
        MappingProxyType({
            "c": "Constants",
            "t": "Types",
            "p": "Protocols",
            "m": "Models",
            "u": "Utilities",
        })
    )

    @classmethod
    def resolve(
        cls,
        *,
        family_classes: Mapping[FlextInfraRefactorFacadeFamily, type],
        expected_base_chains: Mapping[
            FlextInfraRefactorFacadeFamily,
            Sequence[FlextInfraRefactorExpectedBase],
        ],
    ) -> tuple[FlextInfraRefactorFamilyMROResolution, ...]:
        """Resolve and validate MRO for all facade families."""
        resolutions: list[FlextInfraRefactorFamilyMROResolution] = []
        for family in ("c", "t", "p", "m", "u"):
            facade_class = family_classes[family]
            expected_chain = expected_base_chains[family]
            resolutions.append(
                cls._resolve_family(
                    family=family,
                    facade_class=facade_class,
                    expected_chain=expected_chain,
                )
            )
        return tuple(resolutions)

    @classmethod
    def resolve_from_classification(
        cls,
        *,
        family_classes: Mapping[FlextInfraRefactorFacadeFamily, type],
        classification: ProjectClassification,
    ) -> tuple[FlextInfraRefactorFamilyMROResolution, ...]:
        """Resolve MRO using family chains produced by project classifier."""
        expected_base_chains = cls._normalize_classifier_chains(
            family_chains=classification.family_chains,
        )
        return cls.resolve(
            family_classes=family_classes,
            expected_base_chains=expected_base_chains,
        )

    @classmethod
    def _resolve_family(
        cls,
        *,
        family: FlextInfraRefactorFacadeFamily,
        facade_class: type,
        expected_chain: Sequence[FlextInfraRefactorExpectedBase],
    ) -> FlextInfraRefactorFamilyMROResolution:
        expected_names = cls._normalize_expected_chain(expected_chain=expected_chain)
        cls._validate_base_policy(
            family=family,
            facade_class=facade_class,
            expected_names=expected_names,
        )

        resolved_mro = tuple(entry.__name__ for entry in inspect.getmro(facade_class))
        accessible_namespaces = cls._collect_accessible_namespaces(
            family=family,
            facade_class=facade_class,
        )
        cls._validate_expected_accessibility(
            family=family,
            expected_names=expected_names,
            accessible_namespaces=accessible_namespaces,
        )

        return FlextInfraRefactorFamilyMROResolution(
            family=family,
            expected_bases=expected_names,
            resolved_mro=resolved_mro,
            accessible_namespaces=accessible_namespaces,
        )

    @classmethod
    def _normalize_classifier_chains(
        cls,
        *,
        family_chains: Mapping[str, Sequence[str]],
    ) -> dict[FlextInfraRefactorFacadeFamily, tuple[str, ...]]:
        normalized: dict[FlextInfraRefactorFacadeFamily, tuple[str, ...]] = {}
        for family in ("c", "t", "p", "m", "u"):
            raw_chain = family_chains.get(family)
            if raw_chain is None:
                msg = f"Missing expected family chain for {family!r}."
                raise FlextInfraRefactorMROError(msg)
            normalized[family] = tuple(raw_chain)
        return normalized

    @classmethod
    def _normalize_expected_chain(
        cls,
        *,
        expected_chain: Sequence[FlextInfraRefactorExpectedBase],
    ) -> tuple[str, ...]:
        expected_names: list[str] = []
        for base in expected_chain:
            if isinstance(base, str):
                expected_names.append(base)
                continue
            expected_names.append(base.__name__)
        return tuple(expected_names)

    @classmethod
    def _validate_base_policy(
        cls,
        *,
        family: FlextInfraRefactorFacadeFamily,
        facade_class: type,
        expected_names: tuple[str, ...],
    ) -> None:
        direct_base_names = tuple(base.__name__ for base in facade_class.__bases__)
        if len(direct_base_names) < len(expected_names):
            msg = (
                f"family={family} has fewer direct bases than expected: "
                f"expected={expected_names!r} direct={direct_base_names!r}"
            )
            raise FlextInfraRefactorMROError(msg)

        if direct_base_names[: len(expected_names)] != expected_names:
            msg = (
                f"family={family} direct base order violates policy: "
                f"expected={expected_names!r} direct={direct_base_names!r}"
            )
            raise FlextInfraRefactorMROError(msg)

        mro_types = inspect.getmro(facade_class)
        mro_names = tuple(entry.__name__ for entry in mro_types)
        mro_index = {name: index for index, name in enumerate(mro_names)}

        missing = tuple(name for name in expected_names if name not in mro_index)
        if missing:
            msg = (
                f"family={family} missing expected bases in MRO: "
                f"missing={missing!r} mro={mro_names!r}"
            )
            raise FlextInfraRefactorMROError(msg)

        previous_index = -1
        for base_name in expected_names:
            current_index = mro_index[base_name]
            if current_index <= previous_index:
                msg = (
                    f"family={family} MRO order is not C3-coherent for expected chain: "
                    f"expected={expected_names!r} mro={mro_names!r}"
                )
                raise FlextInfraRefactorMROError(msg)
            previous_index = current_index

    @classmethod
    def _validate_expected_accessibility(
        cls,
        *,
        family: FlextInfraRefactorFacadeFamily,
        expected_names: tuple[str, ...],
        accessible_namespaces: tuple[str, ...],
    ) -> None:
        missing_namespaces: list[str] = []
        for base_name in expected_names:
            namespace = cls._namespace_from_class_name(
                class_name=base_name,
                family=family,
            )
            if namespace is None:
                continue
            if namespace in accessible_namespaces:
                continue
            missing_namespaces.append(namespace)

        if missing_namespaces:
            msg = (
                f"family={family} expected namespaces are not accessible: "
                f"missing={tuple(missing_namespaces)!r} "
                f"accessible={accessible_namespaces!r}"
            )
            raise FlextInfraRefactorMROError(msg)

    @classmethod
    def _collect_accessible_namespaces(
        cls,
        *,
        family: FlextInfraRefactorFacadeFamily,
        facade_class: type,
    ) -> tuple[str, ...]:
        namespace_order: list[str] = []
        for current in inspect.getmro(facade_class):
            if current is object:
                continue

            class_namespace = cls._namespace_from_class_name(
                class_name=current.__name__,
                family=family,
            )
            if class_namespace is not None:
                cls._append_unique(namespace_order, class_namespace)

            for member_name, member in vars(current).items():
                if member_name.startswith("_"):
                    continue
                if not isinstance(member, type):
                    continue
                cls._append_unique(namespace_order, member_name)

        return tuple(namespace_order)

    @classmethod
    def _namespace_from_class_name(
        cls,
        *,
        class_name: str,
        family: FlextInfraRefactorFacadeFamily,
    ) -> str | None:
        suffix = cls._FAMILY_SUFFIXES[family]
        if not class_name.endswith(suffix):
            return None

        root = class_name[: -len(suffix)]
        root = root.removeprefix("Flext")
        if not root:
            return None
        return root

    @staticmethod
    def _append_unique(namespaces: list[str], candidate: str) -> None:
        if candidate not in namespaces:
            namespaces.append(candidate)


__all__ = [
    "FlextInfraRefactorExpectedBase",
    "FlextInfraRefactorFacadeFamily",
    "FlextInfraRefactorFamilyMROResolution",
    "FlextInfraRefactorMROError",
    "FlextInfraRefactorMROResolver",
]
