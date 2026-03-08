"""Integration tests for policy-driven MRO resolution."""

from __future__ import annotations

from flext_infra.refactor.mro_resolver import (
    FlextInfraRefactorMROResolver,
)


class FlextLdapModels:
    """Stub LDAP models facade."""

    pass


class FlextCliModels:
    """Stub CLI models facade."""

    pass


class AlgarOudMigModels(FlextLdapModels, FlextCliModels):
    """Stub composed models facade."""

    pass


class FlextLdapConstants:
    """Stub LDAP constants facade."""

    pass


class FlextCliConstants:
    """Stub CLI constants facade."""

    pass


class AlgarOudMigConstants(FlextLdapConstants, FlextCliConstants):
    """Stub composed constants facade."""

    pass


class FlextLdapTypes:
    """Stub LDAP typings facade."""

    pass


class FlextCliTypes:
    """Stub CLI typings facade."""

    pass


class AlgarOudMigTypes(FlextLdapTypes, FlextCliTypes):
    """Stub composed typings facade."""

    pass


class FlextLdapProtocols:
    """Stub LDAP protocols facade."""

    pass


class FlextCliProtocols:
    """Stub CLI protocols facade."""

    pass


class AlgarOudMigProtocols(FlextLdapProtocols, FlextCliProtocols):
    """Stub composed protocols facade."""

    pass


class FlextLdapUtilities:
    """Stub LDAP utilities facade."""

    pass


class FlextCliUtilities:
    """Stub CLI utilities facade."""

    pass


class AlgarOudMigUtilities(FlextLdapUtilities, FlextCliUtilities):
    """Stub composed utilities facade."""

    pass


def test_mro_resolver_accepts_expected_order() -> None:
    resolutions = FlextInfraRefactorMROResolver.resolve(
        family_classes={
            "c": AlgarOudMigConstants,
            "t": AlgarOudMigTypes,
            "p": AlgarOudMigProtocols,
            "m": AlgarOudMigModels,
            "u": AlgarOudMigUtilities,
        },
        expected_base_chains={
            "c": ["FlextLdapConstants", "FlextCliConstants"],
            "t": ["FlextLdapTypes", "FlextCliTypes"],
            "p": ["FlextLdapProtocols", "FlextCliProtocols"],
            "m": ["FlextLdapModels", "FlextCliModels"],
            "u": ["FlextLdapUtilities", "FlextCliUtilities"],
        },
    )
    assert len(resolutions) == 5
    model_resolution = next(res for res in resolutions if res.family == "m")
    assert model_resolution.expected_bases == ("FlextLdapModels", "FlextCliModels")


def test_mro_resolver_rejects_wrong_order() -> None:
    raised = False
    try:
        FlextInfraRefactorMROResolver.resolve(
            family_classes={
                "c": AlgarOudMigConstants,
                "t": AlgarOudMigTypes,
                "p": AlgarOudMigProtocols,
                "m": AlgarOudMigModels,
                "u": AlgarOudMigUtilities,
            },
            expected_base_chains={
                "c": ["FlextLdapConstants", "FlextCliConstants"],
                "t": ["FlextLdapTypes", "FlextCliTypes"],
                "p": ["FlextLdapProtocols", "FlextCliProtocols"],
                "m": ["FlextCliModels", "FlextLdapModels"],
                "u": ["FlextLdapUtilities", "FlextCliUtilities"],
            },
        )
    except ValueError:
        raised = True
    assert raised
