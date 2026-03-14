"""Integration tests for policy-driven MRO resolution."""

from __future__ import annotations

from flext_infra.refactor.mro_resolver import FlextInfraRefactorMROResolver


class FlextLdapModels:
    """Stub LDAP models facade."""


class FlextCliModels:
    """Stub CLI models facade."""


class AlgarOudMigModels(FlextLdapModels, FlextCliModels):
    """Stub composed models facade."""


class FlextLdapConstants:
    """Stub LDAP constants facade."""


class FlextCliConstants:
    """Stub CLI constants facade."""


class AlgarOudMigConstants(FlextLdapConstants, FlextCliConstants):
    """Stub composed constants facade."""


class FlextLdapTypes:
    """Stub LDAP typings facade."""


class FlextCliTypes:
    """Stub CLI typings facade."""


class AlgarOudMigTypes(FlextLdapTypes, FlextCliTypes):
    """Stub composed typings facade."""


class FlextLdapProtocols:
    """Stub LDAP protocols facade."""


class FlextCliProtocols:
    """Stub CLI protocols facade."""


class AlgarOudMigProtocols(FlextLdapProtocols, FlextCliProtocols):
    """Stub composed protocols facade."""


class FlextLdapUtilities:
    """Stub LDAP utilities facade."""


class FlextCliUtilities:
    """Stub CLI utilities facade."""


class AlgarOudMigUtilities(FlextLdapUtilities, FlextCliUtilities):
    """Stub composed utilities facade."""


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
