from __future__ import annotations

from flext_infra.refactor.mro_resolver import (
    FlextInfraRefactorMROError,
    FlextInfraRefactorMROResolver,
)


class FlextLdapModels:
    pass


class FlextCliModels:
    pass


class AlgarOudMigModels(FlextLdapModels, FlextCliModels):
    pass


class FlextLdapConstants:
    pass


class FlextCliConstants:
    pass


class AlgarOudMigConstants(FlextLdapConstants, FlextCliConstants):
    pass


class FlextLdapTypes:
    pass


class FlextCliTypes:
    pass


class AlgarOudMigTypes(FlextLdapTypes, FlextCliTypes):
    pass


class FlextLdapProtocols:
    pass


class FlextCliProtocols:
    pass


class AlgarOudMigProtocols(FlextLdapProtocols, FlextCliProtocols):
    pass


class FlextLdapUtilities:
    pass


class FlextCliUtilities:
    pass


class AlgarOudMigUtilities(FlextLdapUtilities, FlextCliUtilities):
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
    except FlextInfraRefactorMROError:
        raised = True
    assert raised
