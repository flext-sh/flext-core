"""Integration tests for policy-driven MRO resolution."""

from __future__ import annotations

from flext_infra import FlextInfraConstants, FlextInfraRefactorMROResolver


class TestRefactorPolicyMRO:
    """Single top-level integration test class for MRO policy resolution."""

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

    def test_mro_resolver_accepts_expected_order(self) -> None:
        resolutions = FlextInfraRefactorMROResolver.resolve(
            family_classes={
                FlextInfraConstants.Infra.FacadeFamily.C: self.AlgarOudMigConstants,
                FlextInfraConstants.Infra.FacadeFamily.T: self.AlgarOudMigTypes,
                FlextInfraConstants.Infra.FacadeFamily.P: self.AlgarOudMigProtocols,
                FlextInfraConstants.Infra.FacadeFamily.M: self.AlgarOudMigModels,
                FlextInfraConstants.Infra.FacadeFamily.U: self.AlgarOudMigUtilities,
            },
            expected_base_chains={
                FlextInfraConstants.Infra.FacadeFamily.C: [
                    "FlextLdapConstants",
                    "FlextCliConstants",
                ],
                FlextInfraConstants.Infra.FacadeFamily.T: [
                    "FlextLdapTypes",
                    "FlextCliTypes",
                ],
                FlextInfraConstants.Infra.FacadeFamily.P: [
                    "FlextLdapProtocols",
                    "FlextCliProtocols",
                ],
                FlextInfraConstants.Infra.FacadeFamily.M: [
                    "FlextLdapModels",
                    "FlextCliModels",
                ],
                FlextInfraConstants.Infra.FacadeFamily.U: [
                    "FlextLdapUtilities",
                    "FlextCliUtilities",
                ],
            },
        )
        assert len(resolutions) == 5
        model_resolution = next(res for res in resolutions if res.family == "m")
        assert model_resolution.expected_bases == ("FlextLdapModels", "FlextCliModels")

    def test_mro_resolver_rejects_wrong_order(self) -> None:
        raised = False
        try:
            FlextInfraRefactorMROResolver.resolve(
                family_classes={
                    FlextInfraConstants.Infra.FacadeFamily.C: (
                        self.AlgarOudMigConstants
                    ),
                    FlextInfraConstants.Infra.FacadeFamily.T: self.AlgarOudMigTypes,
                    FlextInfraConstants.Infra.FacadeFamily.P: (
                        self.AlgarOudMigProtocols
                    ),
                    FlextInfraConstants.Infra.FacadeFamily.M: self.AlgarOudMigModels,
                    FlextInfraConstants.Infra.FacadeFamily.U: (
                        self.AlgarOudMigUtilities
                    ),
                },
                expected_base_chains={
                    FlextInfraConstants.Infra.FacadeFamily.C: [
                        "FlextLdapConstants",
                        "FlextCliConstants",
                    ],
                    FlextInfraConstants.Infra.FacadeFamily.T: [
                        "FlextLdapTypes",
                        "FlextCliTypes",
                    ],
                    FlextInfraConstants.Infra.FacadeFamily.P: [
                        "FlextLdapProtocols",
                        "FlextCliProtocols",
                    ],
                    FlextInfraConstants.Infra.FacadeFamily.M: [
                        "FlextCliModels",
                        "FlextLdapModels",
                    ],
                    FlextInfraConstants.Infra.FacadeFamily.U: [
                        "FlextLdapUtilities",
                        "FlextCliUtilities",
                    ],
                },
            )
        except ValueError:
            raised = True
        assert raised
