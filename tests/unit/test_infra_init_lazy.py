"""Tests for flext_infra.__init__ lazy loading error paths.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import flext_infra
import flext_infra.basemk
import flext_infra.check
import flext_infra.codegen
import flext_infra.core
import flext_infra.deps
import flext_infra.docs
import flext_infra.github
import flext_infra.maintenance
import flext_infra.release
import flext_infra.workspace
import pytest


class TestFlextInfraInitLazyLoading:
    """Test __getattr__ error path in flext_infra.__init__."""

    def test_getattr_nonexistent_name_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute raises AttributeError."""
        with pytest.raises(AttributeError) as exc_info:
            _ = flext_infra.NonexistentAttribute

        assert "NonexistentAttribute" in str(exc_info.value)

    def test_getattr_invalid_name_raises_attribute_error(self) -> None:
        """Test that accessing invalid attribute raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.InvalidNameThatDoesNotExist

    def test_getattr_typo_in_name_raises_attribute_error(self) -> None:
        """Test that typos in attribute names raise AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.FlextInfraConstantsTypo

    def test_dir_returns_all_exports(self) -> None:
        """Test that dir() returns all exported names."""
        exports = dir(flext_infra)
        assert "FlextInfraConstants" in exports
        assert "FlextInfraModels" in exports
        assert "FlextInfraProtocols" in exports
        assert "FlextInfraTypes" in exports
        assert "FlextInfraUtilities" in exports

    def test_dir_returns_sorted_list(self) -> None:
        """Test that dir() returns a sorted list."""
        exports = dir(flext_infra)
        assert exports == sorted(exports)

    def test_dir_includes_version_exports(self) -> None:
        """Test that dir() includes version exports."""
        exports = dir(flext_infra)
        assert "__version__" in exports
        assert "__version_info__" in exports

    def test_dir_includes_runtime_aliases(self) -> None:
        """Test that dir() includes runtime aliases."""
        exports = dir(flext_infra)
        assert "c" in exports
        assert "m" in exports
        assert "p" in exports
        assert "t" in exports
        assert "u" in exports

    def test_lazy_import_flext_infra_constants(self) -> None:
        """Test lazy loading of FlextInfraConstants."""
        constants = flext_infra.FlextInfraConstants
        assert constants is not None

    def test_lazy_import_flext_infra_models(self) -> None:
        """Test lazy loading of FlextInfraModels."""
        models = flext_infra.FlextInfraModels
        assert models is not None

    def test_lazy_import_flext_infra_protocols(self) -> None:
        """Test lazy loading of FlextInfraProtocols."""
        protocols = flext_infra.FlextInfraProtocols
        assert protocols is not None

    def test_lazy_import_flext_infra_types(self) -> None:
        """Test lazy loading of FlextInfraTypes."""
        types = flext_infra.FlextInfraTypes
        assert types is not None

    def test_lazy_import_flext_infra_utilities(self) -> None:
        """Test lazy loading of FlextInfraUtilities."""
        utilities = flext_infra.FlextInfraUtilities
        assert utilities is not None

    def test_lazy_import_version_string(self) -> None:
        """Test lazy loading of __version__."""
        version = flext_infra.__version__
        assert isinstance(version, str)
        assert len(version) > 0

    def test_lazy_import_version_info(self) -> None:
        """Test lazy loading of __version_info__."""
        version_info = flext_infra.__version_info__
        assert isinstance(version_info, tuple)
        assert len(version_info) > 0

    def test_lazy_import_runtime_alias_c(self) -> None:
        """Test lazy loading of runtime alias c."""
        c = flext_infra.c
        assert c is not None

    def test_lazy_import_runtime_alias_m(self) -> None:
        """Test lazy loading of runtime alias m."""
        m = flext_infra.m
        assert m is not None

    def test_lazy_import_runtime_alias_p(self) -> None:
        """Test lazy loading of runtime alias p."""
        p = flext_infra.p
        assert p is not None

    def test_lazy_import_runtime_alias_t(self) -> None:
        """Test lazy loading of runtime alias t."""
        t = flext_infra.t
        assert t is not None

    def test_lazy_import_runtime_alias_u(self) -> None:
        """Test lazy loading of runtime alias u."""
        u = flext_infra.u
        assert u is not None

    def test_lazy_import_command_runner(self) -> None:
        """Test lazy loading of FlextInfraCommandRunner."""
        runner = flext_infra.FlextInfraCommandRunner
        assert runner is not None

    def test_lazy_import_discovery_service(self) -> None:
        """Test lazy loading of FlextInfraDiscoveryService."""
        service = flext_infra.FlextInfraDiscoveryService
        assert service is not None

    def test_lazy_import_git_service(self) -> None:
        """Test lazy loading of FlextInfraGitService."""
        service = flext_infra.FlextInfraGitService
        assert service is not None

    def test_lazy_import_json_service(self) -> None:
        """Test lazy loading of FlextInfraJsonService."""
        service = flext_infra.FlextInfraJsonService
        assert service is not None

    def test_lazy_import_toml_service(self) -> None:
        """Test lazy loading of FlextInfraTomlService."""
        service = flext_infra.FlextInfraTomlService
        assert service is not None

    def test_lazy_import_versioning_service(self) -> None:
        """Test lazy loading of FlextInfraVersioningService."""
        service = flext_infra.FlextInfraVersioningService
        assert service is not None

    def test_lazy_import_path_resolver(self) -> None:
        """Test lazy loading of FlextInfraPathResolver."""
        resolver = flext_infra.FlextInfraPathResolver
        assert resolver is not None

    def test_lazy_import_reporting_service(self) -> None:
        """Test lazy loading of FlextInfraReportingService."""
        service = flext_infra.FlextInfraReportingService
        assert service is not None

    def test_lazy_import_output(self) -> None:
        """Test lazy loading of output."""
        output = flext_infra.output
        assert output is not None

    def test_lazy_import_known_verbs(self) -> None:
        """Test lazy loading of KNOWN_VERBS."""
        verbs = flext_infra.KNOWN_VERBS
        assert verbs is not None

    def test_lazy_import_reports_dir_name(self) -> None:
        """Test lazy loading of REPORTS_DIR_NAME."""
        dir_name = flext_infra.REPORTS_DIR_NAME
        assert isinstance(dir_name, str)


class TestFlextInfraSubmoduleInitLazyLoading:
    """Test __getattr__ error paths in flext_infra submodule __init__ files."""

    def test_basemk_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in basemk raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.basemk.NonexistentAttribute

    def test_check_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in check raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.check.NonexistentAttribute

    def test_codegen_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in codegen raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.codegen.NonexistentAttribute

    def test_core_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in core raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.core.NonexistentAttribute

    def test_deps_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in deps raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.deps.NonexistentAttribute

    def test_docs_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in docs raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.docs.NonexistentAttribute

    def test_github_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in github raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.github.NonexistentAttribute

    def test_maintenance_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in maintenance raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.maintenance.NonexistentAttribute

    def test_release_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in release raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.release.NonexistentAttribute

    def test_workspace_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in workspace raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.workspace.NonexistentAttribute
