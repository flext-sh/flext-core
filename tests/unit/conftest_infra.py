from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra import (
    FlextInfraUtilitiesGit,
    FlextInfraUtilitiesIo,
    FlextInfraUtilitiesPaths,
    FlextInfraUtilitiesPatterns,
    FlextInfraUtilitiesReporting,
    FlextInfraUtilitiesSelection,
    FlextInfraUtilitiesSubprocess,
    FlextInfraUtilitiesTemplates,
    FlextInfraUtilitiesToml,
)


@pytest.fixture(autouse=True)
def infra_test_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    src_pkg = workspace / "src" / "infra_pkg"
    src_pkg.mkdir(parents=True, exist_ok=True)
    (workspace / "pyproject.toml").write_text(
        "[project]\nname='infra-pkg'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    (workspace / "Makefile").write_text("help:\n\t@pwd\n", encoding="utf-8")
    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    return workspace


@pytest.fixture
def infra_subprocess() -> FlextInfraUtilitiesSubprocess:
    return FlextInfraUtilitiesSubprocess()


@pytest.fixture
def infra_toml() -> FlextInfraUtilitiesToml:
    return FlextInfraUtilitiesToml()


@pytest.fixture
def infra_git() -> FlextInfraUtilitiesGit:
    return FlextInfraUtilitiesGit()


@pytest.fixture
def infra_io() -> FlextInfraUtilitiesIo:
    return FlextInfraUtilitiesIo()


@pytest.fixture
def infra_path() -> FlextInfraUtilitiesPaths:
    return FlextInfraUtilitiesPaths()


@pytest.fixture
def infra_patterns() -> FlextInfraUtilitiesPatterns:
    return FlextInfraUtilitiesPatterns()


@pytest.fixture
def infra_selection() -> FlextInfraUtilitiesSelection:
    return FlextInfraUtilitiesSelection()


@pytest.fixture
def infra_reporting() -> FlextInfraUtilitiesReporting:
    return FlextInfraUtilitiesReporting()


@pytest.fixture
def infra_templates() -> FlextInfraUtilitiesTemplates:
    return FlextInfraUtilitiesTemplates()


@pytest.fixture
def infra_safe_command_output(
    infra_subprocess: FlextInfraUtilitiesSubprocess,
    infra_test_workspace: Path,
) -> str:
    echo_result = infra_subprocess.capture(
        ["echo", "infra-ok"],
        cwd=infra_test_workspace,
    )
    assert echo_result.is_success
    pwd_result = infra_subprocess.capture(["pwd"], cwd=infra_test_workspace)
    assert pwd_result.is_success
    return f"{echo_result.value.strip()}|{pwd_result.value.strip()}"


@pytest.fixture
def infra_git_repo(
    infra_subprocess: FlextInfraUtilitiesSubprocess,
    infra_test_workspace: Path,
) -> Path:
    repo = infra_test_workspace / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    assert infra_subprocess.run_checked(["git", "init"], cwd=repo).is_success
    assert infra_subprocess.run_checked(
        ["git", "config", "user.email", "infra@example.com"],
        cwd=repo,
    ).is_success
    assert infra_subprocess.run_checked(
        ["git", "config", "user.name", "Infra Fixtures"],
        cwd=repo,
    ).is_success
    return repo
