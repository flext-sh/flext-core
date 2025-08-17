from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flext_core import FlextContainer

class TestServiceIntegrationPatterns:
    @pytest.mark.integration
    @pytest.mark.performance
    def test_service_pipeline_performance(
        self,
        configured_container: FlextContainer,
        mock_external_service: MagicMock,
        performance_threshold: dict[str, float],
        benchmark_data: dict[
            str, list[int] | dict[str, str] | dict[str, dict[str, dict[str, list[int]]]]
        ],
    ) -> None: ...
    @pytest.mark.integration
    @pytest.mark.error_path
    def test_service_error_propagation(
        self,
        configured_container: FlextContainer,
        mock_external_service: MagicMock,
        error_context: dict[str, str | None],
    ) -> None: ...
    @pytest.mark.integration
    @pytest.mark.architecture
    def test_dependency_injection_with_mocks(
        self,
        clean_container: FlextContainer,
        test_user_data: dict[str, str | int | bool | list[str]],
    ) -> None: ...
    @pytest.mark.integration
    @pytest.mark.boundary
    def test_container_service_lifecycle(
        self, clean_container: FlextContainer, temp_directory: Path
    ) -> None: ...
