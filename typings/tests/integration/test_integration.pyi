from unittest.mock import MagicMock

import pytest
from _typeshed import Incomplete

from flext_core import FlextContainer

pytestmark: Incomplete

class TestLibraryIntegration:
    @pytest.mark.integration
    @pytest.mark.core
    def test_all_exports_work(
        self,
        clean_container: FlextContainer,
        sample_data: dict[
            str, str | int | float | bool | list[int] | dict[str, str] | None
        ],
    ) -> None: ...
    @pytest.mark.integration
    @pytest.mark.core
    def test_flext_result_with_container(
        self, clean_container: FlextContainer, mock_external_service: MagicMock
    ) -> None: ...
    def test_entity_id_in_flext_result(self) -> None: ...
    def test_version_info_available(self) -> None: ...
