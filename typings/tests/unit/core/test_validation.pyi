import pytest
from _typeshed import Incomplete

from ...conftest import TestCase as TestCase, TestScenario as TestScenario

pytestmark: Incomplete

class TestFlextValidatorsAdvanced:
    @pytest.fixture
    def validation_test_cases(self) -> list[TestCase]: ...
    @pytest.mark.parametrize_advanced
    def test_validator_scenarios(
        self, validation_test_cases: list[TestCase]
    ) -> None: ...
    @pytest.fixture
    def email_validation_cases(self) -> list[TestCase]: ...
    @pytest.mark.parametrize_advanced
    def test_email_validation_scenarios(
        self, email_validation_cases: list[TestCase]
    ) -> None: ...
    @pytest.fixture
    def numeric_validation_cases(self) -> list[TestCase]: ...
    @pytest.mark.parametrize_advanced
    def test_numeric_validation_scenarios(
        self, numeric_validation_cases: list[TestCase]
    ) -> None: ...

class TestFlextPredicatesAdvanced:
    @pytest.fixture
    def predicate_test_cases(self) -> list[TestCase]: ...
    @pytest.mark.parametrize_advanced
    def test_predicate_scenarios(
        self, predicate_test_cases: list[TestCase]
    ) -> None: ...

class TestValidationSimpleIntegration:
    def test_email_validation_function(self) -> None: ...
    def test_non_empty_string_validation_function(self) -> None: ...
