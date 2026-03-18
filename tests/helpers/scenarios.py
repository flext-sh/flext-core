from __future__ import annotations

from importlib import import_module

_impl = import_module("tests.helpers._scenarios_impl")


class TestHelperScenarios:
    ValidationScenario = _impl.ValidationScenario
    ParserScenario = _impl.ParserScenario
    ReliabilityScenario = _impl.ReliabilityScenario
    ValidationScenarios = _impl.ValidationScenarios
    ParserScenarios = _impl.ParserScenarios
    ReliabilityScenarios = _impl.ReliabilityScenarios
