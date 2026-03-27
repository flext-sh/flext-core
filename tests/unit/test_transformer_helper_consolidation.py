"""Tests for HelperConsolidationTransformer."""

from __future__ import annotations

import libcst as cst
from flext_infra import (
    FlextInfraHelperConsolidationTransformer as HelperConsolidationTransformer,
)


class TestHelperConsolidationTransformer:
    """Test suite for helper consolidation transformer."""

    def test_helper_becomes_staticmethod(self) -> None:
        """Test that loose function becomes @staticmethod."""
        source = '\ndef util_helper(x: int) -> int:\n    """Helper docstring."""\n    return x * 2\n'
        mappings = {"util_helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)
        assert "@staticmethod" in modified.code
        assert "class FlextUtilities:" in modified.code
        assert "def util_helper" in modified.code

    def test_helper_type_hints_preserved(self) -> None:
        """Test that type hints are preserved."""
        source = "\ndef typed_helper(x: int, y: str) -> bool:\n    return True\n"
        mappings = {"typed_helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)
        assert "x: int" in modified.code
        assert "y: str" in modified.code
        assert "-> bool" in modified.code

    def test_helper_docstring_preserved(self) -> None:
        """Test that docstrings are preserved."""
        source = '\ndef documented_helper():\n    """This is a documented helper."""\n    pass\n'
        mappings = {"documented_helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)
        assert "This is a documented helper" in modified.code

    def test_multiple_helpers_same_namespace(self) -> None:
        """Test multiple helpers consolidated into same namespace."""
        source = "\ndef helper_one():\n    pass\n\ndef helper_two():\n    pass\n"
        mappings = {"helper_one": "FlextUtilities", "helper_two": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)
        assert modified.code.count("def helper_") == 2
        assert "class FlextUtilities:" in modified.code

    def test_unmapped_helper_preserved(self) -> None:
        """Test that unmapped helpers stay at module level."""
        source = (
            "\ndef mapped_helper():\n    pass\n\ndef unmapped_helper():\n    pass\n"
        )
        mappings = {"mapped_helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)
        assert "def unmapped_helper()" in modified.code
        lines = modified.code.strip().split("\n")
        unmapped_line = next(line for line in lines if "unmapped_helper" in line)
        assert not unmapped_line.startswith(" ")

    def test_existing_namespace_extended(self) -> None:
        """Test that existing namespace class is extended."""
        source = "\nclass FlextUtilities:\n    pass\n\ndef new_helper():\n    pass\n"
        mappings = {"new_helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)
        assert modified.code.count("class FlextUtilities:") == 1
        assert "def new_helper" in modified.code

    def test_helper_call_references_updated(self) -> None:
        """Test that internal calls to helpers are updated."""
        source = "\ndef helper():\n    pass\n\ndef caller():\n    helper()\n"
        mappings = {"helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)
        assert "FlextUtilities.helper()" in modified.code
