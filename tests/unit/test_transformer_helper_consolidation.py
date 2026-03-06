"""Tests for HelperConsolidationTransformer."""

from __future__ import annotations

import libcst as cst

from flext_infra.refactor.transformers.helper_consolidation import (
    HelperConsolidationTransformer,
)


class TestHelperConsolidationTransformer:
    """Test suite for helper consolidation transformer."""

    def test_helper_becomes_staticmethod(self) -> None:
        """Test that loose function becomes @staticmethod."""
        source = '''
def util_helper(x: int) -> int:
    """Helper docstring."""
    return x * 2
'''
        mappings = {"util_helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)

        assert "@staticmethod" in modified.code
        assert "class FlextUtilities:" in modified.code
        assert "def util_helper" in modified.code

    def test_helper_type_hints_preserved(self) -> None:
        """Test that type hints are preserved."""
        source = """
def typed_helper(x: int, y: str) -> bool:
    return True
"""
        mappings = {"typed_helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)

        assert "x: int" in modified.code
        assert "y: str" in modified.code
        assert "-> bool" in modified.code

    def test_helper_docstring_preserved(self) -> None:
        """Test that docstrings are preserved."""
        source = '''
def documented_helper():
    """This is a documented helper."""
    pass
'''
        mappings = {"documented_helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)

        assert "This is a documented helper" in modified.code

    def test_multiple_helpers_same_namespace(self) -> None:
        """Test multiple helpers consolidated into same namespace."""
        source = """
def helper_one():
    pass

def helper_two():
    pass
"""
        mappings = {
            "helper_one": "FlextUtilities",
            "helper_two": "FlextUtilities",
        }
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)

        # Should have both helpers in same class
        assert modified.code.count("def helper_") == 2
        assert "class FlextUtilities:" in modified.code

    def test_unmapped_helper_preserved(self) -> None:
        """Test that unmapped helpers stay at module level."""
        source = """
def mapped_helper():
    pass

def unmapped_helper():
    pass
"""
        mappings = {"mapped_helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)

        # Unmapped helper should still be at module level
        assert "def unmapped_helper()" in modified.code
        # Should be outside class (no indentation before def)
        lines = modified.code.strip().split("\n")
        unmapped_line = next(line for line in lines if "unmapped_helper" in line)
        assert not unmapped_line.startswith(" ")

    def test_existing_namespace_extended(self) -> None:
        """Test that existing namespace class is extended."""
        source = """
class FlextUtilities:
    pass

def new_helper():
    pass
"""
        mappings = {"new_helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)

        # Should only have one class definition
        assert modified.code.count("class FlextUtilities:") == 1
        assert "def new_helper" in modified.code

    def test_helper_call_references_updated(self) -> None:
        """Test that internal calls to helpers are updated."""
        source = """
def helper():
    pass

def caller():
    helper()
"""
        mappings = {"helper": "FlextUtilities"}
        transformer = HelperConsolidationTransformer(mappings)
        module = cst.parse_module(source)
        modified = module.visit(transformer)

        # Call should be updated to namespaced version
        assert "FlextUtilities.helper()" in modified.code
