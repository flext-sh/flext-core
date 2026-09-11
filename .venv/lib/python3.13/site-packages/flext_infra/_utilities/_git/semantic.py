"""Canonical Git semantic mixin — re-exported from the modular chain.

This module previously held a monolithic 798-line FlextInfraUtilitiesGitSemanticMixin
that duplicated the content of the 7 modular semantic_*.py mixins. The modular chain
(top: FlextInfraUtilitiesGitSemanticIndexMixin) is the canonical owner.
"""

from __future__ import annotations

from flext_infra._utilities._git.semantic_index import (
    FlextInfraUtilitiesGitSemanticIndexMixin,
)

FlextInfraUtilitiesGitSemanticMixin = FlextInfraUtilitiesGitSemanticIndexMixin

__all__: list[str] = ["FlextInfraUtilitiesGitSemanticMixin"]
