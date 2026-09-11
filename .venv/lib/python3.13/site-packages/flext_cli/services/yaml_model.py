"""Typed YAML model service."""

from __future__ import annotations

from flext_cli._utilities.yaml_model import FlextCliUtilitiesYamlModel
from flext_cli.base import s


class FlextCliYamlModel(s, FlextCliUtilitiesYamlModel):
    """Expose model-only YAML egress for public API composition."""

    # NOTE (multi-agent, mro-j2yt.1): public API MRO composition is deferred
    # until the private implementation and real round trip are independently green.


__all__: tuple[str, ...] = ("FlextCliYamlModel",)
