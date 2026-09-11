"""Private MRO composition for the generic DOCX byte boundary."""

from __future__ import annotations

from ._docx._reader import FlextCliUtilitiesDocxReader
from ._docx._renderer import FlextCliUtilitiesDocxRenderer


class FlextCliUtilitiesDocx(FlextCliUtilitiesDocxReader, FlextCliUtilitiesDocxRenderer):
    """Compose reading and rendering operations for generic DOCX bytes."""

    # NOTE (multi-agent, mro-j2yt.1): one MRO path exposes every generic DOCX
    # byte operation; consumers exchange only validated plans and bytes.


__all__: tuple[str, ...] = ("FlextCliUtilitiesDocx",)
