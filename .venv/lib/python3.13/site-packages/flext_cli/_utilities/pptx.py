"""Private MRO composition for the generic PPTX byte boundary."""

from __future__ import annotations

from ._pptx._reader import FlextCliUtilitiesPptxReader
from ._pptx._renderer import FlextCliUtilitiesPptxRenderer
from ._pptx._serializer import FlextCliUtilitiesPptxSerializer
from ._pptx._types import FlextCliUtilitiesPptxTypes


class FlextCliUtilitiesPptx(
    FlextCliUtilitiesPptxTypes,
    FlextCliUtilitiesPptxReader,
    FlextCliUtilitiesPptxSerializer,
    FlextCliUtilitiesPptxRenderer,
):
    """Compose reading, rendering, and re-exported types for generic PPTX bytes."""

    # NOTE (multi-agent, mro-j2yt.1): one MRO path exposes every generic PPTX
    # byte operation; consumers exchange only validated plans and bytes.


__all__: tuple[str, ...] = ("FlextCliUtilitiesPptx",)
