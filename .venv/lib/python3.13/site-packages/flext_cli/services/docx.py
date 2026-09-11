"""Generic typed DOCX service."""

from __future__ import annotations

from flext_cli._utilities.docx import FlextCliUtilitiesDocx
from flext_cli.base import s


class FlextCliDocx(s, FlextCliUtilitiesDocx):
    """Expose byte-only DOCX operations for later public API composition."""

    # NOTE (multi-agent, mro-j2yt.1): this service contains no document or
    # customer rules; consumers provide immutable plans and receive bytes/models.


__all__: tuple[str, ...] = ("FlextCliDocx",)
