"""Generic typed XLSX service."""

from __future__ import annotations

from flext_cli._utilities.xlsx import FlextCliUtilitiesXlsx
from flext_cli.base import s


class FlextCliXlsx(s, FlextCliUtilitiesXlsx):
    """Expose byte-only XLSX operations for later public API composition."""

    # NOTE (multi-agent, mro-j2yt.1): this service contains no document or
    # customer rules; consumers provide immutable plans and receive bytes/models.


__all__: tuple[str, ...] = ("FlextCliXlsx",)
