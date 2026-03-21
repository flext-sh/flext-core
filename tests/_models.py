"""Auto-generated centralized models."""

from __future__ import annotations

from pydantic import BaseModel


class EmailResponse(BaseModel):
    """Email response model."""

    status: str
    message_id: str
