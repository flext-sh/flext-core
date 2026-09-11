"""Generic PPTX byte serializer and object opener."""

from __future__ import annotations

from io import BytesIO
from zipfile import BadZipFile

from pptx import Presentation
from pptx.exc import PackageNotFoundError
from pptx.presentation import Presentation as PresentationType

from flext_cli import c, p, r


class FlextCliUtilitiesPptxSerializer:
    """Serialize and deserialize PPTX presentation objects."""

    # NOTE (multi-agent, mro-j2yt.1): consumers may pass python-pptx objects
    # inside this boundary; only bytes leave the boundary.

    @classmethod
    def pptx_save(cls, presentation: PresentationType) -> p.Result[bytes]:
        """Serialize a presentation object to bytes."""
        target = BytesIO()
        try:
            presentation.save(target)
        except (OSError, TypeError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[bytes].fail(f"{c.Cli.PptxError.SERIALIZE_FAILED}: {detail}")
        content = target.getvalue()
        if not content:
            return r[bytes].fail(str(c.Cli.PptxError.SERIALIZE_FAILED))
        return r[bytes].ok(content)

    @classmethod
    def pptx_open(cls, source: bytes) -> p.Result[PresentationType]:
        """Deserialize bytes into a presentation object."""
        try:
            presentation = Presentation(BytesIO(source))
        except (OSError, ValueError, KeyError, BadZipFile, PackageNotFoundError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[PresentationType].fail(
                f"{c.Cli.PptxError.PRESENTATION_LOAD_FAILED}: {detail}"
            )
        return r[PresentationType].ok(presentation)


__all__: tuple[str, ...] = ("FlextCliUtilitiesPptxSerializer",)
