"""Re-export python-pptx public types so consumers avoid direct imports."""

from __future__ import annotations

from pptx import Presentation as _Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn as _qn
from pptx.oxml.xmlchemy import BaseOxmlElement
from pptx.presentation import Presentation as PresentationDocument
from pptx.shapes.autoshape import Shape
from pptx.shapes.picture import Picture
from pptx.slide import Slide, SlideLayout
from pptx.text.text import TextFrame
from pptx.util import Emu, Inches, Length, Pt


class FlextCliUtilitiesPptxTypes:
    """Namespace of python-pptx types exposed through the generic boundary."""

    # NOTE (multi-agent, mro-j2yt.1): re-exporting keeps the external PPTX
    # dependency owned by flext-cli so cosmos-docgen can drop direct imports.

    Presentation = staticmethod(_Presentation)
    PresentationDocument = PresentationDocument
    RGBColor = RGBColor
    MSO_SHAPE = MSO_SHAPE
    MSO_ANCHOR = MSO_ANCHOR
    MSO_AUTO_SIZE = MSO_AUTO_SIZE
    PP_ALIGN = PP_ALIGN
    qn = staticmethod(_qn)
    BaseOxmlElement = BaseOxmlElement
    Shape = Shape
    Picture = Picture
    Slide = Slide
    SlideLayout = SlideLayout
    TextFrame = TextFrame
    Emu = Emu
    Inches = Inches
    Length = Length
    Pt = Pt


__all__: tuple[str, ...] = ("FlextCliUtilitiesPptxTypes",)
