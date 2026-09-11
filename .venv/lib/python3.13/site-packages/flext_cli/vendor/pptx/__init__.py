"""Thin re-export boundary for python-pptx.

Import this package instead of ``pptx`` directly. The public ``Presentation``
function is exposed here; the underlying ``Presentation`` class is available in
``flext_cli._vendor.pptx.presentation`` for type annotations.
"""

from pptx import Presentation
from pptx.dml.color import ColorFormat, RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, MSO_VERTICAL_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import BaseOxmlElement
from pptx.shapes.autoshape import Shape
from pptx.shapes.picture import Picture
from pptx.slide import Slide, SlideLayout
from pptx.text.text import TextFrame
from pptx.util import Emu, Inches, Length, Pt

__all__ = [
    "Presentation",
    "ColorFormat",
    "RGBColor",
    "MSO_SHAPE",
    "MSO_ANCHOR",
    "MSO_AUTO_SIZE",
    "MSO_VERTICAL_ANCHOR",
    "PP_ALIGN",
    "qn",
    "BaseOxmlElement",
    "Shape",
    "Picture",
    "Slide",
    "SlideLayout",
    "TextFrame",
    "Emu",
    "Inches",
    "Length",
    "Pt",
]
