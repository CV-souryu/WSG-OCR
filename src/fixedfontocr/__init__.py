"""FixedFontOCR - deterministic CPU baseline OCR with a WGPU-ready model format."""

from .api import FixedFontOCR
from .types import CharResult, OCRResult, Profile
from .classifier import TemplateClassifier

__all__ = [
    "FixedFontOCR",
    "OCRResult",
    "CharResult",
    "Profile",
    "TemplateClassifier",
]

__version__ = "0.1.0"
