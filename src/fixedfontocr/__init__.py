"""FixedFontOCR - deterministic fixed-font OCR with CPU/WGPU backends."""

from .api import FixedFontOCR
from .backends import (
    AutoBackend,
    Backend,
    BackendResult,
    CPUBackend,
    WGPUBackend,
    benchmark_backends,
)
from .types import CharResult, OCRResult, Profile
from .classifier import TemplateClassifier
from .model import write_hybrid_model
from .postprocess import pick

__all__ = [
    "FixedFontOCR",
    "AutoBackend",
    "Backend",
    "BackendResult",
    "CPUBackend",
    "WGPUBackend",
    "OCRResult",
    "CharResult",
    "Profile",
    "TemplateClassifier",
    "benchmark_backends",
    "pick",
    "write_hybrid_model",
]

__version__ = "0.1.0"
