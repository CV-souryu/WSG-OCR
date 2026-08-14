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
from .types import (
    CharResult,
    Component,
    DecodePath,
    LexiconMatch,
    OCRResult,
    Profile,
    VisualCandidate,
    VisualLattice,
    VisualScores,
)
from .classifier import TemplateClassifier
from .frontend import VisualFrontend, extract_frontend
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
    "Component",
    "VisualCandidate",
    "VisualLattice",
    "VisualScores",
    "DecodePath",
    "LexiconMatch",
    "VisualFrontend",
    "extract_frontend",
    "Profile",
    "TemplateClassifier",
    "benchmark_backends",
    "pick",
    "write_hybrid_model",
]

__version__ = "0.1.0"
