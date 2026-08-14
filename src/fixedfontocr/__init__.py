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
from .geometry import FontGeometryDatabase
from .lexicon import (
    LEXICON_DOMAINS,
    LEXICON_FILES,
    Lexicon,
    LexiconLayer,
    apply_lexicon,
    load_lexicon,
)
from .model import write_hybrid_model
from .postprocess import pick
from .scorer import (
    VisualCalibration,
    VisualWeights,
    calibrate_visual,
    unified_visual_score,
)

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
    "FontGeometryDatabase",
    "Lexicon",
    "LexiconLayer",
    "LEXICON_FILES",
    "LEXICON_DOMAINS",
    "load_lexicon",
    "apply_lexicon",
    "extract_frontend",
    "Profile",
    "TemplateClassifier",
    "benchmark_backends",
    "pick",
    "write_hybrid_model",
    "VisualWeights",
    "VisualCalibration",
    "unified_visual_score",
    "calibrate_visual",
]

__version__ = "0.1.0"
