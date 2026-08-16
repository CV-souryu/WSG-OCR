"""Goal 2: the Visual Frontend.

The old pipeline derived a single binary mask from the RGB image and fed
that one representation to every downstream consumer. The Visual Frontend
instead extracts both representations in one pass over
``np.ndarray[H, W, 3]``:

    RGB
     |
     +-> binary mask      -> connected components / font geometry / Template
     |
     +-> soft foreground  -> TinyCNN

The binary mask is a hard decision and keeps the segmentation/template path
deterministic. The soft foreground is a 0..255 foreground-strength map with
no hard cutoff, so anti-aliasing, alpha, edge gray and low-resolution
intensity information survive for the CNN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .preprocess import NormalizeSpec, glyph_normalize_geometry, normalize, normalize_grayscale
from .types import Component, Profile


@dataclass
class VisualFrontend:
    """The two representations extracted from one RGB image.

    Attributes
    ----------
    image:
        The validated ``uint8 [H, W, 3]`` input.
    binary_mask:
        Boolean ink mask (``color_mask``), consumed by segmentation, font
        geometry and the template matcher.
    soft_foreground:
        ``uint8 [H, W]`` foreground strength in 0..255, consumed by the
        TinyCNN. Edge pixels keep intermediate intensities instead of being
        snapped to the binarization threshold.
    profile:
        The profile used for both extractions.
    """

    image: NDArray[np.uint8]
    binary_mask: NDArray[np.bool_]
    soft_foreground: NDArray[np.uint8]
    profile: Profile

    def binary_glyph(
        self,
        component: Component,
        target: int | None = None,
        geometry: tuple[float, float] | None = None,
        spec: NormalizeSpec | None = None,
    ) -> NDArray[np.uint8]:
        """Normalize one component's binary mask to ``target x target``.

        ``geometry`` is the Goal 3 ``(baseline_offset, scale)`` pair from
        :func:`glyph_normalize_geometry`; without it the legacy centered
        normalization is used.
        """

        t = self._target(target)
        if geometry is None or spec is None:
            return normalize(component.mask, t)
        row = spec.baseline_row
        return normalize(
            component.mask,
            t,
            baseline_offset=geometry[0],
            scale=geometry[1],
            baseline_row=row,
        )

    def soft_glyph(
        self,
        component: Component,
        target: int | None = None,
        geometry: tuple[float, float] | None = None,
        spec: NormalizeSpec | None = None,
    ) -> NDArray[np.uint8]:
        """Normalize one component's soft ROI to ``target x target``.

        The ROI is cropped from :attr:`soft_foreground` using the
        component's image-space bbox, so the CNN sees the same anti-aliased
        intensities that were present before binarization.
        """

        y0 = int(component.y)
        y1 = y0 + int(component.h)
        x0 = int(component.x)
        x1 = x0 + int(component.w)
        roi = self.soft_foreground[y0:y1, x0:x1]
        t = self._target(target)
        if geometry is None or spec is None:
            return normalize_grayscale(roi, t)
        row = spec.baseline_row
        return normalize_grayscale(
            roi,
            t,
            baseline_offset=geometry[0],
            scale=geometry[1],
            baseline_row=row,
        )

    def soft_glyph_batch(
        self,
        components: list[Component],
        target: int | None = None,
        geometries: list[tuple[float, float] | None] | None = None,
        spec: NormalizeSpec | None = None,
    ) -> NDArray[np.uint8]:
        """Stack soft-normalized glyphs for the whole component batch."""

        if not components:
            t = self._target(target)
            return np.empty((0, t, t), dtype=np.uint8)
        geoms = geometries or [None] * len(components)
        return np.stack(
            [
                self.soft_glyph(c, target, geometry=g, spec=spec)
                for c, g in zip(components, geoms)
            ]
        )

    def _target(self, target: int | None) -> int:
        return self.profile.target_size if target is None else int(target)


def extract_frontend(
    image: NDArray[np.uint8],
    profile: Profile,
    with_soft: bool = True,
) -> VisualFrontend:
    """Extract binary + soft foreground from one RGB image in a single pass.

    Parameters
    ----------
    image:
        ``uint8 [H, W, 3]`` RGB input.
    profile:
        Segmentation/color profile defining both the hard mask and the soft
        foreground-strength map.

    Returns
    -------
    VisualFrontend
        A frontend object holding both representations; callers pass
        ``binary_mask`` to segmentation/template and ``soft_foreground`` to
        the TinyCNN.
    """

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {image.dtype}")

    return VisualFrontend(
        image=image,
        binary_mask=profile.color_mask(image),
        soft_foreground=profile.soft_foreground(image) if with_soft else None,
        profile=profile,
    )
