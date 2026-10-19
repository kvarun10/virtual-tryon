"""Image / label-map loading, resizing and normalisation.

Conventions used everywhere in this project
-------------------------------------------
* Images in memory are ``H x W x 3`` uint8 **RGB** numpy arrays (PIL order). OpenCV-based
  code (SCHP) converts to BGR internally and nowhere else.
* Label maps are ``H x W`` uint8 class indices (never colour images).
* Model tensors are ``C x H x W`` float32 in **[-1, 1]** (``(x / 255 - 0.5) / 0.5``), the usual
  range for GAN generators with a tanh output.
* Images are resized with bilinear/area interpolation; label maps ALWAYS with nearest
  neighbour, because averaging class indices produces meaningless classes.
* ``letterbox`` keeps the aspect ratio and pads (DeepFashion images are 750 x 1101 portrait, a
  plain stretch to 256 x 256 would squash bodies horizontally). The returned
  :class:`ResizeInfo` allows mapping results back to the original frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

PathLike = Union[str, Path]


class ImageLoadError(IOError):
    """Raised for missing, unreadable or malformed image / label files."""


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_image(path_or_image: Union[PathLike, Image.Image, np.ndarray]) -> np.ndarray:
    """Load an RGB uint8 array from a path, a PIL image or an array.

    EXIF orientation is applied (phone photos are often stored rotated); alpha channels are
    composited onto white; greyscale is expanded to 3 channels.
    """
    if isinstance(path_or_image, np.ndarray):
        arr = path_or_image
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ImageLoadError(f"Expected H x W x 3 image array, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            raise ImageLoadError(f"Expected uint8 image array, got {arr.dtype}")
        return np.ascontiguousarray(arr[:, :, :3])
    if isinstance(path_or_image, Image.Image):
        img = path_or_image
    else:
        path = Path(path_or_image)
        if not path.is_file():
            raise ImageLoadError(f"Image file not found: {path}")
        try:
            img = Image.open(path)
            img.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageLoadError(f"Cannot decode image {path}: {exc}") from exc
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


def load_label_map(path: PathLike, channel: int = None) -> np.ndarray:
    """Load a label PNG as ``H x W`` uint8 class indices.

    * mode ``L`` / ``P``: the stored values ARE the class indices (the palette of a ``P`` image is
      only for display and is ignored).
    * multi-channel images require ``channel`` (e.g. DensePose IUV PNGs); otherwise this raises,
      because a colour-coded map would need :func:`src.data.labels.decode_color_map` instead.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Label file not found: {path}")
    try:
        img = Image.open(path)
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError(f"Cannot decode label map {path}: {exc}") from exc
    if img.mode in ("L", "P", "I", "I;16"):
        arr = np.asarray(img)
        if channel not in (None, 0):
            raise ImageLoadError(f"{path} is single-channel; channel={channel} is invalid")
    else:
        arr = np.asarray(img)
        if channel is None:
            raise ImageLoadError(
                f"{path} has mode {img.mode} (multi-channel). Pass channel=<index> for packed maps "
                "such as DensePose, or decode colour-coded maps with decode_color_map()."
            )
        if arr.ndim != 3 or channel >= arr.shape[2]:
            raise ImageLoadError(f"{path}: channel {channel} not available in shape {arr.shape}")
        arr = arr[:, :, channel]
    if arr.max(initial=0) > 255:
        raise ImageLoadError(f"{path}: label values exceed 255")
    return arr.astype(np.uint8)


def image_header_ok(path: PathLike) -> Tuple[bool, str]:
    """Cheap integrity check: parse the header without decoding pixels."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except Exception as exc:  # noqa: BLE001 - report any failure
        return False, f"{type(exc).__name__}: {exc}"


def image_full_ok(path: PathLike) -> Tuple[bool, str]:
    """Full decode check (catches truncated files that pass ``verify``)."""
    try:
        with Image.open(path) as img:
            img.load()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------------------
# Resizing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResizeInfo:
    """How an ``orig_size`` image was placed into an ``out_size`` canvas (sizes are (H, W))."""

    orig_size: Tuple[int, int]
    out_size: Tuple[int, int]
    scale_y: float
    scale_x: float
    pad_top: int
    pad_left: int
    content_size: Tuple[int, int]  # (H, W) of the resized content inside the canvas


def compute_resize(orig_size: Tuple[int, int], out_size: Tuple[int, int], mode: str = "letterbox") -> ResizeInfo:
    oh, ow = orig_size
    th, tw = out_size
    if mode == "stretch":
        return ResizeInfo(orig_size, out_size, th / oh, tw / ow, 0, 0, (th, tw))
    if mode != "letterbox":
        raise ValueError(f"Unknown resize mode '{mode}'")
    scale = min(th / oh, tw / ow)
    ch, cw = max(1, round(oh * scale)), max(1, round(ow * scale))
    return ResizeInfo(orig_size, out_size, scale, scale, (th - ch) // 2, (tw - cw) // 2, (ch, cw))


def _place(content: np.ndarray, info: ResizeInfo, fill) -> np.ndarray:
    th, tw = info.out_size
    shape = (th, tw) + content.shape[2:]
    canvas = np.empty(shape, dtype=content.dtype)
    canvas[...] = fill
    ch, cw = info.content_size
    canvas[info.pad_top : info.pad_top + ch, info.pad_left : info.pad_left + cw] = content
    return canvas


def resize_image(image: np.ndarray, out_size: Tuple[int, int], mode: str = "letterbox", pad_value: int = 255) -> Tuple[np.ndarray, ResizeInfo]:
    info = compute_resize(image.shape[:2], out_size, mode)
    ch, cw = info.content_size
    shrinking = ch < image.shape[0]
    resample = Image.BOX if shrinking else Image.BILINEAR  # BOX = area averaging, avoids aliasing
    content = np.asarray(Image.fromarray(image).resize((cw, ch), resample=resample))
    return _place(content, info, pad_value), info


def resize_labels(label_map: np.ndarray, out_size: Tuple[int, int], mode: str = "letterbox", pad_value: int = 0) -> Tuple[np.ndarray, ResizeInfo]:
    info = compute_resize(label_map.shape[:2], out_size, mode)
    ch, cw = info.content_size
    content = np.asarray(Image.fromarray(label_map).resize((cw, ch), resample=Image.NEAREST))
    return _place(content, info, pad_value), info


def undo_resize_labels(label_map: np.ndarray, info: ResizeInfo) -> np.ndarray:
    """Map a canvas-sized label map back to the original image size (nearest neighbour)."""
    ch, cw = info.content_size
    crop = label_map[info.pad_top : info.pad_top + ch, info.pad_left : info.pad_left + cw]
    oh, ow = info.orig_size
    return np.asarray(Image.fromarray(crop).resize((ow, oh), resample=Image.NEAREST))


# --------------------------------------------------------------------------------------
# Tensor conversion
# --------------------------------------------------------------------------------------


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """``H x W x 3`` uint8 RGB -> ``3 x H x W`` float32 in [-1, 1]."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected H x W x 3 uint8 RGB, got {image.shape} {image.dtype}")
    t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    return t / 127.5 - 1.0


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """``3 x H x W`` in [-1, 1] -> ``H x W x 3`` uint8 (values are clamped)."""
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected 3 x H x W tensor, got {tuple(tensor.shape)}")
    arr = ((tensor.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round()
    return arr.permute(1, 2, 0).numpy().astype(np.uint8)


def labels_to_tensor(label_map: np.ndarray) -> torch.Tensor:
    """``H x W`` uint8 -> ``H x W`` int64 (the dtype ``F.cross_entropy`` expects)."""
    if label_map.ndim != 2:
        raise ValueError(f"Expected H x W label map, got {label_map.shape}")
    return torch.from_numpy(np.ascontiguousarray(label_map)).long()
