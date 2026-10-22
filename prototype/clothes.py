"""Reference implementation of the tablet cut-out (method C) using the exported ONNX model.

The Kotlin port (android/.../pipeline/cutout) mirrors this file. Per garment photo:
  1. clothes model (ONNX fp16) -> label map at photo size
  2. per piece: drop if it's the base outfit / already worn on the body figure, else remove base colours
  3. clean up; piece masks in the garment photo frame (+ the similarity transform to the body figure)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

import pipeline as P

MODEL = P.MODELS / "clothes_segformer_b2_fp16.onnx"
SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# label ids (ATR): 0 bg, 1 hat, 2 hair, 3 sunglasses, 4 upper, 5 skirt, 6 pants, 7 dress, 8 belt,
# 9/10 shoes, 11 face, 12/13 legs, 14/15 arms, 16 bag, 17 scarf
PIECES = {
    "top": ([4], "top"),
    "dress": ([7], "top"),
    "bottom": ([5, 6], "bottom"),
    "shoes": ([9, 10], None),
    "hat": ([1], None),
    "bag": ([16], None),
    "belt": ([8], None),
    "scarf": ([17], None),
}
BASE_IDS = {"top": [4, 7], "bottom": [5, 6]}
SKIN_IDS = [11, 14, 15]  # face, arms: colour reference for lighting (experiment_light.py)
L_WEIGHT = 0.5           # exposure changes mostly move lightness, so it counts half


@dataclass
class Knobs:
    base_delta_e: float = 16.0
    base_piece_frac: float = 0.6
    min_piece_frac: float = 0.002
    min_component_frac: float = 0.02


@dataclass
class BaseOutfit:
    """Colours learned from the body figure photo (same view)."""
    palettes: dict[str, np.ndarray]                       # "top"/"bottom" -> k x 3 LAB
    worn: dict[str, np.ndarray] = field(default_factory=dict)  # accessory -> median LAB
    skin: np.ndarray | None = None                        # median skin LAB on the body figure


@dataclass
class Piece:
    name: str
    mask: np.ndarray        # bool, garment photo frame
    region: np.ndarray      # bool, model label region before base removal
    base_frac: float        # share of region that matched the base colour


class ClothesModel:
    def __init__(self):
        self.session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])

    def labels(self, rgb: np.ndarray) -> np.ndarray:
        x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        logits = self.session.run(None, {"pixel_values": np.ascontiguousarray(x)})[0][0]  # 18 x 128 x 128
        h, w = rgb.shape[:2]
        up = np.stack([cv2.resize(c.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) for c in logits])
        return up.argmax(axis=0).astype(np.uint8)


def lab_of(rgb: np.ndarray) -> np.ndarray:
    return P.to_lab(rgb)


def learn_base(body_rgb: np.ndarray, body_labels: np.ndarray) -> BaseOutfit:
    lab = lab_of(body_rgb)
    palettes = {}
    for key, ids in BASE_IDS.items():
        px = lab[np.isin(body_labels, ids)]
        if len(px) < 50:
            palettes[key] = np.zeros((0, 3), np.float32)
            continue
        px = px[np.random.default_rng(0).choice(len(px), min(len(px), 20000), replace=False)].astype(np.float32)
        k = min(3, len(px))
        _, lbl, centers = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3,
                                     cv2.KMEANS_PP_CENTERS)
        share = np.bincount(lbl.ravel(), minlength=k) / len(lbl)
        palettes[key] = centers[share >= 0.1]
    worn = {}
    for name, (ids, base_key) in PIECES.items():
        m = np.isin(body_labels, ids)
        if base_key is None and m.mean() >= 0.002:
            worn[name] = np.median(lab[m], axis=0)
    skin = np.isin(body_labels, SKIN_IDS)
    return BaseOutfit(palettes, worn, np.median(lab[skin], axis=0) if skin.sum() > 500 else None)


def weighted_de(lab: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """Distance to the nearest of `colours` with lightness weighted L_WEIGHT."""
    w = np.array([L_WEIGHT, 1, 1], np.float32)
    return np.min(np.linalg.norm((lab[..., None, :] - colours.reshape(-1, 3)[None, None]) * w, axis=3), axis=2)


def clean(mask: np.ndarray, min_frac: float) -> np.ndarray:
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return np.isin(labels, np.flatnonzero(areas >= min_frac * areas.max()) + 1)


def extract_pieces(rgb: np.ndarray, labels: np.ndarray, base: BaseOutfit, k: Knobs = Knobs()) -> tuple[list[Piece], list[str]]:
    lab = lab_of(rgb)
    # Lighting: shift colours so my skin matches the body figure's skin.
    skin = np.isin(labels, SKIN_IDS)
    if base.skin is not None and skin.sum() > 500:
        lab = lab + (base.skin - np.median(lab[skin], axis=0))
    pieces, notes = [], []
    for name, (ids, base_key) in PIECES.items():
        region = np.isin(labels, ids)
        if region.mean() < k.min_piece_frac:
            continue
        if name in base.worn and weighted_de(np.median(lab[region], axis=0)[None, None], base.worn[name])[0, 0] < k.base_delta_e:
            notes.append(f"{name}: already worn on body figure")
            continue
        keep, frac = region, 0.0
        palette = base.palettes.get(base_key) if base_key else None
        if palette is not None and len(palette):
            d = weighted_de(lab, palette)
            is_base = region & (d < k.base_delta_e)
            frac = float(is_base.sum() / max(1, region.sum()))
            if frac >= k.base_piece_frac:
                notes.append(f"{name}: base outfit ({frac:.0%} base colour)")
                continue
            keep = region & ~is_base
        keep = clean(keep, k.min_component_frac)
        if keep.mean() >= k.min_piece_frac:
            pieces.append(Piece(name, keep, region, frac))
    return pieces, notes
