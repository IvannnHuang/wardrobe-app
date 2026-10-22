"""Cut-out method comparison (no ground truth: judge the sheets by eye, numbers are hints).

Methods, per garment photo:
  A  comparison with the body figure (pipeline.py, current tablet method)
  B  clothes-segmentation model only: every clothing pixel it finds
  C  model + base-colour removal: base top colour removed from tops, base bottom colour from
     bottoms; a piece that is mostly base colour is dropped (it's the base outfit itself)
  D  C + guided-filter edge refinement
  C is run with three model sizes (SegFormer B0 / B2 / B3 trained on clothes) to compare.

Usage (from prototype/):  .venv/Scripts/python experiment.py
Outputs: output/experiment/<set>/<garment>/<view>_methods.jpg, pieces PNGs, summary.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

import pipeline as P

OUT = P.OUTPUT / "experiment"

# Photo sets: folder, and whether the photos are mirror selfies.
SETS = {
    "mirror-1": (P.PHOTOS, True),
    "tablet-1": (P.PHOTOS / "tablet-1", False),
    "tablet-2": (P.PHOTOS / "tablet-2", False),
}

MODELS = {
    "b0": "mattmdjaga/segformer_b0_clothes",
    "b2": "mattmdjaga/segformer_b2_clothes",
    "b3": "sayeed99/segformer_b3_clothes",
}

# Pinned so the exported tablet model is reproducible.
REVISIONS = {"b2": "584abc1e1d260e23c0fc627c5217a09b2b461046"}

# Clothes-model labels grouped into garment pieces. Which base colour each piece is checked against.
PIECES = {
    "hat": ([1], None),
    "top": ([4], "top"),
    "dress": ([7], "top"),
    "bottom": ([5, 6], "bottom"),
    "belt": ([8], None),
    "shoes": ([9, 10], None),
    "bag": ([16], None),
    "scarf": ([17], None),
}
CLOTHING = sorted({c for ids, _ in PIECES.values() for c in ids})
BASE_TOP, BASE_BOTTOM = [4, 7], [5, 6]


@dataclass
class Knobs:
    base_delta_e: float = 16.0     # pixels this close (LAB) to the base outfit colour count as base
    base_piece_frac: float = 0.6   # a piece that is >= this share base colour is the base garment itself
    min_piece_frac: float = 0.002  # ignore pieces smaller than this share of the image
    guided_radius: int = 8
    guided_eps: float = 1e-3


@dataclass
class Result:
    method: str
    mask: np.ndarray                      # HxW float 0..1 (all kept pieces)
    pieces: dict[str, np.ndarray] = field(default_factory=dict)
    seconds: float = 0.0
    note: str = ""


# ---------------------------------------------------------------- clothes model

class Parser:
    def __init__(self, name: str):
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODELS[name], revision=REVISIONS.get(name)).eval()
        self.params_m = sum(p.numel() for p in self.model.parameters()) / 1e6

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.time()
        # SegFormer preprocessing: 512x512, ImageNet mean/std (same as the tablet would do).
        x = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        logits = self.model(pixel_values=torch.from_numpy(x.transpose(2, 0, 1)[None])).logits
        up = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        return up.argmax(dim=1)[0].numpy().astype(np.uint8), time.time() - start


def base_colours(body_rgb: np.ndarray, body_labels: np.ndarray) -> dict[str, np.ndarray]:
    """Main LAB colours (k-means, up to 3) of the base top and base bottom on the body figure."""
    lab = P.to_lab(body_rgb)
    out = {}
    for key, ids in (("top", BASE_TOP), ("bottom", BASE_BOTTOM)):
        px = lab[np.isin(body_labels, ids)]
        if len(px) < 50:
            out[key] = np.zeros((0, 3), np.float32)
            continue
        px = px[np.random.default_rng(0).choice(len(px), min(len(px), 20000), replace=False)].astype(np.float32)
        k = min(3, len(px))
        _, labels, centers = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5),
                                        3, cv2.KMEANS_PP_CENTERS)
        # keep clusters that hold at least 10% of the pixels (drops prints/shadows)
        share = np.bincount(labels.ravel(), minlength=k) / len(labels)
        out[key] = centers[share >= 0.1]
    # Pieces without their own base colour (shoes, hat, bag...) that are already on the body figure:
    # remember their median colour, so wearing the same pair again doesn't count as a new garment.
    for name, (ids, base_key) in PIECES.items():
        m = np.isin(body_labels, ids)
        if base_key is None and m.mean() >= 0.002:
            out["worn:" + name] = np.median(lab[m], axis=0)
    return out


def clean(mask: np.ndarray, min_frac: float = 0.02) -> np.ndarray:
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return np.isin(labels, np.flatnonzero(areas >= min_frac * areas.max()) + 1)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter (grey guide), snaps a soft mask to the photo's edges."""
    I = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    p = src.astype(np.float32)
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mean_I, mean_p = box(I), box(p)
    a = (box(I * p) - mean_I * mean_p) / (box(I * I) - mean_I ** 2 + eps)
    b = mean_p - a * mean_I
    return np.clip(box(a) * I + box(b), 0, 1)


def method_b(labels: np.ndarray, k: Knobs) -> Result:
    pieces = {}
    for name, (ids, _) in PIECES.items():
        m = clean(np.isin(labels, ids))
        if m.mean() >= k.min_piece_frac:
            pieces[name] = m
    union = np.any(list(pieces.values()), axis=0) if pieces else np.zeros(labels.shape, bool)
    return Result("B", union.astype(np.float32), pieces, note="+".join(pieces))


def method_c(rgb: np.ndarray, labels: np.ndarray, base: dict[str, np.ndarray], k: Knobs, tag: str) -> Result:
    lab = P.to_lab(rgb)
    pieces, notes = {}, []
    for name, (ids, base_key) in PIECES.items():
        region = np.isin(labels, ids)
        if region.mean() < k.min_piece_frac:
            continue
        keep = region
        worn = base.get("worn:" + name)
        if worn is not None and np.linalg.norm(np.median(lab[region], axis=0) - worn) < k.base_delta_e:
            notes.append(f"{name}=on body figure")
            continue
        palette = base.get(base_key) if base_key else None
        if palette is not None and len(palette):
            d = np.min(np.linalg.norm(lab[..., None, :] - palette[None, None], axis=3), axis=2)
            is_base = region & (d < k.base_delta_e)
            frac = is_base.sum() / max(1, region.sum())
            if frac >= k.base_piece_frac:
                notes.append(f"{name}=base({frac:.0%})")
                continue
            keep = region & ~is_base
        keep = clean(keep)
        if keep.mean() >= k.min_piece_frac:
            pieces[name] = keep
            notes.append(name)
    union = np.any(list(pieces.values()), axis=0) if pieces else np.zeros(labels.shape, bool)
    return Result(f"C-{tag}", union.astype(np.float32), pieces, note=" ".join(notes))


def method_d(rgb: np.ndarray, c: Result, k: Knobs) -> Result:
    pieces = {n: guided_filter(rgb, m.astype(np.float32), k.guided_radius, k.guided_eps) for n, m in c.pieces.items()}
    union = np.max(list(pieces.values()), axis=0) if pieces else np.zeros(c.mask.shape, np.float32)
    return Result("D-" + c.method.split("-")[1], union, pieces, note=c.note)


# ---------------------------------------------------------------- sheets

def tile(rgb: np.ndarray, mask: np.ndarray | None, title: str, sub: str, h: int = 640) -> np.ndarray:
    if mask is None:
        img = rgb
    else:
        H, W = mask.shape
        yy, xx = np.mgrid[:H, :W]
        board = np.where(((yy // 24 + xx // 24) % 2)[..., None] == 1, 200, 240)
        a = mask[..., None]
        img = (rgb * a + board * (1 - a)).astype(np.uint8)
    img = cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)
    img = np.ascontiguousarray(img[..., ::-1])
    cv2.rectangle(img, (0, 0), (img.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sub[:48], (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main() -> None:
    k = Knobs()
    print("loading models…")
    mp_models = P.Models()
    parsers = {name: Parser(name) for name in MODELS}
    for name, p in parsers.items():
        print(f"  {name}: {p.params_m:.1f} M parameters")
    summary = []

    for set_name, (folder, mirror) in SETS.items():
        s = P.Settings(mirror=mirror)
        for garment_dir in sorted(p for p in (folder / "garments").iterdir() if p.is_dir()):
            for view in P.VIEWS:
                g_path, b_path = P.find_photo(garment_dir, view), P.find_photo(folder / "body", view)
                if g_path is None or b_path is None:
                    continue
                out_dir = OUT / set_name / garment_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                body = mp_models.analyse(P.load_photo(b_path, s))
                garment_raw = mp_models.analyse(P.load_photo(g_path, s))
                rgb = garment_raw.rgb
                results: list[Result] = []

                # A: comparison with the body figure (result is in the body figure's frame;
                # warp it back into the garment photo's frame so all tiles line up).
                t = time.time()
                report = P.Report(garment_dir.name, view)
                try:
                    aligned, valid = P.align(garment_raw, body, s, report)
                    rgba, _, _ = P.cut_out(aligned, valid, body, s, report)
                    src = garment_raw.landmarks[report.align_points, :2]
                    dst = body.landmarks[report.align_points, :2]
                    inv, _ = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)
                    a_mask = cv2.warpAffine(rgba[..., 3].astype(np.float32) / 255, inv, (rgb.shape[1], rgb.shape[0]))
                    results.append(Result("A", a_mask, seconds=time.time() - t, note=f"align err {report.align_error_px:.0f}px"))
                except RuntimeError as e:
                    results.append(Result("A", np.zeros(rgb.shape[:2], np.float32), note=f"failed: {e}"))

                # Clothes model on garment and body figure, each size.
                timings = {}
                for name, parser in parsers.items():
                    g_labels, sec = parser(rgb)
                    b_labels, _ = parser(body.rgb)
                    timings[name] = round(sec, 2)
                    base = base_colours(body.rgb, b_labels)
                    if name == "b2":
                        results.append(method_b(g_labels, k))
                    c = method_c(rgb, g_labels, base, k, name)
                    c.seconds = sec
                    results.append(c)
                    if name == "b2":
                        d = method_d(rgb, c, k)
                        results.append(d)
                        for piece, m in d.pieces.items():
                            Image.fromarray(np.dstack([rgb, (m * 255).astype(np.uint8)]), "RGBA").save(
                                out_dir / f"{view}_D-b2_{piece}.png")

                order = ["A", "B", "C-b0", "C-b2", "C-b3", "D-b2"]
                by = {r.method: r for r in results}
                tiles = [tile(rgb, None, f"{set_name} {view}", garment_dir.name)]
                tiles += [tile(rgb, by[m].mask, m, by[m].note) for m in order if m in by]
                sheet = np.hstack(tiles)
                cv2.imwrite(str(out_dir / f"{view}_methods.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
                summary.append({
                    "set": set_name, "garment": garment_dir.name, "view": view,
                    "parser_seconds_cpu": timings,
                    "methods": {r.method: {"area": round(float((r.mask > 0.5).mean()), 4), "note": r.note} for r in results},
                })
                print(f"{set_name}/{garment_dir.name}/{view}: " + " | ".join(f"{r.method}: {r.note}" for r in results))

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("parameters (M):", {n: round(p.params_m, 1) for n, p in parsers.items()})


if __name__ == "__main__":
    main()
