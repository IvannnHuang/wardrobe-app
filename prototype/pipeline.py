"""Cut-out prototype (Phase 0, step A).

For each garment photo: pose -> person mask -> align to the body figure -> remove background,
skin/hair and everything that looks the same as in the body figure (base outfit / body) ->
transparent PNG + a comparison sheet that shows *why* each pixel was kept or removed.

Usage (from prototype/):
    .venv/Scripts/python pipeline.py               # all garments, all views found
    .venv/Scripts/python pipeline.py --mirror      # photos were taken in a mirror
    .venv/Scripts/python pipeline.py --garment hoodie-black-zip --view front

Inputs:  ../test-photos/body/<view>.jpg|png and ../test-photos/garments/<name>/<view>.jpg|png
Outputs: output/<name>/<view>_cutout.png, <view>_comparison.jpg, <view>_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy import ndimage
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
PHOTOS = ROOT.parent / "test-photos"
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"

VIEWS = ("front", "right", "back", "left")
PHOTO_EXTS = (".jpg", ".jpeg", ".png")

# selfie_multiclass_256x256 classes
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER = range(6)

# MediaPipe pose landmark ids used for alignment: shoulders, hips, knees, ankles.
ALIGN_POINTS = (11, 12, 23, 24, 25, 26, 27, 28)
# Wrist, pinky, index, thumb for each hand.
HANDS = ((15, 17, 19, 21), (16, 18, 20, 22))


@dataclass
class Settings:
    mirror: bool = False            # flip photos taken in a mirror; also masks the phone-holding hand
    work_long_side: int = 1600      # process at this size (px) for speed
    min_visibility: float = 0.6     # landmarks below this are not used for alignment
    change_threshold: float = 14.0  # colour difference (Delta E, LAB) that counts as "changed"
    person_threshold: float = 0.5
    edge_band_px: int = 16          # beyond this distance from visible base outfit, person pixels count as garment
    min_component_frac: float = 0.02  # drop blobs smaller than this fraction of the largest blob
    feather_px: float = 1.5
    fill_hidden: bool = True        # inpaint garment areas hidden behind a hand/phone
    inpaint_radius: int = 6


@dataclass
class Analysis:
    rgb: np.ndarray        # HxWx3 uint8
    landmarks: np.ndarray  # 33x3: x px, y px, visibility
    person: np.ndarray     # HxW float32 0..1
    classes: np.ndarray    # HxW uint8 segmenter class


@dataclass
class Report:
    garment: str
    view: str
    align_points: list[int] = field(default_factory=list)
    align_scale: float = 0.0
    align_rotation_deg: float = 0.0
    align_error_px: float = 0.0
    lighting_shift_lab: list[float] = field(default_factory=list)
    garment_area_frac: float = 0.0
    filled_frac: float = 0.0        # share of the garment that was invented by inpainting
    removed_px: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


# ---------------------------------------------------------------- loading + models

def find_photo(folder: Path, view: str) -> Path | None:
    for ext in PHOTO_EXTS:
        for candidate in (folder / f"{view}{ext}", folder / f"{view}{ext.upper()}"):
            if candidate.exists():
                return candidate
    return None


def load_photo(path: Path, s: Settings) -> np.ndarray:
    """Upright (EXIF), optionally un-mirrored, downscaled RGB uint8."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = s.work_long_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    rgb = np.asarray(img)
    return np.ascontiguousarray(rgb[:, ::-1]) if s.mirror else rgb


class Models:
    def __init__(self) -> None:
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_full.task")),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=True,
        ))
        self.segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "selfie_multiclass_256x256.tflite")),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        ))

    def analyse(self, rgb: np.ndarray) -> Analysis:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        h, w = rgb.shape[:2]

        pose = self.pose.detect(image)
        if not pose.pose_landmarks:
            raise RuntimeError("no person found by the pose model")
        landmarks = np.array([(p.x * w, p.y * h, p.visibility) for p in pose.pose_landmarks[0]], np.float32)
        person = np.squeeze(pose.segmentation_masks[0].numpy_view()).astype(np.float32)

        classes = np.squeeze(self.segmenter.segment(image).category_mask.numpy_view()).astype(np.uint8)
        if classes.shape != (h, w):
            classes = cv2.resize(classes, (w, h), interpolation=cv2.INTER_NEAREST)
        return Analysis(rgb, landmarks, person, classes)


# ---------------------------------------------------------------- steps

def align(garment: Analysis, body: Analysis, s: Settings, report: Report) -> tuple[Analysis, np.ndarray]:
    """Similarity transform (scale + rotation + shift) mapping garment landmarks onto the body figure."""
    ids = [i for i in ALIGN_POINTS
           if garment.landmarks[i, 2] >= s.min_visibility and body.landmarks[i, 2] >= s.min_visibility]
    if len(ids) < 2:
        raise RuntimeError("fewer than 2 landmarks visible in both photos; cannot align")
    src, dst = garment.landmarks[ids, :2], body.landmarks[ids, :2]
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    projected = src @ matrix[:, :2].T + matrix[:, 2]
    report.align_points = ids
    report.align_scale = float(np.hypot(*matrix[0, :2]))
    report.align_rotation_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    report.align_error_px = float(np.mean(np.linalg.norm(projected - dst, axis=1)))
    if len(ids) < 4:
        report.warnings.append(f"only {len(ids)} alignment points; alignment may be poor")
    if report.align_error_px > 0.01 * max(body.rgb.shape):
        report.warnings.append(f"pose differs between photos (mean landmark error {report.align_error_px:.1f}px)")

    h, w = body.rgb.shape[:2]
    warp = lambda img, interp: cv2.warpAffine(img, matrix, (w, h), flags=interp, borderValue=0)
    valid = warp(np.ones(garment.rgb.shape[:2], np.uint8), cv2.INTER_NEAREST).astype(bool)
    landmarks = garment.landmarks.copy()
    landmarks[:, :2] = garment.landmarks[:, :2] @ matrix[:, :2].T + matrix[:, 2]
    warped = Analysis(
        rgb=warp(garment.rgb, cv2.INTER_LINEAR),
        landmarks=landmarks,
        person=warp(garment.person, cv2.INTER_LINEAR),
        classes=warp(garment.classes, cv2.INTER_NEAREST),
    )
    return warped, valid


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """Real LAB (L 0..100), lightly blurred so JPEG noise and 1px misalignment don't count as change."""
    blurred = cv2.GaussianBlur(rgb, (5, 5), 0)
    return cv2.cvtColor(blurred.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


def match_lighting(lab_g: np.ndarray, lab_b: np.ndarray, background: np.ndarray, report: Report) -> np.ndarray:
    """Shift garment colours so the shared background (wall) matches the body figure's."""
    if background.sum() < 1000:
        report.warnings.append("too little shared background to match lighting")
        report.lighting_shift_lab = [0.0, 0.0, 0.0]
        return lab_g
    shift = np.median(lab_b[background], axis=0) - np.median(lab_g[background], axis=0)
    report.lighting_shift_lab = [round(float(v), 2) for v in shift]
    if abs(shift[0]) > 8:
        report.warnings.append(f"brightness changed between photos (L shift {shift[0]:.1f}); lock exposure")
    return lab_g + shift


def reference_colours(body: Analysis, lab_b: np.ndarray, reliable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every pixel: the colour of the nearest body-figure pixel where the base outfit is clearly visible,
    and the distance to it. Next to the torso that is the base top, next to the legs the base bottom; where
    the body figure has a hand or phone in the way, the surrounding base outfit is used instead."""
    if not reliable.any():
        h, w = reliable.shape
        return lab_b, np.full((h, w), np.inf, np.float32)
    dist, (iy, ix) = ndimage.distance_transform_edt(~reliable, return_indices=True)
    return lab_b[iy, ix], dist


def phone_hand_region(a: Analysis) -> tuple[np.ndarray, float]:
    """Mirror selfies: box around the raised (phone-holding) hand, extended upward over the phone."""
    h, w = a.rgb.shape[:2]
    hand = min(HANDS, key=lambda ids: a.landmarks[ids[0], 1])  # higher wrist = phone hand
    pts = a.landmarks[list(hand), :2]
    size = max(20.0, float(np.max(np.linalg.norm(pts - pts[0], axis=1))))
    x0, y0 = pts.min(axis=0) - (size * 0.8, size * 3.0)
    x1, y1 = pts.max(axis=0) + (size * 0.8, size * 0.6)
    region = np.zeros((h, w), bool)
    region[max(0, int(y0)):min(h, int(y1)), max(0, int(x0)):min(w, int(x1))] = True
    return region, size


def torso_region(a: Analysis, grow: float = 0.15) -> np.ndarray:
    """Quad through shoulders and hips, slightly enlarged. Skin inside it is a hand in front of the body."""
    pts = a.landmarks[[11, 12, 24, 23], :2]
    centre = pts.mean(axis=0)
    pts = centre + (pts - centre) * (1 + grow)
    mask = np.zeros(a.rgb.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def fill_hidden(rgb: np.ndarray, keep: np.ndarray, hidden: np.ndarray, s: Settings) -> np.ndarray:
    """Inpaint `hidden` using only garment pixels as the source (everything else is marked unknown,
    so white tee or skin next to the hole can't bleed in)."""
    ys, xs = np.nonzero(keep | hidden)
    h, w = keep.shape
    y0, y1 = max(0, ys.min() - 20), min(h, ys.max() + 21)
    x0, x1 = max(0, xs.min() - 20), min(w, xs.max() + 21)
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    # Shrink the source a little so mixed edge pixels (skin/phone fringe) don't bleed into the fill.
    source = cv2.erode(keep[y0:y1, x0:x1].astype(np.uint8), np.ones((7, 7), np.uint8))
    unknown = (1 - source) * 255
    filled = cv2.inpaint(crop, unknown, s.inpaint_radius, cv2.INPAINT_TELEA)
    out = rgb.copy()
    region = hidden[y0:y1, x0:x1]
    out[y0:y1, x0:x1][region] = filled[region]
    return out


def cleanup(mask: np.ndarray, s: Settings) -> np.ndarray:
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if count <= 1:
        return m.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= s.min_component_frac * areas.max()) + 1
    return np.isin(labels, keep)


def cut_out(garment: Analysis, valid: np.ndarray, body: Analysis, s: Settings, report: Report):
    """Returns (RGBA cut-out, reason map, colour change). Reason codes are listed in REASONS."""
    person_g = (garment.person > s.person_threshold) & valid
    person_b = body.person > s.person_threshold
    if s.mirror:
        box_g, hand_size = phone_hand_region(garment)
        blocked_b, _ = phone_hand_region(body)
    else:
        box_g, hand_size = np.zeros_like(person_g), 25.0
        blocked_b = np.zeros_like(person_b)
    blocked_g = box_g.copy()

    lab_b = to_lab(body.rgb)
    lab_g = match_lighting(to_lab(garment.rgb), lab_b, ~person_g & ~person_b & valid, report)

    # Compare each garment pixel with the base outfit at (or nearest to) the same place on the body figure.
    # Any body-figure pixel except hair/face is a valid "before" (base outfit or bare arm). Don't rely on the
    # segmenter's clothes/skin labels here: it mislabels dark clothes as skin. Hair is excluded so a dark
    # hood beside the head isn't mistaken for hair.
    reliable_b = person_b & ~np.isin(body.classes, (HAIR, FACE_SKIN)) & ~blocked_b
    reference, dist = reference_colours(body, lab_b, reliable_b)
    change = np.linalg.norm(lab_g - reference, axis=2)
    # Far from any visible base outfit (e.g. a hood beside the head, a wide sleeve) = garment sticking out.
    changed = (change > s.change_threshold) | (dist > s.edge_band_px)

    skin_hair = np.isin(garment.classes, (HAIR, BODY_SKIN, FACE_SKIN))

    raw = person_g & ~skin_hair & changed
    if blocked_g.any():
        # Inside the phone/hand box keep only pixels that look like the rest of the garment.
        garment_colour = np.median(lab_g[raw & ~blocked_g], axis=0)
        blocked_g &= np.linalg.norm(lab_g - garment_colour, axis=2) > s.change_threshold * 1.5
    raw &= ~blocked_g
    keep = cleanup(raw, s)

    # Hidden garment = something in front of it (hand on the torso, phone), as opposed to base outfit
    # showing through (e.g. tee in an open zip), which stays transparent. Closing merges hand + phone
    # and swallows small dark bits between them (the phone's camera bump).
    hidden = np.zeros_like(keep)
    rgb = garment.rgb
    if s.fill_hidden and keep.any():
        in_front = (blocked_g | ((garment.classes == BODY_SKIN) & torso_region(garment))) & changed
        k = max(15, int(hand_size * 0.6)) | 1
        closed = cv2.morphologyEx((person_g & in_front).astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
        # Never fill what matches the base top colour (tee in an open zip stays transparent).
        top_b = reliable_b & torso_region(body)
        base_top = np.median(lab_b[top_b], axis=0) if top_b.any() else lab_b.reshape(-1, 3).mean(axis=0)
        base_like = np.linalg.norm(lab_g - base_top, axis=2) < s.change_threshold
        hidden = closed & person_g & (changed | keep) & ~base_like
        # Mop up mixed edge pixels (finger outlines) left between the fill and the garment.
        solid = cv2.morphologyEx((keep | hidden).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        hidden |= solid.astype(bool) & person_g & ~keep & ~base_like & cv2.dilate(
            hidden.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        keep &= ~hidden
        rgb = fill_hidden(garment.rgb, keep, hidden, s)

    reason = np.zeros(person_b.shape, np.uint8)            # 0 background
    reason[person_g] = 4                                   # removed by cleanup (default inside person)
    reason[person_g & ~changed] = 2                        # same as base outfit on body figure
    reason[person_g & skin_hair] = 1                       # skin / hair
    reason[person_g & blocked_g] = 3                       # hand + phone (mirror mode)
    reason[keep] = 5                                       # kept as garment
    reason[hidden] = 6                                     # hidden, filled by inpainting

    garment_mask = keep | hidden
    report.garment_area_frac = round(float(garment_mask.mean()), 4)
    report.filled_frac = round(float(hidden.sum() / max(1, garment_mask.sum())), 4)
    report.removed_px = {REASONS[i][0]: int((reason == i).sum()) for i in range(1, 5)}
    if keep.sum() == 0:
        report.warnings.append("nothing kept: garment looks identical to the body figure")
    if report.filled_frac > 0.03:
        report.warnings.append(f"{report.filled_frac:.1%} of the garment is filled in (hidden behind hand/phone)")

    alpha = cv2.GaussianBlur(garment_mask.astype(np.float32), (0, 0), s.feather_px)
    alpha *= np.clip(np.maximum(garment.person, hidden) * 1.5, 0, 1)
    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    return rgba, reason, change


# reason code -> (name, BGR colour for the overlay)
REASONS = {
    0: ("background", (0, 0, 0)),
    1: ("skin_hair", (0, 140, 255)),       # orange
    2: ("same_as_body", (255, 120, 0)),    # blue
    3: ("hand_phone", (200, 0, 200)),      # purple
    4: ("cleanup", (0, 0, 255)),           # red
    5: ("kept", (0, 200, 0)),              # green
    6: ("filled", (255, 255, 0)),          # cyan
}


# ---------------------------------------------------------------- output

def checkerboard(h: int, w: int, size: int = 24) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    board = ((yy // size + xx // size) % 2).astype(np.uint8)
    return np.where(board[..., None] == 1, 205, 245).astype(np.uint8).repeat(3, axis=2)


def composite(rgba: np.ndarray, under_rgb: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3] * a + under_rgb * (1 - a)).astype(np.uint8)


def draw_landmarks(rgb: np.ndarray, landmarks: np.ndarray, color, ids=ALIGN_POINTS) -> np.ndarray:
    out = rgb.copy()
    r = max(3, rgb.shape[0] // 250)
    for i in ids:
        x, y, v = landmarks[i]
        if v > 0.3:
            cv2.circle(out, (int(x), int(y)), r, color, -1)
    return out


def comparison_sheet(body: Analysis, garment: Analysis, rgba, reason, change, report: Report) -> np.ndarray:
    h, w = body.rgb.shape[:2]
    overlay_bgr = np.zeros((h, w, 3), np.uint8)
    for code, (_, bgr) in REASONS.items():
        overlay_bgr[reason == code] = bgr
    reasons_rgb = cv2.addWeighted(garment.rgb, 0.45, overlay_bgr[..., ::-1], 0.55, 0)
    heat = cv2.applyColorMap(np.clip(change * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]

    panels = [
        ("body figure", draw_landmarks(body.rgb, body.landmarks, (0, 255, 0))),
        ("garment, aligned", draw_landmarks(draw_landmarks(garment.rgb, body.landmarks, (0, 255, 0)),
                                            garment.landmarks, (255, 0, 255))),
        ("colour change vs body", heat),
        ("why kept / removed", reasons_rgb),
        ("cut-out", composite(rgba, checkerboard(h, w))),
        ("cut-out on body figure", composite(rgba, body.rgb)),
    ]
    target_h = 900
    tiles = []
    for title, img in panels:
        img = cv2.resize(img, (round(w * target_h / h), target_h), interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img[..., ::-1])  # to BGR for OpenCV drawing
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    sheet = np.hstack(tiles)

    legend = np.full((40, sheet.shape[1], 3), 30, np.uint8)
    x = 10
    for code in (5, 6, 1, 2, 3, 4):
        name, bgr = REASONS[code]
        cv2.rectangle(legend, (x, 10), (x + 20, 30), bgr, -1)
        cv2.putText(legend, name, (x + 28, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        x += 40 + 12 * len(name)
    note = "green dots: body figure landmarks   magenta: garment landmarks after alignment"
    cv2.putText(legend, note, (x + 20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([sheet, legend])


# ---------------------------------------------------------------- driver

def process(models: Models, body: Analysis, garment_dir: Path, view: str, s: Settings) -> Report:
    start = time.time()
    report = Report(garment=garment_dir.name, view=view)
    out_dir = OUTPUT / garment_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    garment_raw = models.analyse(load_photo(find_photo(garment_dir, view), s))
    garment, valid = align(garment_raw, body, s, report)
    rgba, reason, change = cut_out(garment, valid, body, s, report)

    Image.fromarray(rgba, "RGBA").save(out_dir / f"{view}_cutout.png")
    sheet = comparison_sheet(body, garment, rgba, reason, change, report)
    cv2.imwrite(str(out_dir / f"{view}_comparison.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    report.seconds = round(time.time() - start, 2)
    (out_dir / f"{view}_report.json").write_text(json.dumps(asdict(report), indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mirror", action="store_true", help="photos were taken in a mirror")
    parser.add_argument("--garment", help="only this garment folder")
    parser.add_argument("--view", choices=VIEWS, help="only this view")
    args = parser.parse_args()
    s = Settings(mirror=args.mirror)

    models = Models()
    garment_dirs = sorted(p for p in (PHOTOS / "garments").iterdir() if p.is_dir())
    if args.garment:
        garment_dirs = [p for p in garment_dirs if p.name == args.garment]
    for view in [args.view] if args.view else VIEWS:
        body_path = find_photo(PHOTOS / "body", view)
        if body_path is None:
            continue
        body = models.analyse(load_photo(body_path, s))
        for garment_dir in garment_dirs:
            if find_photo(garment_dir, view) is None:
                continue
            try:
                r = process(models, body, garment_dir, view, s)
            except RuntimeError as e:
                print(f"FAIL {garment_dir.name}/{view}: {e}")
                continue
            print(f"OK   {r.garment}/{r.view}: garment {r.garment_area_frac:.1%} of frame, filled {r.filled_frac:.1%}, "
                  f"align error {r.align_error_px:.1f}px, {r.seconds}s")
            for w in r.warnings:
                print(f"     ! {w}")


if __name__ == "__main__":
    main()
