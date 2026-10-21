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

# Skeleton used to repose garments: torso quad + limb bones (landmark id pairs), grouped per body part.
TORSO = (11, 12, 24, 23)
LIMBS = {
    "left_arm": ((11, 13), (13, 15), (15, 19)),
    "right_arm": ((12, 14), (14, 16), (16, 20)),
    "left_leg": ((23, 25), (25, 27)),
    "right_leg": ((24, 26), (26, 28)),
}
GROUPS = ("torso", *LIMBS)
LIMB_POINTS = (13, 14, 15, 16, 25, 26, 27, 28)  # elbows, wrists, knees, ankles


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
    repose: bool = True             # bend the garment's limbs onto the body figure's limbs before cutting out
    arms_down: bool = False         # also output garment + body figure in a standard arms-down pose


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
    limb_error_before_px: float = 0.0   # elbows/wrists/knees/ankles vs body figure, after global align
    limb_error_after_px: float = 0.0    # ... and after reposing
    lighting_shift_lab: list[float] = field(default_factory=list)
    garment_area_frac: float = 0.0
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


# ---------------------------------------------------------------- reposing

def _bones() -> list[tuple[int, tuple[int, int]]]:
    """(group index, (landmark a, landmark b)) for every limb bone."""
    return [(GROUPS.index(g), b) for g, bones in LIMBS.items() for b in bones]


def _distance_maps(lm: np.ndarray, shape: tuple[int, int]) -> tuple[list[np.ndarray], np.ndarray]:
    """Distance (px) from every pixel to the torso and to each bone, plus per-group distance (GxHxW)."""
    h, w = shape
    torso = np.full((h, w), 255, np.uint8)
    cv2.fillPoly(torso, [lm[list(TORSO), :2].astype(np.int32)], 0)
    maps = [cv2.distanceTransform(torso, cv2.DIST_L2, 5)]
    for _, (a, b) in _bones():
        line = np.full((h, w), 255, np.uint8)
        cv2.line(line, tuple(int(v) for v in lm[a, :2]), tuple(int(v) for v in lm[b, :2]), 0, 3)
        maps.append(cv2.distanceTransform(line, cv2.DIST_L2, 5))
    groups = [0] + [g for g, _ in _bones()]
    per_group = np.stack([np.min([m for m, g in zip(maps, groups) if g == gi], axis=0)
                          for gi in range(len(GROUPS))])
    return maps, per_group


def _bone_inverse(src: np.ndarray, dst: np.ndarray, a: int, b: int) -> np.ndarray:
    """2x3 similarity mapping target-pose pixels around bone a->b back to the source photo."""
    d_src, d_dst = src[b, :2] - src[a, :2], dst[b, :2] - dst[a, :2]
    rot = np.eye(2)
    if np.hypot(*d_dst) >= 1:
        z = complex(*d_src) / complex(*d_dst)
        rot = np.array([[z.real, -z.imag], [z.imag, z.real]])
    return np.hstack([rot, (src[a, :2] - rot @ dst[a, :2])[:, None]])


def _limit_turns(src: np.ndarray, dst: np.ndarray, max_turn_deg: float) -> tuple[np.ndarray, list[str]]:
    """Limbs whose bones would have to rotate more than `max_turn_deg` are held where they are (moved
    with the torso only) for the smooth bend: big rotations smear. `repose` then moves them as rigid pieces."""
    torso = cv2.estimateAffinePartial2D(src[list(TORSO), :2], dst[list(TORSO), :2])[0]
    dst = dst.copy()
    skipped = []
    for name, bones in LIMBS.items():
        turns = []
        for a, b in bones:
            d_src, d_dst = src[b, :2] - src[a, :2], dst[b, :2] - dst[a, :2]
            turn = np.degrees(np.arctan2(d_dst[1], d_dst[0]) - np.arctan2(d_src[1], d_src[0]))
            turns.append(abs((turn + 180) % 360 - 180))
        if max(turns) > max_turn_deg:
            ids = sorted({i for bone in bones for i in bone} - set(TORSO))
            ids += [i for hand in HANDS if hand[0] in ids for i in hand[1:]]
            dst[ids, :2] = src[ids, :2] @ torso[:, :2].T + torso[:, 2]
            skipped.append(name)
    return dst, skipped


def _limb_pieces(lm: np.ndarray, person: np.ndarray, limb: str, radius_frac: float = 0.22) -> list[np.ndarray]:
    """Person pixels belonging to each bone of `limb` (within a radius of the bone line). The radius is
    based on the limb's length, so a sleeve held across the chest is cut out with the arm, not the torso.
    Pieces overlap a little at the joints so straightening an elbow doesn't open a gap."""
    bones = LIMBS[limb]
    length = np.mean([np.linalg.norm(lm[b, :2] - lm[a, :2]) for a, b in bones[:2]])
    radius = max(12, int(radius_frac * length))
    pieces = []
    for a, b in bones:
        piece = np.zeros(person.shape, np.uint8)
        cv2.line(piece, tuple(int(v) for v in lm[a, :2]), tuple(int(v) for v in lm[b, :2]), 1, 2 * radius)
        cv2.circle(piece, tuple(int(v) for v in lm[b, :2]), radius, 1, -1)
        pieces.append(piece.astype(bool) & (person > 0.5))
    if limb.endswith("arm"):  # the hand piece also takes whatever the hand is holding
        hand = next(h for h in HANDS if h[0] == bones[-1][0])
        pts = lm[list(hand), :2]
        centre = pts.mean(axis=0)
        reach = max(radius, int(1.6 * np.max(np.linalg.norm(pts - centre, axis=1))))
        held = np.zeros(person.shape, np.uint8)
        cv2.circle(held, tuple(int(v) for v in centre), reach, 1, -1)
        pieces[-1] |= held.astype(bool) & (person > 0.5)
    return pieces


def repose(images: list[np.ndarray], src: np.ndarray, dst: np.ndarray, person: np.ndarray,
           interps: list[int], max_turn_deg: float = 40.0, reach_px: float = 80.0):
    """Move images from pose `src` to pose `dst` (33x3 landmarks in the same frame).

    Small changes: torso, upper arm, forearm+hand, thigh and shin each get their own similarity transform
    and every pixel follows a distance-weighted blend of nearby bones (linear blend skinning). Pixels
    further than about `reach_px` from the skeleton stay put, so the background doesn't bend.
    Big changes (a bone turning more than `max_turn_deg`): blending would smear, so that limb is cut into
    rigid pieces (upper / lower / hand) that are rotated into place and laid on top.
    Either way, what a moved limb used to cover is left transparent (invalid), never duplicated or filled.
    Returns (warped images, valid mask, final target landmarks, limbs moved as rigid pieces)."""
    h, w = person.shape
    full_dst = dst
    dst, rigid = _limit_turns(src, dst, max_turn_deg)
    dst_maps, dst_groups = _distance_maps(dst, (h, w))
    _, src_groups = _distance_maps(src, (h, w))

    transforms = [cv2.estimateAffinePartial2D(dst[list(TORSO), :2], src[list(TORSO), :2])[0]]
    transforms += [_bone_inverse(src, dst, a, b) for _, (a, b) in _bones()]
    transforms.append(np.hstack([np.eye(2), np.zeros((2, 1))]))      # identity for far-away pixels
    dst_maps.append(np.full((h, w), reach_px, np.float32))

    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    total = np.zeros((h, w), np.float32)
    for m, d in zip(transforms, dst_maps):
        wgt = 1.0 / (d + 4.0) ** 4
        map_x += wgt * (m[0, 0] * xx + m[0, 1] * yy + m[0, 2])
        map_y += wgt * (m[1, 0] * xx + m[1, 1] * yy + m[1, 2])
        total += wgt
    map_x /= total
    map_y /= total

    warped = [cv2.remap(img, map_x, map_y, interp, borderValue=0) for img, interp in zip(images, interps)]

    # Ownership check: the source pixel must belong to the body part the target pixel belongs to.
    target_group = np.argmin(dst_groups, axis=0)
    src_owner = np.argmin(src_groups, axis=0).astype(np.uint8)
    owner = cv2.remap(src_owner, map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
    own_d = cv2.remap(np.min(src_groups, axis=0), map_x, map_y, cv2.INTER_LINEAR)
    # distance, in the source, from the sampled pixel to the part it should belong to
    tgt_d = np.zeros((h, w), np.float32)
    for gi in range(len(GROUPS)):
        sel = target_group == gi
        if sel.any():
            tgt_d[sel] = cv2.remap(src_groups[gi], map_x, map_y, cv2.INTER_LINEAR)[sel]
    src_person = cv2.remap(person, map_x, map_y, cv2.INTER_LINEAR) > 0.5
    ghost = (owner != target_group) & (own_d * 2 < tgt_d) & src_person
    inside = (map_x >= 0) & (map_x < w - 1) & (map_y >= 0) & (map_y < h - 1)
    valid = inside & ~ghost

    for limb in rigid:
        pieces = _limb_pieces(src, person, limb)
        # Remove the limb from where it was ...
        old = cv2.remap(np.any(pieces, axis=0).astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST) > 0
        valid &= ~old
        # ... and lay each rigid piece down at its new place, upper arm first, hand last (on top).
        for (a, b), piece in zip(LIMBS[limb], pieces):
            m = _bone_inverse(src, full_dst, a, b)
            sx = (m[0, 0] * xx + m[0, 1] * yy + m[0, 2]).astype(np.float32)
            sy = (m[1, 0] * xx + m[1, 1] * yy + m[1, 2]).astype(np.float32)
            laid = cv2.remap(piece.astype(np.uint8), sx, sy, cv2.INTER_NEAREST, borderValue=0) > 0
            for k, (img, interp) in enumerate(zip(images, interps)):
                moved = cv2.remap(img, sx, sy, interp, borderValue=0)
                warped[k][laid] = moved[laid]
            valid |= laid
        ids = sorted({i for bone in LIMBS[limb] for i in bone} - set(TORSO))
        ids += [i for hand in HANDS if hand[0] in ids for i in hand[1:]]
        dst = dst.copy()
        dst[ids, :2] = full_dst[ids, :2]
    return warped, valid, dst, rigid


def repose_analysis(a: Analysis, valid: np.ndarray, dst: np.ndarray, report: Report) -> tuple[Analysis, np.ndarray]:
    (rgb, person, classes, v), ok, final, skipped = repose(
        [a.rgb, a.person, a.classes, valid.astype(np.uint8)], a.landmarks, dst, a.person,
        [cv2.INTER_LINEAR, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_NEAREST])
    for limb in skipped:
        report.warnings.append(f"repose: {limb} turned a lot, moved as rigid pieces (expect a crease at the joint)")
    landmarks = final.copy()
    landmarks[:, 2] = a.landmarks[:, 2]
    return Analysis(rgb, landmarks, np.where(ok, person, 0).astype(np.float32), classes), v.astype(bool) & ok


def arms_down_pose(lm: np.ndarray, upper_deg: float = 8.0, fore_deg: float = 12.0) -> np.ndarray:
    """Same body with both arms hanging down, slightly out from the body (angles from vertical)."""
    out = lm.copy()
    for shoulder, elbow, wrist, hand, other in ((11, 13, 15, (17, 19, 21), 12), (12, 14, 16, (18, 20, 22), 11)):
        outward = float(np.sign(lm[shoulder, 0] - lm[other, 0])) or 1.0
        upper = np.linalg.norm(lm[elbow, :2] - lm[shoulder, :2])
        fore = np.linalg.norm(lm[wrist, :2] - lm[elbow, :2])
        u, f = np.radians(upper_deg), np.radians(fore_deg)
        out[elbow, :2] = lm[shoulder, :2] + upper * np.array([outward * np.sin(u), np.cos(u)])
        out[wrist, :2] = out[elbow, :2] + fore * np.array([outward * np.sin(f), np.cos(f)])
        old_dir = lm[wrist, :2] - lm[elbow, :2]
        new_dir = out[wrist, :2] - out[elbow, :2]
        turn = np.arctan2(new_dir[1], new_dir[0]) - np.arctan2(old_dir[1], old_dir[0])
        rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        for i in hand:
            out[i, :2] = out[wrist, :2] + rot @ (lm[i, :2] - lm[wrist, :2])
    return out


def limb_error(a: np.ndarray, b: np.ndarray) -> float:
    ok = (a[list(LIMB_POINTS), 2] > 0.5) & (b[list(LIMB_POINTS), 2] > 0.5)
    pts = np.array(LIMB_POINTS)[ok]
    return float(np.mean(np.linalg.norm(a[pts, :2] - b[pts, :2], axis=1))) if len(pts) else 0.0


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


def phone_hand_region(a: Analysis) -> np.ndarray:
    """Mirror selfies: box around the raised (phone-holding) hand, extended upward over the phone."""
    h, w = a.rgb.shape[:2]
    hand = min(HANDS, key=lambda ids: a.landmarks[ids[0], 1])  # higher wrist = phone hand
    pts = a.landmarks[list(hand), :2]
    size = max(20.0, float(np.max(np.linalg.norm(pts - pts[0], axis=1))))
    x0, y0 = pts.min(axis=0) - (size * 0.8, size * 3.0)
    x1, y1 = pts.max(axis=0) + (size * 0.8, size * 0.6)
    region = np.zeros((h, w), bool)
    region[max(0, int(y0)):min(h, int(y1)), max(0, int(x0)):min(w, int(x1))] = True
    return region


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
    blocked_g = phone_hand_region(garment) if s.mirror else np.zeros_like(person_g)
    blocked_b = phone_hand_region(body) if s.mirror else np.zeros_like(person_b)

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

    reason = np.zeros(person_b.shape, np.uint8)            # 0 background
    reason[person_g] = 4                                   # removed by cleanup (default inside person)
    reason[person_g & ~changed] = 2                        # same as base outfit on body figure
    reason[person_g & skin_hair] = 1                       # skin / hair
    reason[person_g & blocked_g] = 3                       # hand + phone (mirror mode)
    reason[keep] = 5                                       # kept as garment

    report.garment_area_frac = round(float(keep.mean()), 4)
    report.removed_px = {REASONS[i][0]: int((reason == i).sum()) for i in range(1, 5)}
    if keep.sum() == 0:
        report.warnings.append("nothing kept: garment looks identical to the body figure")

    alpha = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), s.feather_px) * np.clip(garment.person * 1.5, 0, 1)
    rgba = np.dstack([garment.rgb, (alpha * 255).astype(np.uint8)])
    return rgba, reason, change


# reason code -> (name, BGR colour for the overlay)
REASONS = {
    0: ("background", (0, 0, 0)),
    1: ("skin_hair", (0, 140, 255)),       # orange
    2: ("same_as_body", (255, 120, 0)),    # blue
    3: ("hand_phone", (200, 0, 200)),      # purple
    4: ("cleanup", (0, 0, 255)),           # red
    5: ("kept", (0, 200, 0)),              # green
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


def comparison_sheet(body: Analysis, garment: Analysis, rgba, reason, change, report: Report,
                     before: np.ndarray | None = None, extra: list | None = None) -> np.ndarray:
    h, w = body.rgb.shape[:2]
    overlay_bgr = np.zeros((h, w, 3), np.uint8)
    for code, (_, bgr) in REASONS.items():
        overlay_bgr[reason == code] = bgr
    reasons_rgb = cv2.addWeighted(garment.rgb, 0.45, overlay_bgr[..., ::-1], 0.55, 0)
    heat = cv2.applyColorMap(np.clip(change * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]

    panels = [
        ("body figure", draw_landmarks(body.rgb, body.landmarks, (0, 255, 0))),
        *([("garment, before repose", draw_landmarks(before, body.landmarks, (0, 255, 0), LIMB_POINTS))]
          if before is not None else []),
        ("garment, aligned", draw_landmarks(draw_landmarks(garment.rgb, body.landmarks, (0, 255, 0)),
                                            garment.landmarks, (255, 0, 255))),
        ("colour change vs body", heat),
        ("why kept / removed", reasons_rgb),
        ("cut-out", composite(rgba, checkerboard(h, w))),
        ("cut-out on body figure", composite(rgba, body.rgb)),
        *(extra or []),
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
    for code in (5, 1, 2, 3, 4):
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
    report.limb_error_before_px = round(limb_error(garment.landmarks, body.landmarks), 1)
    before = None
    if s.repose:
        before = garment.rgb
        garment, valid = repose_analysis(garment, valid, body.landmarks, report)
    report.limb_error_after_px = round(limb_error(garment.landmarks, body.landmarks), 1)
    rgba, reason, change = cut_out(garment, valid, body, s, report)
    Image.fromarray(rgba, "RGBA").save(out_dir / f"{view}_cutout.png")

    extra = []
    if s.arms_down:
        (body_rgb,), body_ok, target, rigid = repose([body.rgb], body.landmarks, arms_down_pose(body.landmarks),
                                                     body.person, [cv2.INTER_LINEAR])
        body_rgb = np.where(body_ok[..., None], body_rgb, checkerboard(*body_ok.shape))
        for limb in rigid:
            report.warnings.append(f"arms-down: {limb} turned a lot, moved as rigid pieces")
        (cut_rgb, cut_alpha), ok, _, _ = repose([rgba[..., :3], rgba[..., 3]], body.landmarks, target,
                                                body.person, [cv2.INTER_LINEAR, cv2.INTER_LINEAR])
        rgba_down = np.dstack([cut_rgb, np.where(ok, cut_alpha, 0).astype(np.uint8)])
        Image.fromarray(rgba_down, "RGBA").save(out_dir / f"{view}_cutout_arms_down.png")
        extra = [("body figure, arms down", draw_landmarks(body_rgb, target, (0, 255, 0), range(11, 23))),
                 ("cut-out, arms down", composite(rgba_down, body_rgb))]

    sheet = comparison_sheet(body, garment, rgba, reason, change, report, before, extra)
    cv2.imwrite(str(out_dir / f"{view}_comparison.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    report.seconds = round(time.time() - start, 2)
    (out_dir / f"{view}_report.json").write_text(json.dumps(asdict(report), indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mirror", action="store_true", help="photos were taken in a mirror")
    parser.add_argument("--no-repose", action="store_true", help="don't bend garment limbs onto the body figure")
    parser.add_argument("--arms-down", action="store_true", help="also output a standard arms-down pose")
    parser.add_argument("--garment", help="only this garment folder")
    parser.add_argument("--view", choices=VIEWS, help="only this view")
    args = parser.parse_args()
    s = Settings(mirror=args.mirror, repose=not args.no_repose, arms_down=args.arms_down)

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
            print(f"OK   {r.garment}/{r.view}: garment {r.garment_area_frac:.1%} of frame, "
                  f"align error {r.align_error_px:.1f}px, limbs {r.limb_error_before_px}px -> "
                  f"{r.limb_error_after_px}px, {r.seconds}s")
            for w in r.warnings:
                print(f"     ! {w}")


if __name__ == "__main__":
    main()
