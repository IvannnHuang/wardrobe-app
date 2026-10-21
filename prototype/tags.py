"""Measured tags (Phase 2.3), reference implementation for the Kotlin port in domain/tags.

All measurements are made in the garment photo's own frame, using that photo's pose landmarks (px) and
person mask, so they don't depend on matching the body figure's pose. The body figure is only used for
leg width (garment width vs my legs) and side-view thickness (garment outline vs my body outline).
Every tag has a confidence 0..1; rules marked LOW are heuristics waiting for more test photos.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# Named colours in CIELAB (D65). Nearest name wins.
COLOUR_NAMES = {
    "black": (12, 0, 0), "charcoal": (30, 0, -2), "grey": (55, 0, -1), "light grey": (78, 0, -1),
    "white": (96, 0, 1), "cream": (92, 0, 12), "beige": (80, 4, 18), "tan": (66, 10, 28),
    "brown": (38, 16, 26), "camel": (62, 14, 38), "olive": (48, -10, 36), "khaki": (70, -2, 28),
    "green": (50, -45, 30), "forest green": (32, -28, 14), "mint": (88, -22, 8), "teal": (48, -28, -8),
    "navy": (20, 10, -36), "blue": (45, 15, -60), "light blue": (78, -8, -22), "denim": (45, -2, -24),
    "purple": (38, 40, -40), "lavender": (78, 14, -18), "burgundy": (28, 38, 12), "red": (50, 68, 48),
    "pink": (72, 32, 2), "orange": (65, 40, 64), "mustard": (68, 8, 62), "yellow": (88, -6, 80),
}


def to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


# ---------------------------------------------------------------- colour

def colours(rgb: np.ndarray, mask: np.ndarray, k: int = 4, min_share: float = 0.05) -> list[dict]:
    """Main colours with share, name, lightness, chroma and warm/cool/neutral."""
    px = to_lab(rgb)[mask]
    if len(px) < 50:
        return []
    px = px[np.random.default_rng(0).choice(len(px), min(len(px), 30000), replace=False)].astype(np.float32)
    k = min(k, len(px))
    _, lbl, centers = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.3), 3,
                                 cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lbl.ravel(), minlength=k) / len(lbl)
    # merge clusters closer than 10 Delta E (shading of one colour)
    order = list(np.argsort(-share))
    merged: list[list] = []
    for i in order:
        for m in merged:
            if np.linalg.norm(centers[i] - m[0]) < 10:
                m[0] = (m[0] * m[1] + centers[i] * share[i]) / (m[1] + share[i])
                m[1] += share[i]
                break
        else:
            merged.append([centers[i].copy(), share[i]])
    # then merge clusters that get the same colour name (e.g. two shades of black)
    named: dict[str, list] = {}
    for c, s in merged:
        name = min(COLOUR_NAMES, key=lambda n: np.linalg.norm(np.array(COLOUR_NAMES[n]) - c))
        if name in named:
            m = named[name]
            m[0] = (m[0] * m[1] + c * s) / (m[1] + s)
            m[1] += s
        else:
            named[name] = [c, s]
    out = []
    for name, (c, s) in sorted(named.items(), key=lambda kv: -kv[1][1]):
        if s < min_share:
            continue
        L, a, b = (float(v) for v in c)
        chroma = math.hypot(a, b)
        hue = math.degrees(math.atan2(b, a)) % 360
        temp = "neutral" if chroma < 10 else ("warm" if hue < 110 or hue > 330 else "cool")
        out.append({"name": name, "share": round(float(s), 3), "lab": [round(L, 1), round(a, 1), round(b, 1)],
                    "lightness": round(L / 100, 2), "chroma": round(chroma, 1), "temperature": temp})
    return out


# ---------------------------------------------------------------- body regions

def _capsule(shape, p0, p1, r) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.line(m, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 1, max(1, int(2 * r)))
    cv2.circle(m, tuple(int(v) for v in p0), int(r), 1, -1)
    cv2.circle(m, tuple(int(v) for v in p1), int(r), 1, -1)
    return m.astype(bool)


def torso_height(lm: np.ndarray) -> float:
    return float(np.linalg.norm((lm[11, :2] + lm[12, :2]) / 2 - (lm[23, :2] + lm[24, :2]) / 2))


def body_regions(lm: np.ndarray, person: np.ndarray, min_vis: float = 0.5) -> dict[str, np.ndarray]:
    """Body regions from landmarks, limited to the person mask. Hidden limbs (side views) are skipped."""
    shape = person.shape
    T = torso_height(lm)
    vis = lambda *ids: all(lm[i, 2] >= min_vis for i in ids)
    regions: dict[str, np.ndarray] = {}

    def add(name, m):
        regions[name] = regions.get(name, np.zeros(shape, bool)) | (m & person)

    torso_pts = [i for i in (11, 12, 24, 23) if lm[i, 2] >= min_vis]
    if len(torso_pts) >= 3:
        m = np.zeros(shape, np.uint8)
        pts = lm[[11, 12, 24, 23], :2]
        c = pts.mean(axis=0)
        cv2.fillPoly(m, [(c + (pts - c) * np.array([1.15, 1.0])).astype(np.int32)], 1)
        add("torso", m.astype(bool))
    if vis(0):
        m = np.zeros(shape, np.uint8)
        cv2.circle(m, tuple(int(v) for v in lm[0, :2]), int(0.35 * T), 1, -1)
        add("head", m.astype(bool))
    knees = np.zeros(shape, bool)
    for knee in (25, 26):
        if vis(knee):
            m = np.zeros(shape, np.uint8)
            cv2.circle(m, tuple(int(v) for v in lm[knee, :2]), int(0.14 * T), 1, -1)
            knees |= m.astype(bool)
    if knees.any():
        add("knees", knees)
    for s, e, w, hip, knee, ankle, heel, toe in ((11, 13, 15, 23, 25, 27, 29, 31), (12, 14, 16, 24, 26, 28, 30, 32)):
        if vis(s, e):
            add("upperArms", _capsule(shape, lm[s, :2], lm[e, :2], 0.16 * T))
        if vis(e, w):
            add("forearms", _capsule(shape, lm[e, :2], lm[w, :2], 0.13 * T))
        if vis(hip, knee):
            add("thighs", _capsule(shape, lm[hip, :2], lm[knee, :2], 0.22 * T) & ~knees)
        if vis(knee, ankle):
            add("lowerLegs", _capsule(shape, lm[knee, :2], lm[ankle, :2], 0.15 * T) & ~knees)
        if vis(heel, toe):
            add("feet", _capsule(shape, lm[heel, :2], lm[toe, :2], 0.10 * T))
    # arms win over torso where they overlap (arms at the sides)
    arms = regions.get("upperArms", np.zeros(shape, bool)) | regions.get("forearms", np.zeros(shape, bool))
    if "torso" in regions:
        regions["torso"] &= ~arms
    return regions


def coverage(mask: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, float]:
    return {r: round(float((mask & m).sum() / m.sum()), 3) for r, m in regions.items() if m.sum() > 200}


# ---------------------------------------------------------------- lengths, rise, legs

def sleeve_length(cov: dict) -> tuple[str, float]:
    up, fo = cov.get("upperArms"), cov.get("forearms")
    if up is None:
        return "unknown", 0.0
    fo = fo or 0.0
    if fo > 0.75:
        return "long", 0.9
    if fo > 0.35:
        return "three-quarter", 0.7
    if up > 0.6:
        return "short", 0.8
    if up > 0.2:
        return "cap", 0.6
    return "sleeveless", 0.8


def relative_y(y: float, lm: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    """0 at the mid-point of landmarks a, 1 at the mid-point of b (vertical)."""
    ya = (lm[a[0], 1] + lm[a[1], 1]) / 2
    yb = (lm[b[0], 1] + lm[b[1], 1]) / 2
    span = yb - ya
    if abs(span) < 1:
        return 0.0
    return float((y - ya) / span)


def top_length(mask: np.ndarray, lm: np.ndarray) -> tuple[str, float, float]:
    ys = np.nonzero(mask.any(axis=1))[0]
    if len(ys) == 0:
        return "unknown", 0.0, 0.0
    t = relative_y(np.percentile(np.nonzero(mask)[0], 99), lm, (23, 24), (25, 26))  # 0 hip, 1 knee
    name = "cropped" if t < -0.25 else "waist" if t < -0.05 else "hip" if t < 0.25 else "long" if t < 0.6 else "tunic"
    return name, round(t, 2), 0.75


def bottoms_length(mask: np.ndarray, lm: np.ndarray) -> tuple[str, float, float]:
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        return "unknown", 0.0, 0.0
    t = relative_y(np.percentile(ys, 99), lm, (25, 26), (27, 28))  # 0 knee, 1 ankle
    if t < -0.6:
        name = "short"
    elif t < -0.15:
        name = "above-knee"
    elif t < 0.2:
        name = "knee"
    elif t < 0.7:
        name = "midi"
    elif t < 0.95:
        name = "ankle"
    else:
        name = "full"
    return name, round(t, 2), 0.75


def waistband_covered(mask: np.ndarray, upper: np.ndarray, band: int = 12) -> bool:
    """True if a top (garment or base top) sits directly above the bottom piece's upper edge."""
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(cols) == 0:
        return False
    tops = np.array([np.argmax(mask[:, x]) for x in cols])
    hits = [upper[max(0, y - band):y, x].any() for x, y in zip(cols, tops)]
    return float(np.mean(hits)) > 0.5


def rise(mask: np.ndarray, lm: np.ndarray, upper: np.ndarray | None = None) -> tuple[str, float, float]:
    """Waistband height: 0 at the hips, 1 at the shoulders. Needs the waistband visible (top tucked in or
    cropped); otherwise "unknown". LOW confidence until checked on real pants."""
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        return "unknown", 0.0, 0.0
    if upper is not None and waistband_covered(mask, upper):
        return "unknown (waistband covered)", 0.0, 0.0
    r = -relative_y(np.percentile(ys, 1), lm, (23, 24), (11, 12))
    name = "low" if r < 0.08 else "mid" if r < 0.22 else "high"
    return name, round(r, 2), 0.4


def _run_width(mask_row: np.ndarray, x: float) -> int:
    """Width of the mask run on this row that contains (or is nearest to) column x."""
    xs = np.nonzero(mask_row)[0]
    if len(xs) == 0:
        return 0
    runs = np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)
    best = min(runs, key=lambda r: 0 if r[0] <= x <= r[-1] else min(abs(r[0] - x), abs(r[-1] - x)))
    return int(best[-1] - best[0] + 1)


def leg_widths(mask: np.ndarray, lm: np.ndarray) -> dict[str, float]:
    """Mean of left/right leg widths (px) at thigh, knee, calf and ankle levels."""
    out = {}
    levels = {"thigh": ((23, 25), (24, 26), 0.3), "knee": ((25, 25), (26, 26), 0.0),
              "calf": ((25, 27), (26, 28), 0.5), "ankle": ((27, 27), (28, 28), 0.0)}
    for name, (left, right, t) in levels.items():
        ws = []
        for a, b in (left, right):
            if lm[a, 2] < 0.5 or lm[b, 2] < 0.5:
                continue
            p = lm[a, :2] + (lm[b, :2] - lm[a, :2]) * t
            y = int(round(p[1]))
            if 0 <= y < mask.shape[0]:
                w = _run_width(mask[y], p[0])
                if w:
                    ws.append(w)
        if ws:
            out[name] = float(np.mean(ws))
    return out


def leg_shape(garment_w: dict, garment_torso: float, body_w: dict, body_torso: float) -> tuple[str, dict, float]:
    """Garment leg widths vs my legs. Each photo's widths are divided by its own torso height, so camera
    distance and pose differences between the photos cancel out. LOW confidence."""
    ratios = {k: round((garment_w[k] / garment_torso) / (body_w[k] / body_torso), 2)
              for k in garment_w if k in body_w and body_w[k] > 0}
    if not {"knee", "ankle"} <= ratios.keys():
        return "unknown", ratios, 0.0
    knee, ankle, thigh = ratios["knee"], ratios["ankle"], ratios.get("thigh", ratios["knee"])
    taper = garment_w["ankle"] / garment_w["knee"]
    if taper > 1.15 and ankle > 1.3:
        name = "flare"
    elif knee < 1.15 and ankle < 1.15:
        name = "skinny"
    elif knee > 1.5 and ankle > 1.5:
        name = "wide"
    elif taper < 0.8 and thigh > 1.2:
        name = "tapered"
    else:
        name = "straight"
    return name, ratios, 0.45


def thickness(piece_body_frame: np.ndarray, body_person: np.ndarray, body_lm: np.ndarray) -> tuple[int, float, float]:
    """Side view: how far the garment stands off my body (torso band), relative to torso height. LOW."""
    T = torso_height(body_lm)
    y0 = int(min(body_lm[11, 1], body_lm[12, 1]) + 0.2 * T)
    y1 = int(max(body_lm[23, 1], body_lm[24, 1]) - 0.1 * T)
    gaps = []
    for y in range(max(0, y0), min(body_person.shape[0], y1)):
        g = np.nonzero(piece_body_frame[y])[0]
        b = np.nonzero(body_person[y])[0]
        if len(g) > 5 and len(b) > 5:
            gaps.append(((g[-1] - g[0]) - (b[-1] - b[0])) / 2)
    if not gaps:
        return 0, 0.0, 0.0
    gap = float(np.median(gaps)) / T
    level = 1 if gap < 0.02 else 2 if gap < 0.04 else 3 if gap < 0.07 else 4 if gap < 0.1 else 5
    return level, round(gap, 3), 0.35


def sheerness(keep: np.ndarray, region: np.ndarray) -> tuple[str, float, float]:
    """Base colour showing through: small base-coloured holes spread inside the piece (big holes are
    openings like an open zip and don't count). LOW confidence."""
    filled = region.copy()
    holes = filled & ~keep
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
    area = max(1, int(region.sum()))
    small = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < 0.01 * area)
    share = small / area
    name = "opaque" if share < 0.03 else "semi-sheer" if share < 0.12 else "sheer"
    return name, round(share, 3), 0.3
