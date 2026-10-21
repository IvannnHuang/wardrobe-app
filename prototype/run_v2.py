"""Run the new pipeline (clothes model cut-out + measured tags) on the test photos.

Usage (from prototype/):  .venv/Scripts/python run_v2.py
Outputs: output/v2/<set>/<garment>/<view>_pieces.jpg, <view>_<piece>.png, tags.json
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image

import clothes as C
import experiment as E
import pipeline as P
import tags as T

OUT = P.OUTPUT / "v2"
PIECE_COLOURS = {"top": (0, 200, 0), "dress": (200, 0, 200), "bottom": (0, 120, 255), "shoes": (255, 160, 0),
                 "hat": (255, 255, 0), "bag": (0, 255, 255), "belt": (128, 128, 255), "scarf": (255, 0, 128)}


def piece_tags(name, mask, region, base_frac, g, body, view, align_scale, align_m, labels=None) -> dict:
    lm, person = g.landmarks, g.person > 0.5
    out: dict = {"colours": T.colours(g.rgb, mask), "baseColourShare": round(base_frac, 3)}
    if view == "front":
        out["coverage"] = T.coverage(mask, T.body_regions(lm, person))
    if name in ("top", "dress"):
        out["sleeve"] = T.sleeve_length(out.get("coverage", {}))[0] if view == "front" else None
        out["topLength"] = T.top_length(mask, lm)[:2]
    if name in ("bottom", "dress"):
        out["bottomsLength"] = T.bottoms_length(mask, lm)[:2]
    if name == "bottom":
        upper = np.isin(labels, [4, 7]) if labels is not None else None
        out["rise"] = T.rise(mask, lm, upper)[:2]
        if view == "front":
            gw = T.leg_widths(mask, lm)
            bw = T.leg_widths(body.person > 0.5, body.landmarks)
            out["legShape"] = T.leg_shape(gw, T.torso_height(lm), bw, T.torso_height(body.landmarks))[:2]
    if view in ("left", "right") and name in ("top", "dress") and align_m is not None:
        h, w = body.rgb.shape[:2]
        in_body = cv2.warpAffine(mask.astype(np.uint8), align_m, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
        out["thickness"] = T.thickness(in_body, body.person > 0.5, body.landmarks)[:2]
    out["sheerness"] = T.sheerness(mask, region)[:2]
    return out


def main() -> None:
    mp_models = P.Models()
    model = C.ClothesModel()
    all_tags = {}
    for set_name, (folder, mirror) in E.SETS.items():
        s = P.Settings(mirror=mirror)
        for gdir in sorted(p for p in (folder / "garments").iterdir() if p.is_dir()):
            out_dir = OUT / set_name / gdir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            garment_tags = {}
            for view in P.VIEWS:
                gp, bp = P.find_photo(gdir, view), P.find_photo(folder / "body", view)
                if gp is None or bp is None:
                    continue
                body = mp_models.analyse(P.load_photo(bp, s))
                g = mp_models.analyse(P.load_photo(gp, s))
                base = C.learn_base(body.rgb, model.labels(body.rgb))
                labels = model.labels(g.rgb)
                pieces, notes = C.extract_pieces(g.rgb, labels, base)

                report = P.Report(gdir.name, view)
                try:
                    P.align(g, body, s, report)
                    src, dst = g.landmarks[report.align_points, :2], body.landmarks[report.align_points, :2]
                    align_m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
                except RuntimeError:
                    align_m = None

                view_tags = {"notes": notes, "pieces": {}}
                for pc in pieces:
                    view_tags["pieces"][pc.name] = piece_tags(pc.name, pc.mask, pc.region, pc.base_frac, g, body,
                                                              view, report.align_scale or 1.0, align_m, labels)
                    Image.fromarray(np.dstack([g.rgb, pc.mask.astype(np.uint8) * 255]), "RGBA").save(
                        out_dir / f"{view}_{pc.name}.png")
                # Check of bottoms tags on the base pants (the only pants in the test set).
                pants = np.isin(labels, [5, 6])
                if pants.mean() > 0.01 and not any(pc.name == "bottom" for pc in pieces):
                    m = C.clean(pants, 0.02)
                    view_tags["debug_base_pants"] = piece_tags("bottom", m, pants, 1.0, g, body, view,
                                                               report.align_scale or 1.0, align_m, labels)
                garment_tags[view] = view_tags

                overlay = g.rgb.copy()
                for pc in pieces:
                    overlay[pc.mask] = (0.45 * overlay[pc.mask] + 0.55 * np.array(PIECE_COLOURS[pc.name])).astype(np.uint8)
                vis = np.hstack([g.rgb, overlay])
                vis = cv2.resize(vis, None, fx=700 / vis.shape[0], fy=700 / vis.shape[0])
                cv2.imwrite(str(out_dir / f"{view}_pieces.jpg"), vis[..., ::-1])
                print(f"{set_name}/{gdir.name}/{view}: pieces {[p.name for p in pieces]} | {'; '.join(notes)}")
            all_tags[f"{set_name}/{gdir.name}"] = garment_tags
    (OUT / "tags.json").write_text(json.dumps(all_tags, indent=1))


if __name__ == "__main__":
    main()
