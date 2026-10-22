"""Lighting robustness of base-outfit removal: which colour comparison survives light changes?

Variants:
  none       compare LAB as is (current)
  skin       shift the garment photo's LAB so its skin (face + arms) matches the body figure's skin
  lowL       Delta E with lightness weighted 0.5 (exposure changes mostly move lightness)
  skin+lowL  both
Metrics per garment photo: base share of the bottom (base pants: want >= 0.6 -> dropped) and of the top
(the new jacket: want low), and the top's kept area relative to the "none" variant.

Usage (from prototype/):  .venv/Scripts/python experiment_light.py
"""

from __future__ import annotations

import json

import numpy as np

import clothes as C
import experiment as E
import pipeline as P

SETS = E.SETS
SKIN = [11, 14, 15]  # face, arms


def base_share(lab, region, palette, l_weight):
    if not len(palette) or region.sum() == 0:
        return 0.0, region
    w = np.array([l_weight, 1, 1], np.float32)
    d = np.min(np.linalg.norm((lab[..., None, :] - palette[None, None]) * w, axis=3), axis=2)
    is_base = region & (d < 16)
    return float(is_base.sum() / region.sum()), region & ~is_base


def main() -> None:
    mp_models = P.Models()
    model = C.ClothesModel()
    rows = []
    for set_name, (folder, mirror) in SETS.items():
        s = P.Settings(mirror=mirror)
        for gdir in sorted(p for p in (folder / "garments").iterdir() if p.is_dir()):
            for view in P.VIEWS:
                gp, bp = P.find_photo(gdir, view), P.find_photo(folder / "body", view)
                if not gp or not bp:
                    continue
                b_rgb, g_rgb = P.load_photo(bp, s), P.load_photo(gp, s)
                b_lab, g_lab = C.lab_of(b_rgb), C.lab_of(g_rgb)
                b_lbl, g_lbl = model.labels(b_rgb), model.labels(g_rgb)
                base = C.learn_base(b_rgb, b_lbl)
                b_skin, g_skin = np.isin(b_lbl, SKIN), np.isin(g_lbl, SKIN)
                shift = (np.median(b_lab[b_skin], axis=0) - np.median(g_lab[g_skin], axis=0)
                         if b_skin.sum() > 500 and g_skin.sum() > 500 else np.zeros(3, np.float32))
                top, bottom = np.isin(g_lbl, [4]), np.isin(g_lbl, [5, 6])
                row = {"photo": f"{set_name}/{gdir.name}/{view}", "skin_shift": [round(float(v), 1) for v in shift]}
                ref_area = None
                for name, use_skin, lw in (("none", False, 1.0), ("skin", True, 1.0), ("lowL", False, 0.5), ("skin+lowL", True, 0.5)):
                    lab = g_lab + shift if use_skin else g_lab
                    bot_share, _ = base_share(lab, bottom, base.palettes["bottom"], lw)
                    top_share, top_keep = base_share(lab, top, base.palettes["top"], lw)
                    area = float(top_keep.mean())
                    ref_area = ref_area or area
                    row[name] = {"bottom_base": round(bot_share, 2), "top_base": round(top_share, 2),
                                 "top_area_vs_none": round(area / ref_area, 2) if ref_area else None}
                rows.append(row)
                print(row["photo"], "skin shift", row["skin_shift"], "|",
                      " | ".join(f"{k}: bottom {row[k]['bottom_base']:.2f} top {row[k]['top_base']:.2f} area {row[k]['top_area_vs_none']}"
                                 for k in ("none", "skin", "lowL", "skin+lowL")))
    summary = {}
    for k in ("none", "skin", "lowL", "skin+lowL"):
        summary[k] = {
            "bottoms_dropped": sum(r[k]["bottom_base"] >= 0.6 for r in rows),
            "photos": len(rows),
            "mean_top_base": round(float(np.mean([r[k]["top_base"] for r in rows])), 3),
            "min_top_area_vs_none": min(r[k]["top_area_vs_none"] for r in rows),
        }
    print(json.dumps(summary, indent=1))
    (P.OUTPUT / "experiment_light.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
