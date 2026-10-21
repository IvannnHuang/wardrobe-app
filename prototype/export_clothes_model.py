"""Export the clothes model for the tablet and check that the exported versions still agree with PyTorch.

Variants: fp32 ONNX, fp16 ONNX, int8 (dynamic, weights only) ONNX. For each test photo we compare the
label map against PyTorch: pixel agreement over the person, and IoU of each clothing piece.
The chosen file is copied to models/clothes_segformer_b2.onnx (git-ignored, see scripts/fetch-models.sh).

Usage (from prototype/):  .venv/Scripts/python export_clothes_model.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic
import onnx

import experiment as E
import pipeline as P

OUT = P.OUTPUT / "export"
NAME = "b2"
CHOSEN = "clothes_segformer_b2_fp16.onnx"
SIZE = 512


class Wrapper(torch.nn.Module):
    """Plain tensor in → logits out (1x18x128x128), so the ONNX graph has one input and one output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def preprocess(rgb: np.ndarray) -> np.ndarray:
    import cv2
    x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return x.transpose(2, 0, 1)[None].copy()


def labels_from_logits(logits: np.ndarray, shape) -> np.ndarray:
    t = torch.nn.functional.interpolate(torch.from_numpy(logits.astype(np.float32)), size=shape, mode="bilinear",
                                        align_corners=False)
    return t.argmax(dim=1)[0].numpy().astype(np.uint8)


def compare(ref: np.ndarray, got: np.ndarray) -> dict:
    person = ref > 0
    out = {"agree_person": round(float((ref == got)[person].mean()), 4)}
    for piece, (ids, _) in E.PIECES.items():
        a, b = np.isin(ref, ids), np.isin(got, ids)
        if a.sum() < 500:
            continue
        out[f"iou_{piece}"] = round(float((a & b).sum() / max(1, (a | b).sum())), 4)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parser = E.Parser(NAME)
    fp32 = OUT / "b2_fp32.onnx"
    torch.onnx.export(Wrapper(parser.model).eval(), torch.zeros(1, 3, SIZE, SIZE), str(fp32),
                      input_names=["pixel_values"], output_names=["logits"], opset_version=17, dynamo=False)
    fp16 = OUT / "b2_fp16.onnx"
    onnx.save(float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(fp16))
    int8 = OUT / "b2_int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)

    variants = {"fp32": fp32, "fp16": fp16, "int8": int8}
    sessions = {k: ort.InferenceSession(str(v), providers=["CPUExecutionProvider"]) for k, v in variants.items()}

    photos = []
    for set_name, (folder, mirror) in E.SETS.items():
        for sub in ["body"] + [f"garments/{p.name}" for p in sorted((folder / "garments").iterdir()) if p.is_dir()]:
            for view in P.VIEWS:
                f = P.find_photo(folder / sub, view)
                if f:
                    photos.append((f"{set_name}/{sub}/{view}", P.load_photo(f, P.Settings(mirror=mirror))))

    report = {k: {"size_mb": round(v.stat().st_size / 1e6, 1), "per_photo": {}} for k, v in variants.items()}
    for name, rgb in photos:
        x = preprocess(rgb)
        with torch.no_grad():
            ref = labels_from_logits(parser.model(pixel_values=torch.from_numpy(x)).logits.numpy(), rgb.shape[:2])
        for k, sess in sessions.items():
            t = time.time()
            logits = sess.run(None, {"pixel_values": x})[0]
            sec = time.time() - t
            r = compare(ref, labels_from_logits(logits, rgb.shape[:2]))
            r["sec"] = round(sec, 2)
            report[k]["per_photo"][name] = r
    for k, r in report.items():
        rows = list(r["per_photo"].values())
        r["mean_agree_person"] = round(float(np.mean([x["agree_person"] for x in rows])), 4)
        ious = [v for x in rows for kk, v in x.items() if kk.startswith("iou_")]
        r["mean_iou"] = round(float(np.mean(ious)), 4)
        r["min_iou"] = round(float(np.min(ious)), 4)
        r["mean_sec_cpu"] = round(float(np.mean([x["sec"] for x in rows])), 2)
        print(f"{k}: {r['size_mb']} MB, agree {r['mean_agree_person']:.2%}, IoU mean {r['mean_iou']:.3f} "
              f"min {r['min_iou']:.3f}, {r['mean_sec_cpu']} s/photo (PC CPU)")
    (OUT / "report.json").write_text(json.dumps(report, indent=1))

    # fp16 chosen (2026-10-21): identical masks to PyTorch at half the size; int8 broke some pieces.
    for dest in (P.MODELS, P.ROOT.parent / "android/app/src/main/assets/models"):
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(fp16, dest / CHOSEN)
    print(f"copied {CHOSEN} ({hashlib.sha256(fp16.read_bytes()).hexdigest()[:16]}…) to prototype/models and app assets")


if __name__ == "__main__":
    main()
