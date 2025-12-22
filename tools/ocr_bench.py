# tools/ocr_bench.py
# 로컬 이미지 OCR 벤치/테스트 CLI
# - 엔진: tesseract(기본, 레이아웃 복원), winrt(보조), easyocr, paddleocr
# - 목적: 빠르게 captures/*.png로 품질 확인

from __future__ import annotations

import argparse
import os
import time
from typing import List

import cv2
import numpy as np
from PIL import Image

from app.capture.ocr import (
    image_to_text,            # Tesseract main (layout 가능)
    image_to_text_winrt,      # WinRT aux (ko/en merge)
    image_to_text_easyocr,
    image_to_text_paddleocr,
)


def open_image(path: str, scale: int = 1) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if scale and scale != 1:
        w, h = img.size
        img = img.resize((w * scale, h * scale), Image.BICUBIC)
    return img


def open_bgr(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"image load failed: {path}")
    return bgr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--img", required=True)
    p.add_argument("--engine", choices=["tesseract", "winrt", "easyocr", "paddleocr"], default="tesseract")
    p.add_argument("--lang", default="kor+eng")
    p.add_argument("--scale", type=int, default=1)

    # tesseract options
    p.add_argument("--psm", type=int, default=6)
    p.add_argument("--oem", type=int, default=3)

    # common
    p.add_argument("--code-mode", action="store_true")
    p.add_argument("--layout", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--indent-step", type=int, default=4)

    p.add_argument("--out", default="")
    p.add_argument("--print", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.img):
        print(f"image not found: {args.img}")
        return 2

    t0 = time.time()

    try:
        if args.engine == "tesseract":
            # main: use image_to_data layout when --layout is set (recommended)
            bgr = open_bgr(args.img)
            result = image_to_text(
                bgr,
                args.lang,
                scale=int(args.scale),
                code_mode=bool(args.code_mode),
                layout=bool(args.layout or args.code_mode),  # code-mode면 기본 layout 권장
                normalize=bool(args.normalize or args.code_mode),
                indent_step=int(args.indent_step),
                psm=int(args.psm),
                oem=int(args.oem),
            )

        elif args.engine == "winrt":
            pil_img = open_image(args.img, scale=int(args.scale))
            result = image_to_text_winrt(
                pil_img,
                scale=1,  # 이미 PIL에서 스케일 적용했음
                code_mode=bool(args.code_mode),
                normalize=bool(args.normalize or args.code_mode),
                indent_step=int(args.indent_step),
            )

        elif args.engine == "easyocr":
            bgr = open_bgr(args.img)
            langs: List[str] = ["ko", "en"] if ("kor" in args.lang.lower() or "korean" in args.lang.lower()) else ["en"]
            # easyocr는 내부에서 자체 라인 합치기 -> 들여쓰기 복원은 별도 작업 필요
            result = image_to_text_easyocr(bgr, langs=langs, gpu=False)

        else:
            bgr = open_bgr(args.img)
            plang = "korean" if ("kor" in args.lang.lower() or "korean" in args.lang.lower()) else "en"
            result = image_to_text_paddleocr(bgr, lang=plang, use_gpu=False)

    except Exception as e:
        print(f"OCR failed: {e}")
        return 1

    dt_ms = (time.time() - t0) * 1000.0
    print(f"done. engine={args.engine} scale={args.scale} time={dt_ms:.1f}ms")

    if args.print or not args.out:
        print(result)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result if result.endswith("\n") else (result + "\n"))
        print(f"saved: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
