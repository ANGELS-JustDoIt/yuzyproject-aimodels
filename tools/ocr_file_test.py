# tools/ocr_file_test.py

import cv2
from app.capture.ocr import image_to_text

IMG_PATH = "crop.png"
LANG = "kor+eng"
SCALE = 3

if __name__ == "__main__":
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise RuntimeError(f"이미지 로드 실패: {IMG_PATH}")

    text = image_to_text(img, lang=LANG, scale=SCALE)
    print("\n===== OCR RESULT =====\n")
    print(text)
