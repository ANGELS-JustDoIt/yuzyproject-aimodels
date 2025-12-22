# app/main.py

from app.service.capture_service import capture_crop_ocr_to_clipboard

if __name__ == "__main__":
    text = capture_crop_ocr_to_clipboard(lang="kor+eng")
    print("\n✅ DONE (copied to clipboard)\n")
    print(text)