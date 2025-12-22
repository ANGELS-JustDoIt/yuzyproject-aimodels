# app/service/capture_service.py

from app.capture.screen_capture import capture_fullscreen_bgr
from app.capture.roi_crop import select_roi
from app.capture.ocr import image_to_text
from app.clipboard.clipboard import copy_to_clipboard


def capture_crop_ocr_to_clipboard(lang: str = "kor+eng") -> str:
    """
    한 번의 호출로:
    화면 캡처 → ROI 선택 → OCR → 클립보드 복사 → 텍스트 반환
    """
    screen = capture_fullscreen_bgr()
    cropped = select_roi(screen)
    text = image_to_text(cropped, lang=lang)

    copy_to_clipboard(text)
    return text


if __name__ == "__main__":
    try:
        result = capture_crop_ocr_to_clipboard()
        print("\n✅ OCR 완료 + 클립보드 복사 완료\n")
        print(result)

    except Exception as e:
        print(f"[ERROR] {e}")
