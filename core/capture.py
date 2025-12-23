# core/capture.py
# OCR 캡처 기능을 서버 엔드포인트로 제공

import sys
import os
import asyncio
import re
from typing import List, Optional, Union
import numpy as np
import cv2
from PIL import Image
from mss import mss
import pyperclip

# ocrtest.py의 함수들을 import
# ocrtest.py의 모든 OCR 관련 함수들을 여기서 사용
import importlib.util

# ocrtest.py의 경로
ocr_test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocrtest.py")
spec = importlib.util.spec_from_file_location("ocrtest", ocr_test_path)
ocrtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ocrtest)

# ocrtest.py의 함수들을 직접 사용
sanitize_text = ocrtest.sanitize_text
preprocess_for_code_pil = ocrtest.preprocess_for_code_pil
open_image_any = ocrtest.open_image_any
WordBox = ocrtest.WordBox
LineBox = ocrtest.LineBox
cluster_lines = ocrtest.cluster_lines
estimate_char_width = ocrtest.estimate_char_width
normalize_code_line = ocrtest.normalize_code_line
reconstruct_text_from_words = ocrtest.reconstruct_text_from_words
merge_winrt_lines = ocrtest.merge_winrt_lines
_run_coro_sync = ocrtest._run_coro_sync
_create_winrt_engine = ocrtest._create_winrt_engine
_winrt_recognize_async = ocrtest._winrt_recognize_async
_winrt_words_from_result = ocrtest._winrt_words_from_result
_winrt_lines_text = ocrtest._winrt_lines_text
get_winrt_words = ocrtest.get_winrt_words
image_to_text_winrt = ocrtest.image_to_text_winrt
_build_whitelist = ocrtest._build_whitelist
tesseract_word_boxes = ocrtest.tesseract_word_boxes
image_to_text = ocrtest.image_to_text
capture_fullscreen_bgr = ocrtest.capture_fullscreen_bgr
copy_to_clipboard = ocrtest.copy_to_clipboard
select_roi_auto = ocrtest.select_roi_auto
merge_tesseract_winrt_results = ocrtest.merge_tesseract_winrt_results
get_tesseract_words = ocrtest.get_tesseract_words
check_winrt_available = ocrtest.check_winrt_available

def capture_and_ocr() -> dict:
    """
    화면 캡처 -> 드래그로 영역 선택 -> OCR 인식 -> 클립보드 저장
    
    Returns:
        dict: {
            "success": bool,
            "text": str (OCR 결과),
            "method": str (사용된 OCR 방법),
            "error": str (에러 메시지, 실패 시)
        }
    """
    try:
        # Tesseract와 WinRT 사용 가능 여부 확인
        tesseract_available = ocrtest.pytesseract is not None
        winrt_available, winrt_error = check_winrt_available()
        
        if not tesseract_available and not winrt_available:
            return {
                "success": False,
                "error": "Tesseract와 WinRT 모두 사용 불가능합니다. OCR을 수행할 수 없습니다.",
                "text": "",
                "method": ""
            }
        
        # 화면 캡처
        print("📸 화면 캡처 중...")
        screen = capture_fullscreen_bgr()
        print("✅ 화면 캡처 완료")
        
        # 드래그로 영역 선택
        print("🖱️ OpenCV 창을 열고 영역 선택을 기다리는 중...")
        print("   (서버 컴퓨터에서 '원하는 부분을 드래그로 박스치세요' 창이 열립니다)")
        cropped = select_roi_auto(screen, window_name="원하는 부분을 드래그로 박스치세요")
        print("✅ 영역 선택 완료")
        
        # OCR 인식
        print("🔍 OCR 인식 시작...")
        ocr_result = None
        ocr_method = None
        
        tesseract_words = None
        tesseract_text = None
        winrt_words = None
        winrt_text = None
        
        if tesseract_available:
            try:
                # 레이아웃용 WordBox 가져오기
                tesseract_words = get_tesseract_words(
                    cropped,
                    lang="kor+eng",
                    scale=4,
                    code_mode=True,
                    remove_emoji=True
                )
                # 전체 텍스트도 가져오기 (비교용)
                tesseract_text = image_to_text(
                    cropped,
                    lang="kor+eng",
                    scale=4,
                    code_mode=True,
                    layout=True,
                    normalize=True
                )
            except Exception as e:
                print(f"⚠ Tesseract OCR 실패: {e}")
        
        if winrt_available:
            try:
                # WordBox 가져오기 (병합용)
                winrt_words = get_winrt_words(
                    cropped,
                    scale=3,
                    code_mode=True,
                    remove_emoji=True
                )
                # 전체 텍스트도 가져오기 (비교용)
                winrt_text = image_to_text_winrt(
                    cropped,
                    scale=3,
                    code_mode=True,
                    normalize=True
                )
            except Exception as e:
                print(f"⚠ WinRT OCR 실패: {e}")
        
        # 결과 병합
        if tesseract_words and winrt_words:
            try:
                pil_img = open_image_any(cropped)
                ocr_result = merge_tesseract_winrt_results(
                    tesseract_words,
                    winrt_words,
                    pil_img
                )
                ocr_method = "Tesseract + WinRT 병합"
            except Exception as e:
                print(f"⚠ 병합 실패: {e}, Tesseract 결과 사용")
                if tesseract_text:
                    ocr_result = tesseract_text
                    ocr_method = "Tesseract (병합 실패)"
        elif tesseract_text:
            ocr_result = tesseract_text
            ocr_method = "Tesseract"
        elif winrt_text:
            ocr_result = winrt_text
            ocr_method = "WinRT"
        
        if ocr_result:
            # 클립보드에 저장
            print(f"✅ OCR 완료 ({ocr_method})")
            copy_to_clipboard(ocr_result)
            print("📋 클립보드에 저장 완료")
            return {
                "success": True,
                "text": ocr_result,
                "method": ocr_method,
                "error": None
            }
        else:
            return {
                "success": False,
                "error": "모든 OCR 엔진이 실패했습니다.",
                "text": "",
                "method": ""
            }
    
    except ValueError as e:
        # 사용자가 ROI 선택을 취소한 경우
        if "취소" in str(e) or "cancel" in str(e).lower():
            return {
                "success": False,
                "error": "사용자가 영역 선택을 취소했습니다.",
                "text": "",
                "method": ""
            }
        return {
            "success": False,
            "error": str(e),
            "text": "",
            "method": ""
        }
    except Exception as e:
        import traceback
        error_msg = f"OCR 처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        return {
            "success": False,
            "error": error_msg,
            "text": "",
            "method": ""
        }

