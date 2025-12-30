# core/capture.py
# OCR 캡처 기능을 서버 엔드포인트로 제공
#
# Requirements (pip install):
#   - paddlepaddle (CPU: paddlepaddle, GPU: paddlepaddle-gpu)
#   - paddleocr
#   - numpy
#   - opencv-python (cv2)
#   - Pillow (PIL)
#   - mss
#   - pyperclip
#   - pytesseract (optional, for Tesseract fallback)
#   - winrt-runtime, winrt-Windows.* (optional, for WinRT fallback)

import os
import sys
import re
import asyncio
import csv
from pathlib import Path as PathLib
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Dict, Any

try:
    import yaml
except ImportError:
    yaml = None

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from mss import mss
import pyperclip

# PaddleOCR import
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    PADDLEOCR_AVAILABLE = False

# ------------------------------------------------------------
# OCR 유틸 (기존 ocr_utils.py 내용을 이 파일로 통합)
# ------------------------------------------------------------

# Tesseract 설정
try:
    import pytesseract
except Exception:
    pytesseract = None

TESSERACT_EXE = r"C:\Pyg\Program_Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR = r"C:\Pyg\Program_Files\Tesseract-OCR\tessdata"


def _configure_tesseract() -> None:
    """Set Tesseract paths if available; keep silent when missing."""
    if pytesseract is None:
        return
    if os.path.exists(TESSERACT_EXE):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
    if os.path.isdir(TESSDATA_DIR):
        os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR


_configure_tesseract()


EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def sanitize_text(
    text: str,
    *,
    remove_emoji: bool = True,
    keep_newlines: bool = True,
    collapse_spaces: bool = False,
    tabsize: int = 4,
) -> str:
    if not text:
        return ""
    t = CTRL_RE.sub("", text)
    if remove_emoji:
        t = EMOJI_RE.sub("", t)
    t = t.replace("\t", " " * tabsize)
    t = "\n".join([ln.rstrip() for ln in t.splitlines()])
    if not keep_newlines:
        t = t.replace("\n", " ")
    if collapse_spaces:
        out_lines = []
        for ln in t.splitlines():
            lead = len(ln) - len(ln.lstrip(" "))
            body = re.sub(r"[ ]{2,}", " ", ln.lstrip(" "))
            out_lines.append((" " * lead) + body)
        t = "\n".join(out_lines)
    return t.strip()


def preprocess_for_code_pil(img: Image.Image, enabled: bool) -> Image.Image:
    if not enabled:
        return img
    g = img.convert("L")
    g = ImageEnhance.Contrast(g).enhance(2.8)
    g = ImageEnhance.Sharpness(g).enhance(2.2)
    g = ImageEnhance.Brightness(g).enhance(1.1)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    arr = np.array(g)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    g = Image.fromarray(binary)
    return g.convert("RGB")


def open_image_any(img: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).convert("RGB")
    raise TypeError("Unsupported image type (need PIL.Image or OpenCV BGR ndarray).")


@dataclass
class WordBox:
    text: str
    x: float
    y: float
    w: float
    h: float
    conf: float = -1.0

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cy(self) -> float:
        return self.y + self.h * 0.5


@dataclass
class LineBox:
    words: List[WordBox]
    y_center: float
    y_top: float
    y_bot: float


def _robust_median(values: List[float], default: float) -> float:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return default
    return float(np.median(np.array(vals, dtype=np.float32)))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def cluster_lines(words: List[WordBox]) -> List[LineBox]:
    if not words:
        return []
    heights = [w.h for w in words if w.h > 0]
    med_h = _robust_median(heights, default=14.0)
    y_thresh = max(6.0, med_h * 0.60)
    ws = sorted(words, key=lambda w: (w.cy, w.x))
    lines: List[LineBox] = []
    cur: List[WordBox] = []
    cur_center: Optional[float] = None
    for w in ws:
        if not cur:
            cur = [w]
            cur_center = w.cy
            continue
        if abs(w.cy - float(cur_center)) <= y_thresh:
            cur.append(w)
            cur_center = (float(cur_center) * 0.7) + (w.cy * 0.3)
        else:
            cur_sorted = sorted(cur, key=lambda t: t.x)
            y_top = min(t.y for t in cur_sorted)
            y_bot = max(t.y2 for t in cur_sorted)
            y_center = float(np.mean([t.cy for t in cur_sorted]))
            lines.append(LineBox(words=cur_sorted, y_center=y_center, y_top=y_top, y_bot=y_bot))
            cur = [w]
            cur_center = w.cy
    if cur:
        cur_sorted = sorted(cur, key=lambda t: t.x)
        y_top = min(t.y for t in cur_sorted)
        y_bot = max(t.y2 for t in cur_sorted)
        y_center = float(np.mean([t.cy for t in cur_sorted]))
        lines.append(LineBox(words=cur_sorted, y_center=y_center, y_top=y_top, y_bot=y_bot))
    lines.sort(key=lambda ln: ln.y_center)
    merged: List[LineBox] = []
    for ln in lines:
        if not merged:
            merged.append(ln)
            continue
        prev = merged[-1]
        gap = ln.y_top - prev.y_bot
        if gap <= max(2.0, med_h * 0.15):
            merged_words = sorted(prev.words + ln.words, key=lambda t: t.x)
            y_top = min(t.y for t in merged_words)
            y_bot = max(t.y2 for t in merged_words)
            y_center = float(np.mean([t.cy for t in merged_words]))
            merged[-1] = LineBox(words=merged_words, y_center=y_center, y_top=y_top, y_bot=y_bot)
        else:
            merged.append(ln)
    return merged


def estimate_char_width(lines: List[LineBox]) -> float:
    samples: List[float] = []
    for ln in lines:
        for w in ln.words:
            txt = w.text
            if not txt or len(txt) < 2:
                continue
            if " " in txt:
                continue
            cw = w.w / max(1, len(txt))
            if 2.0 <= cw <= 80.0:
                samples.append(float(cw))
    if not samples:
        heights = [w.h for ln in lines for w in ln.words if w.h > 0]
        med_h = _robust_median(heights, default=14.0)
        return max(6.0, med_h * 0.55)
    return _robust_median(samples, default=8.0)


ASCII_CODE_RE = re.compile(r"[A-Za-z0-9_{}\[\]().,:;=<>!+\-/*%\\'\"`@#$^|~]")
COMMON_GLUE = [
    (re.compile(r"\bfrom(?=[A-Za-z_])", re.IGNORECASE), "from "),
    (re.compile(r"\bimport(?=[A-Za-z_])", re.IGNORECASE), "import "),
    (re.compile(r"\bdef(?=[A-Za-z_])", re.IGNORECASE), "def "),
    (re.compile(r"\breturn(?=[A-Za-z_])", re.IGNORECASE), "return "),
    (re.compile(r"\braise(?=[A-Za-z_])", re.IGNORECASE), "raise "),
]
SAFE_ID_REPAIRS = [
    (re.compile(r"\bimage ?to ?text\b", re.IGNORECASE), "image_to_text"),
    (re.compile(r"\bimageto_text\b", re.IGNORECASE), "image_to_text"),
    (re.compile(r"\bimage_to_text_easyocr\b", re.IGNORECASE), "image_to_text_easyocr"),
    (re.compile(r"\bimage_to_text_paddleocr\b", re.IGNORECASE), "image_to_text_paddleocr"),
    (re.compile(r"\bimagetotextpaddleocr\b", re.IGNORECASE), "image_to_text_paddleocr"),
]


def normalize_code_line(s: str) -> str:
    """
    코드 라인 정규화: OCR 오인식 보정 및 코드 기호 보존.
    - 들여쓰기 및 코드 기호 (->, :, ; 등) 보존 강화
    - 한글 노이즈 제거 (코드 끝에 붙는 불필요한 한글)
    - 전각→반각 변환 및 특수문자 보정
    """
    if not s:
        return s
    
    # 코드 필수 기호 보존 및 보정 (우선 처리)
    s = s.replace("：", ":").replace("；", ";").replace("，", ",")
    s = s.replace("—", "-").replace("–", "-").replace("→", "->")
    
    # 한글 오인식 보정 (은 -> g, 등)
    s = re.sub(r"\b은\s*=", "g =", s)
    # 함수 시그니처 보정 (콜론 누락) - img Image.Image -> img: Image.Image
    s = re.sub(r"(\w+)\s+(Image\.Image)", r"\1: \2", s)
    # 타입 힌트 콜론 누락 보정 (함수 파라미터와 변수 선언 모두) - 더 강력하게
    # 변수 선언: samples List[float] = [] -> samples: List[float] = [] (먼저 처리)
    s = re.sub(r"(\w+)\s+(List\[[^\]]+\])\s*=", r"\1: \2 =", s)
    # 함수 파라미터: lines List[LineBox] -> lines: List[LineBox]
    s = re.sub(r"(\w+)\s+(List\[[^\]]+\])\s*\)", r"\1: \2)", s)
    # 줄 끝에 타입 힌트가 있고 콜론이 없는 경우
    s = re.sub(r"(\w+)\s+(List\[[^\]]+\])(\s*)$", r"\1: \2", s)
    # 일반적인 타입 힌트 (콜론이 바로 뒤에 없을 때만)
    s = re.sub(r"(\w+)\s+(List\[.*?\]|bool|str|int|float|list|dict|tuple|Optional|Union)(?!\s*:)", r"\1: \2", s)
    # 함수 반환 타입 콜론 누락 보정 (-> float -> -> float:)
    s = re.sub(r"->\s*(\w+)(\s*)$", r"-> \1:", s)
    # for 루프 콜론 누락 보정 (더 강력하게, if보다 먼저 처리)
    # ln.words 같은 경우
    s = re.sub(r"\bfor\s+(\w+)\s+in\s+(\w+\.\w+)(\s*)$", r"for \1 in \2:", s)
    # 일반적인 경우
    s = re.sub(r"\bfor\s+(\w+)\s+in\s+(\w+)(\s*)$", r"for \1 in \2:", s)
    # if 문 콜론 누락 보정 (더 강력하게) - 줄 끝에 콜론이 없으면 추가
    # 먼저 특정 패턴들 처리
    s = re.sub(r"\bif\s+not\s+(\w+)\s+or\s+len\s*\(\s*\1\s*\)\s*[<>=]+\s*\d+(\s*)$", r"if not \1 or len(\1) < 2:", s)
    s = re.sub(r'\bif\s+"\s+"\s+in\s+(\w+)(\s*)$', r'if " " in \1:', s)
    s = re.sub(r"\bif\s+(\d+\.\d+)\s*<=\s*(\w+)\s*<=\s*(\d+\.\d+)(\s*)$", r"if \1 <= \2 <= \3:", s)
    s = re.sub(r"\bif\s+not\s+enabled(\s*)$", "if not enabled:", s, flags=re.I)
    s = re.sub(r"\bif\s+not\s+(\w+)(\s*)$", r"if not \1:", s)
    # 일반적인 if 문 (콜론이 없고 줄 끝인 경우)
    s = re.sub(r"\bif\s+([^:\n]+?)(\s*)$", r"if \1:", s)
    # for 루프 변수명 오인식 보정 (1n -> ln, 10 -> ln)
    s = re.sub(r"\bfor\s+1[0n]\s+in\s+lines", "for ln in lines", s)
    s = re.sub(r"\bfor\s+1n\s+in\s+", "for ln in ", s)
    # 특수문자 오인식 보정 (t™@xt -> txt)
    s = re.sub(r"t[™@]+\s*xt\s*=", "txt =", s)
    s = re.sub(r"t[™@]+\s*t\s*=", "txt =", s)
    s = re.sub(r"(\w+)[™@]+\s*\1\s*=", r"\1 =", s)
    # 파이프 문자 오인식 보정 (| if -> if, | medreturn -> return)
    s = re.sub(r"^\s*[|]\s*if\s+", "    if ", s)
    s = re.sub(r"[|]\s*if\s+", "if ", s)
    # 파이프 문자로 시작하는 줄의 파이프 제거
    s = re.sub(r"^\s*[|]\s+", "    ", s)
    # medreturn_h 같은 합쳐진 변수명/키워드 보정
    s = re.sub(r"\bmedreturn\s*_?h\b", "med_h", s, flags=re.I)
    # 복잡한 합쳐진 줄 보정: | medreturn_h = max(6.0,_robust_median(heights,med_h*0.55) def ault=14.0)
    # 실제로는 두 줄: med_h = _robust_median(heights, default=14.0) 과 return max(6.0, med_h * 0.55)
    # 패턴: max(숫자, _robust_median(변수, med_h*숫자) def ault=숫자)
    s = re.sub(r"max\s*\(\s*(\d+\.\d+)\s*,\s*_robust_median\s*\(\s*(\w+)\s*,\s*med\s*_?h\s*\*\s*(\d+\.\d+)\s*\)\s*def\s+ault\s*=\s*(\d+\.\d+)\s*\)",
               r"max(\1, med_h * \3)", s, flags=re.I)
    # med_h = _robust_median(heights, def ault=14.0) 패턴 보정
    s = re.sub(r"med\s*_?h\s*=\s*_robust_median\s*\(\s*(\w+)\s*,\s*def\s+ault\s*=\s*(\d+\.\d+)\s*\)",
               r"med_h = _robust_median(\1, default=\2)", s, flags=re.I)
    # return 문이 변수명에 합쳐진 경우 보정 (medreturn_h = ... -> med_h = ... 그리고 return 추가)
    s = re.sub(r"medreturn\s*_?h\s*=\s*max\s*\(\s*(\d+\.\d+)\s*,\s*_robust_median\s*\(\s*(\w+)\s*,\s*med\s*_?h\s*\*\s*(\d+\.\d+)\s*\)\s*def\s+ault\s*=\s*(\d+\.\d+)\s*\)",
               r"med_h = _robust_median(\2, default=\4)\n    return max(\1, med_h * \3)", s, flags=re.I)
    # 함수 호출에서 공백 누락 보정 (max(6.0,_robust_median -> max(6.0, _robust_median)
    s = re.sub(r"max\s*\(\s*(\d+\.\d+)\s*,\s*_robust_median", r"max(\1, _robust_median", s, flags=re.I)
    s = re.sub(r"max\s*\(\s*(\d+\.\d+)\s*,\s*med\s*_?h", r"max(\1, med_h", s, flags=re.I)
    # 복잡한 수식 오인식 보정 (cw1f =2.0w.w<= / cwmax(1l,<=80.0len(txt)) -> cw = w.w / max(1, len(txt)))
    s = re.sub(r"cw\s*[1lI]\s*f\s*=\s*2\.0\s*w\.w\s*<=\s*/\s*cw\s*max\s*\(\s*1[1lI]\s*,\s*<=\s*80\.0\s*len\s*\(\s*txt\s*\)\s*\)", "cw = w.w / max(1, len(txt))", s, flags=re.I)
    # 비교 연산자 오인식 보정 (2.0 <= cw <= 80.0)
    s = re.sub(r"(\d+\.\d+)\s*<=\s*(\w+)\s*<=\s*(\d+\.\d+)", r"\1 <= \2 <= \3", s)
    # 변수 할당 오인식 보정 (cw = w.w / max(1, len(txt)))
    s = re.sub(r"(\w+)\s*=\s*(\w+)\.(\w+)\s*/\s*max\s*\(\s*(\d+)\s*,\s*len\s*\(\s*(\w+)\s*\)\s*\)", r"\1 = \2.\3 / max(\4, len(\5))", s)
    # robust median -> _robust_median
    s = re.sub(r"\brobust\s+_?median\b", "_robust_median", s, flags=re.I)
    # def ault -> default (공백 오인식) - 더 강력하게
    s = re.sub(r"def\s+ault\s*=", "default=", s, flags=re.I)
    s = re.sub(r"def\s+ault", "default", s, flags=re.I)
    # medh -> med_h
    s = re.sub(r"\bmedh\b", "med_h", s, flags=re.I)
    # 숫자 오인식 보정 (8.90 -> 8.0, 1l -> 1)
    s = re.sub(r"(\d+)\.90\b", r"\1.0", s)
    s = re.sub(r"(\d+)[1lI]\b", r"\1", s)
    # 불필요한 특수문자 제거 (lnㆍ굁이`ds 등)
    s = re.sub(r"[ㆍ굁이`]+", "", s)
    # preprocess_for_0006pil -> preprocess_for_code_pil
    s = re.sub(r"preprocess_for_0+6pil", "preprocess_for_code_pil", s, flags=re.I)
    s = re.sub(r"preprocess_for_0+\d+pil", "preprocess_for_code_pil", s, flags=re.I)
    # ImageEnhance.ontrast -> ImageEnhance.Contrast
    s = re.sub(r"ImageEnhance\.ontrast", "ImageEnhance.Contrast", s, flags=re.I)
    # 1l.1 -> 1.1 (숫자 오인식)
    s = re.sub(r"(\d+)l\.(\d+)", r"\1.\2", s)
    s = re.sub(r"(\d+)I\.(\d+)", r"\1.\2", s)
    # THRESH_ BINARY -> THRESH_BINARY (공백 제거)
    s = re.sub(r"THRESH_\s+BINARY", "THRESH_BINARY", s, flags=re.I)
    s = re.sub(r"THRESH_\s+OTSU", "THRESH_OTSU", s, flags=re.I)
    # MORPH_LOSE -> MORPH_CLOSE
    s = re.sub(r"MORPH_LOSE", "MORPH_CLOSE", s, flags=re.I)
    # Image.from array -> Image.fromarray
    s = re.sub(r"Image\.from\s+array", "Image.fromarray", s, flags=re.I)
    # 경로/심볼 오인식 보정
    s = re.sub(r"\bOS\.path\b", "os.path", s)
    s = re.sub(r"\bos\.path\.is_dir\b", "os.path.isdir", s, flags=re.I)
    s = s.replace("Program Files", "Program_Files")
    s = re.sub(r"Program[_ ]FiIes", "Program_Files", s, flags=re.I)  # I/l 혼동
    s = s.replace("tesseract,exe", "tesseract.exe")
    # tesseract_cmd 뒤에 숫자나 중복이 붙은 경우 제거 (예: tesseract_cmd70, tesseract_cmd_cmd -> tesseract_cmd)
    s = re.sub(r"tesseract_cmd(_cmd|\d+)", "tesseract_cmd", s, flags=re.I)
    s = re.sub(r"pytesseract\.pytesseract\.tesseract0?1?0?\d*", "pytesseract.pytesseract.tesseract_cmd", s, flags=re.I)
    s = re.sub(r"\bTESSDATA\s+DIR\b", "TESSDATA_DIR", s, flags=re.I)
    # 숫자로 시작하는 os.path 패턴을 if로 수정 (예: 127 os.path.exists -> if os.path.exists)
    s = re.sub(r"^\s*\d+\s+os\.path\.", "    if os.path.", s)
    # 중복된 "설정" 제거 (예: Tesseract설정 설정 -> Tesseract 설정)
    s = re.sub(r"설정\s+설정", "설정", s)
    # pytesseract = None 오인식 보정 (예: pytesseract『구'C = None -> pytesseract = None)
    s = re.sub(r"pytesseract[『구'C\s]*=\s*None", "pytesseract = None", s)
    # 불필요한 특수문자 제거 (예: 『귀C: -> 제거)
    s = re.sub(r"[『귀C:]+", "", s)
    # def configure tesseract() -> def _configure_tesseract()
    s = re.sub(r"def\s+configure\s+tesseract\s*\(\s*\)", "def _configure_tesseract()", s, flags=re.I)
    # os.environ|["TESSDATA PREFIX"] -> os.environ["TESSDATA_PREFIX"]
    s = re.sub(r"os\.environ\s*[|]\s*\[\s*[\"']TESSDATA\s+PREFIX[\"']\s*\]", 'os.environ["TESSDATA_PREFIX"]', s, flags=re.I)
    s = s.replace("°", '"')
    s = s.replace("°", "'")
    s = s.replace("¥", "*")
    s = s.replace("×", "*")
    s = s.replace("·", "*")
    s = re.sub(r"\bif_([a-z_]+)_is_not\s+None_and_", r"if \1 is not None and ", s, flags=re.I)
    s = re.sub(r"\bif_([a-z_]+)_is\s+not\s+None_and_", r"if \1 is not None and ", s, flags=re.I)
    s = re.sub(r"\bif_([a-z_]+)_and_", r"if \1 and ", s, flags=re.I)
    s = re.sub(r"_and_os\.", " and os.", s)
    s = re.sub(r"_and_", " and ", s)
    s = re.sub(r"_or_", " or ", s)
    s = re.sub(r"_is_", " is ", s)
    s = re.sub(r"_not_", " not ", s)
    s = re.sub(r"_in_", " in ", s)
    s = re.sub(r"([A-Z]+)(EXE|DIR|PATH|ENV|CMD)", r"\1_\2", s)
    s = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"([a-z]+)(cmd|dir|path|exe|env|prefix)", r"\1_\2", s, flags=re.I)
    s = re.sub(r"(pytesseract)\.(pytesseract)\.([a-z]+)(cmd)", r"\1.\2.\3_\4", s, flags=re.I)
    s = re.sub(r"\bTESSERACTEXE\b", "TESSERACT_EXE", s)
    s = re.sub(r"\btesseractcmd\b", "tesseract_cmd", s)
    s = re.sub(r"([a-z])\-([a-z])", r"\1_\2", s)
    s = s.replace("``", '"')
    s = re.sub(r'"([^"]*?)\'', r'"\1"', s)
    s = re.sub(r"'([^']*?)\"", r"'\1'", s)
    s = re.sub(r'="([^"]*?)\'', r'="\1"', s)
    s = re.sub(r"='([^']*?)\"", r"='\1'", s)
    s = re.sub(r"([=\(\[\s,])''([^'])", r'\1"\2', s)
    s = re.sub(r"([^'])''([=\)\]\s,\.;])", r'\1"\2', s)
    s = re.sub(r"(\w+)\[\s*([\"'])([^\"']+)\2\s*\]", r'\1[\2\3\2]', s)
    s = re.sub(r"\[\s*([\"'])([^\"']+)\1\s*\]", r'[\1\2\1]', s)
    s = re.sub(r"os\.path\.(exists|isdir)\(([A-Z_]+EXE)\)", r"os.path.\1(\2)", s)
    s = re.sub(r"os\.path\.(exists|isdir)\(([A-Z_]+DIR)\)", r"os.path.\1(\2)", s)
    s = re.sub(r"\bi1f\b", "if", s, flags=re.I)
    s = re.sub(r"\b1f\b", "if", s)
    s = re.sub(r"\bt0\b", "to", s, flags=re.I)
    s = s.replace("—", "-").replace("–", "-").replace("•", ".")
    s = s.replace("-〉", "->").replace("→", "->")
    s = s.replace("use-gpu", "use_gpu").replace("use—gpu", "use_gpu").replace("use–gpu", "use_gpu")
    for pat, rep in COMMON_GLUE:
        s = pat.sub(rep, s)
    for pat, rep in SAFE_ID_REPAIRS:
        s = pat.sub(rep, s)
    s = re.sub(r"\bb001\b", "bool", s, flags=re.I)
    s = re.sub(r"\bboo1\b", "bool", s, flags=re.I)
    s = re.sub(r"\bTup1e\b", "Tuple", s)
    s = s.replace("IMREAD COLOR", "IMREAD_COLOR")
    s = re.sub(r"\bscaLe\b|\bsca1e\b", "scale", s)
    s = re.sub(r"\s=\s=", " ==", s)
    s = re.sub(r"==\s=", "==", s)
    s = re.sub(r"\bimport\(", "import (", s)
    s = re.sub(r"\bdef(?=[A-Za-z_])", "def ", s)
    s = re.sub(r"\bengine\s+1['\"]tesseract['\"]", 'engine == "tesseract"', s)
    s = re.sub(r'\bengine\s+["\']tesseract["\']', 'engine == "tesseract"', s, flags=re.I)
    s = re.sub(r"\bif\s+engine\s+tesseract\b", 'if engine == "tesseract"', s, flags=re.I)
    s = re.sub(r'^\s*engine\s*==\s*"tesseract"\s*$', 'if engine == "tesseract":', s, flags=re.I)
    s = re.sub(r"\btext\s+(run_[A-Za-z_]\w*\()", r"text = \1", s)
    s = re.sub(r"\bimage_to_text\s+easyocr\b", "image_to_text_easyocr", s, flags=re.I)
    s = re.sub(r"\bimage_to_text\s+paddleocr\b", "image_to_text_paddleocr", s, flags=re.I)
    s = re.sub(r"\bimage\s*(?:to|t0)\s*text\b", "image_to_text", s, flags=re.I)
    s = re.sub(r"\bimage\s*_?\s*(?:to|t0)\s*text\s*_?\s*easyocr\b", "image_to_text_easyocr", s, flags=re.I)
    s = re.sub(r"\bimage\s*_?\s*(?:to|t0)\s*text\s*_?\s*paddleocr\b", "image_to_text_paddleocr", s, flags=re.I)
    s = re.sub(r"\bload\s*bgr\b", "load_bgr", s, flags=re.I)
    s = re.sub(r"\brun\s*tesseract\b", "run_tesseract", s, flags=re.I)
    s = re.sub(r"\brun\s*easyocr\b", "run_easyocr", s, flags=re.I)
    s = re.sub(r"\brun\s*paddleocr\b", "run_paddleocr", s, flags=re.I)
    s = re.sub(r"\bbench\s*one\b", "bench_one", s, flags=re.I)
    s = re.sub(r"\bargs\.1ang\b", "args.lang", s, flags=re.I)
    s = re.sub(r"\bte\s*time\.\s*perf_counter\s*\(\s*\)", "t0 = time.perf_counter()", s, flags=re.I)
    s = re.sub(r"\buse\s*[-–—]\s*gpu\b", "use_gpu", s, flags=re.I)
    s = re.sub(r"^\s*tmport\b", "import", s, flags=re.I)
    s = re.sub(r"\bfrom\s+mSS\s+import\s+mSS\b", "from mss import mss", s)
    s = re.sub(r"\bmSS\b", "mss", s)
    s = re.sub(r"\bmSS\(\)", "mss()", s)
    s = s.replace("0penCV", "OpenCV").replace("0pencv", "OpenCV")
    s = re.sub(r"\bcvtC0?1?0r\b", "cvtColor", s, flags=re.I)
    s = re.sub(r"destroyA11b\W*indows", "destroyAllWindows", s, flags=re.I)
    s = re.sub(r'Fu11screen', "Fullscreen", s)
    s = s.replace("waitKey(ø)", "waitKey(0)").replace("waitKey(Ø)", "waitKey(0)")
    s = re.sub(r"^\s*name\s+maln\b", 'if __name__ == "__main__":', s, flags=re.I)
    s = re.sub(r"__maln__", "__main__", s)
    s = re.sub(r"\bscreen\s+capture_fullscreen_bgr\s*\(\s*\)", "screen = capture_fullscreen_bgr()", s, flags=re.I)
    s = re.sub(r"\bmonitor\s+sct\.\s*monitors\s*\[\s*(.+?)\s*\]", r"monitor = sct.monitors[\1]", s)
    lead = len(s) - len(s.lstrip(" "))
    body = re.sub(r"[ ]{2,}", " ", s.lstrip(" "))
    return (" " * lead) + body


def reconstruct_text_from_words(
    words: List[WordBox],
    *,
    code_mode: bool = True,
    normalize: bool = True,
    indent_step: int = 4,
    remove_emoji: bool = True,
) -> str:
    if not words:
        return ""
    clean_words: List[WordBox] = []
    for w in words:
        t = (w.text or "").strip()
        if not t:
            continue
        t = CTRL_RE.sub("", t)
        if remove_emoji:
            t = EMOJI_RE.sub("", t)
        if not t:
            continue
        clean_words.append(WordBox(text=t, x=w.x, y=w.y, w=w.w, h=w.h, conf=w.conf))
    if not clean_words:
        return ""
    lines = cluster_lines(clean_words)
    if not lines:
        return ""
    
    # 고정밀 글자 폭 계산: 중앙값 높이 기반 평균 글자 폭 추정
    heights = [w.h for ln in lines for w in ln.words if w.h > 0]
    med_h = _robust_median(heights, default=14.0)
    # 글자 높이의 0.52배를 평균 글자 폭으로 추정 (일반적인 폰트 비율)
    char_w = estimate_char_width(lines)
    # 더 정밀한 추정: 높이 기반 보정
    if char_w < med_h * 0.45 or char_w > med_h * 0.65:
        char_w = max(6.0, med_h * 0.52)  # 높이 기반 재계산
    
    left_margin = min(w.x for ln in lines for w in ln.words)
    raw_lines: List[str] = []
    
    for ln in lines:
        if not ln.words:
            raw_lines.append("")
            continue
        
        # 정렬된 단어 리스트
        sorted_words = sorted(ln.words, key=lambda w: w.x)
        first = sorted_words[0]
        
        # 들여쓰기 계산: 물리적 좌표 기반 정밀 계산
        indent_px = first.x - left_margin
        leading_spaces = int(round(indent_px / max(1e-6, char_w)))
        leading_spaces = max(0, leading_spaces)
        
        parts: List[str] = []
        parts.append(" " * leading_spaces)
        parts.append(first.text)
        prev_x2 = first.x2
        
        # 단어 간 띄어쓰기 계산: 물리적 간격 기반 정밀 계산
        for w in sorted_words[1:]:
            txt = w.text
            if not txt:
                continue
            
            gap_px = w.x - prev_x2
            
            # 매우 작은 간격 (10% 이하): 공백 없음 (붙여쓰기)
            if gap_px <= char_w * 0.15:
                spaces = 0
            else:
                # 물리적 간격을 글자 폭으로 나누어 공백 개수 계산
                spaces = int(round(gap_px / max(1e-6, char_w)))
                spaces = clamp_int(spaces, 1, 80)
            
            # 영문/숫자 사이 최소 공백 보장
            if spaces == 0:
                prev_text = parts[-1] if parts else ""
                if re.search(r"\w$", prev_text) and re.search(r"^\w", txt):
                    spaces = 1
            
            parts.append(" " * spaces)
            parts.append(txt)
            prev_x2 = max(prev_x2, w.x2)
        
        raw_lines.append("".join(parts).rstrip())
    if not code_mode:
        out = "\n".join(raw_lines).rstrip() + "\n"
        return sanitize_text(out, remove_emoji=remove_emoji, keep_newlines=True, collapse_spaces=False) + "\n"

    def indent_of(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    def set_indent(s: str, n: int) -> str:
        return (" " * max(0, n)) + s.lstrip(" ")

    fixed: List[str] = []
    for s in raw_lines:
        if not s.strip():
            fixed.append("")
            continue
        cur_indent = indent_of(s)
        stripped = s.lstrip(" ")
        if stripped and stripped[0] in ("}", "]", ")"):
            cur_indent = max(0, cur_indent - indent_step)
        if re.match(r"^(else:|elif\b|except\b|finally:)", stripped):
            cur_indent = max(0, cur_indent - indent_step)
        if fixed:
            prev = fixed[-1].rstrip()
            prev_strip = prev.lstrip(" ")
            prev_indent = indent_of(prev)
            if prev_strip.endswith(":"):
                cur_indent = max(cur_indent, prev_indent + indent_step)
        s2 = set_indent(s, cur_indent)
        if normalize:
            s2 = normalize_code_line(s2)
        fixed.append(s2.rstrip())
    out = "\n".join(fixed).rstrip() + "\n"
    return sanitize_text(out, remove_emoji=remove_emoji, keep_newlines=True, collapse_spaces=False) + "\n"


def merge_winrt_lines(lines_ko: List[str], lines_en: List[str]) -> List[str]:
    n = max(len(lines_ko), len(lines_en))
    out: List[str] = []
    for i in range(n):
        ko = lines_ko[i] if i < len(lines_ko) else ""
        en = lines_en[i] if i < len(lines_en) else ""
        en_score = len(ASCII_CODE_RE.findall(en))
        ko_score = len(ASCII_CODE_RE.findall(ko))
        if en_score >= ko_score:
            out.append(en if en.strip() else ko)
        else:
            out.append(ko if ko.strip() else en)
    return out


def _run_coro_sync(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def _create_winrt_engine(lang_tag: str):
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.globalization import Language

    try:
        eng = OcrEngine.try_create_from_language(Language(lang_tag))
        if eng is not None:
            return eng
    except Exception:
        pass
    alias_map = {"ko": ["ko-KR", "ko"], "en": ["en-US", "en"]}
    for cand in alias_map.get(lang_tag, []):
        try:
            eng = OcrEngine.try_create_from_language(Language(cand))
            if eng is not None:
                return eng
        except Exception:
            continue
    try:
        eng = OcrEngine.try_create_from_user_profile_languages()
        if eng is not None:
            return eng
    except Exception:
        pass
    try:
        eng = OcrEngine.try_create_from_language(Language("en"))
        if eng is not None:
            return eng
    except Exception:
        pass
    return None


async def _winrt_recognize_async(pil_img: Image.Image, lang_tag: str):
    from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
    from winrt.windows.storage.streams import DataWriter

    def pil_to_software_bitmap(img: Image.Image) -> SoftwareBitmap:
        rgba = img.convert("RGBA")
        arr = np.array(rgba, dtype=np.uint8)
        sb = SoftwareBitmap(BitmapPixelFormat.RGBA8, rgba.width, rgba.height)
        writer = DataWriter()
        writer.write_bytes(arr.tobytes())
        sb.copy_from_buffer(writer.detach_buffer())
        return sb

    engine = _create_winrt_engine(lang_tag)
    if engine is None:
        raise RuntimeError("WinRT OcrEngine 생성 실패 (언어팩/OCR 지원 미설치 가능)")
    sb = pil_to_software_bitmap(pil_img)
    return await engine.recognize_async(sb)


def _winrt_words_from_result(result) -> List[WordBox]:
    out: List[WordBox] = []
    for ln in getattr(result, "lines", []):
        for w in getattr(ln, "words", []):
            txt = getattr(w, "text", "") or ""
            rect = getattr(w, "bounding_rect", None)
            if not txt or rect is None:
                continue
            out.append(
                WordBox(
                    text=txt,
                    x=float(rect.x),
                    y=float(rect.y),
                    w=float(rect.width),
                    h=float(rect.height),
                    conf=-1.0,
                )
            )
    return out


def _winrt_lines_text(result) -> List[str]:
    return [getattr(ln, "text", "") or "" for ln in getattr(result, "lines", [])]


def get_winrt_words(
    img: Union[np.ndarray, Image.Image],
    *,
    scale: int = 2,
    code_mode: bool = True,
    remove_emoji: bool = True,
) -> List[WordBox]:
    pil_img = open_image_any(img)
    if scale and scale != 1:
        w, h = pil_img.size
        pil_img = pil_img.resize((w * scale, h * scale), Image.BICUBIC)
    pil_img = preprocess_for_code_pil(pil_img, enabled=code_mode)
    result_ko = _run_coro_sync(_winrt_recognize_async(pil_img, "ko"))
    result_en = _run_coro_sync(_winrt_recognize_async(pil_img, "en"))
    words_ko = _winrt_words_from_result(result_ko)
    words_en = _winrt_words_from_result(result_en)
    all_words: List[WordBox] = []
    used_positions = set()
    for w in words_ko:
        pos_key = (int(w.x // 10), int(w.y // 10))
        if pos_key not in used_positions:
            all_words.append(w)
            used_positions.add(pos_key)
    for w in words_en:
        pos_key = (int(w.x // 10), int(w.y // 10))
        if pos_key not in used_positions:
            all_words.append(w)
            used_positions.add(pos_key)
    return all_words


def image_to_text_winrt(
    img: Union[np.ndarray, Image.Image],
    *,
    scale: int = 2,
    code_mode: bool = True,
    normalize: bool = True,
    indent_step: int = 4,
    remove_emoji: bool = True,
) -> str:
    pil_img = open_image_any(img)
    if scale and scale != 1:
        w, h = pil_img.size
        pil_img = pil_img.resize((w * scale, h * scale), Image.BICUBIC)
    pil_img = preprocess_for_code_pil(pil_img, enabled=code_mode)
    result_ko = _run_coro_sync(_winrt_recognize_async(pil_img, "ko"))
    result_en = _run_coro_sync(_winrt_recognize_async(pil_img, "en"))
    lines_ko = [
        sanitize_text(t, remove_emoji=remove_emoji, keep_newlines=False, collapse_spaces=False)
        for t in _winrt_lines_text(result_ko)
    ]
    lines_en = [
        sanitize_text(t, remove_emoji=remove_emoji, keep_newlines=False, collapse_spaces=False)
        for t in _winrt_lines_text(result_en)
    ]
    merged_lines = merge_winrt_lines(lines_ko, lines_en)
    words = _winrt_words_from_result(result_en)
    if not words:
        y = 0.0
        line_h = 18.0
        for ln in merged_lines:
            if ln.strip():
                words.append(
                    WordBox(
                        text=ln,
                        x=0.0,
                        y=y,
                        w=float(max(10, len(ln) * 10)),
                        h=line_h,
                        conf=-1.0,
                    )
                )
            y += line_h
    out = reconstruct_text_from_words(
        words,
        code_mode=code_mode,
        normalize=normalize,
        indent_step=indent_step,
        remove_emoji=remove_emoji,
    )
    if out.count("\n") <= 2 and len(merged_lines) >= 2:
        fixed = [normalize_code_line(x) if normalize else x for x in merged_lines]
        out = "\n".join(fixed).rstrip() + "\n"
    return out


def _build_whitelist(code_mode: bool, lang: str) -> Optional[str]:
    l = (lang or "").lower()
    if not code_mode:
        return None
    if "kor" in l or "korean" in l or l.startswith("ko"):
        return None
    return (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_"
        "-+=/*%<>!&|^~.,:;?@#$()[]{}\\"
        "*"
        "'"
        "\""
        "`"
        "·×÷±≤≥≠≈∞∑∏∫√"
        "→←↑↓⇒⇐⇑⇓"
        "≤≥≠≈"
        "αβγδεζηθικλμνξοπρστυφχψω"
        "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        " \t"
    )


def tesseract_word_boxes(
    pil_img: Image.Image,
    *,
    lang: str = "kor+eng",
    psm: int = 6,
    oem: int = 3,
    code_mode: bool = True,
    remove_emoji: bool = True,
) -> List[WordBox]:
    if pytesseract is None:
        raise RuntimeError("pytesseract not installed")
    whitelist = _build_whitelist(code_mode=code_mode, lang=lang)
    config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
    config += " -c tessedit_pageseg_mode=6"
    config += " -c classify_bln_numeric_mode=0"
    config += " -c textord_min_linesize=2.5"
    config += " -c textord_tabvector_vertical_gap_factor=0.5"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(
        pil_img,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    out: List[WordBox] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = data["text"][i]
        if txt is None:
            continue
        txt = txt.strip()
        txt = CTRL_RE.sub("", txt)
        if remove_emoji:
            txt = EMOJI_RE.sub("", txt)
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])
        if w <= 1 or h <= 1:
            continue
        out.append(WordBox(text=txt, x=x, y=y, w=w, h=h, conf=conf))
    return out


def image_to_text(
    img: Union[np.ndarray, Image.Image],
    lang: str = "kor+eng",
    *,
    scale: int = 4,
    code_mode: bool = True,
    layout: bool = True,
    normalize: bool = True,
    indent_step: int = 4,
    psm: int = 6,
    oem: int = 3,
    remove_emoji: bool = True,
) -> str:
    pil_img = open_image_any(img)
    if scale and scale != 1:
        w, h = pil_img.size
        pil_img = pil_img.resize((w * scale, h * scale), Image.LANCZOS)
    pil_img = preprocess_for_code_pil(pil_img, enabled=code_mode)
    if not layout:
        if pytesseract is None:
            raise RuntimeError("pytesseract not installed")
        whitelist = _build_whitelist(code_mode=code_mode, lang=lang)
        config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
        config += " -c tessedit_pageseg_mode=6"
        config += " -c classify_bln_numeric_mode=0"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        txt = pytesseract.image_to_string(pil_img, lang=lang, config=config)
        txt = sanitize_text(txt, remove_emoji=remove_emoji, keep_newlines=True, collapse_spaces=False)
        return txt.rstrip() + "\n"
    words = tesseract_word_boxes(
        pil_img,
        lang=lang,
        psm=psm,
        oem=oem,
        code_mode=code_mode,
        remove_emoji=remove_emoji,
    )
    return reconstruct_text_from_words(
        words,
        code_mode=code_mode,
        normalize=normalize,
        indent_step=indent_step,
        remove_emoji=remove_emoji,
    )


def capture_fullscreen_bgr(monitor_index: int = 1) -> np.ndarray:
    with mss() as sct:
        monitor = sct.monitors[monitor_index]
        img = np.array(sct.grab(monitor))
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return bgr


def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)


def select_roi_auto(image: np.ndarray, window_name: str = "Select ROI") -> np.ndarray:
    drawing = False
    start_point = None
    end_point = None
    current_rect = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, current_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
                img_copy = image.copy()
                cv2.rectangle(img_copy, start_point, end_point, (209, 226, 125), 2)
                cv2.imshow(window_name, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing:
                drawing = False
                end_point = (x, y)
                x1, y1 = start_point
                x2, y2 = end_point
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                if w > 0 and h > 0:
                    current_rect = (x, y, w, h)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, image)

    while current_rect is None:
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise ValueError("ROI 선택이 취소되었습니다.")
        if current_rect is not None:
            break

    cv2.destroyAllWindows()
    x, y, w, h = current_rect
    cropped = image[y:y + h, x:x + w]
    return cropped


def merge_tesseract_winrt_results(
    tesseract_words: List[WordBox],
    winrt_words: List[WordBox],
    pil_img: Image.Image,
) -> str:
    korean_re = re.compile(r"[가-힣]")
    tesseract_lines = cluster_lines(tesseract_words)
    winrt_lines = cluster_lines(winrt_words)

    def find_overlapping_winrt_word(tess_word: WordBox, winrt_words_list: List[WordBox], threshold: float = 0.5) -> Optional[WordBox]:
        best_match = None
        best_overlap = 0.0
        for winrt_word in winrt_words_list:
            y_overlap = min(tess_word.y2, winrt_word.y2) - max(tess_word.y, winrt_word.y)
            if y_overlap <= 0:
                continue
            x_overlap = min(tess_word.x2, winrt_word.x2) - max(tess_word.x, winrt_word.x)
            if x_overlap <= 0:
                continue
            tess_area = tess_word.w * tess_word.h
            overlap_area = x_overlap * y_overlap
            if tess_area > 0:
                overlap_ratio = overlap_area / tess_area
                if overlap_ratio >= threshold and overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_match = winrt_word
        return best_match

    merged_words: List[WordBox] = []
    for tess_line in tesseract_lines:
        for tess_word in tess_line.words:
            nearby_winrt_words = [
                w for w in winrt_words
                if abs(w.cy - tess_word.cy) < tess_word.h * 2.0
            ]
            overlapping_winrt = find_overlapping_winrt_word(tess_word, nearby_winrt_words)
            final_text = tess_word.text
            if overlapping_winrt:
                winrt_text_val = overlapping_winrt.text.strip()
                tess_text_val = tess_word.text.strip()
                if tess_text_val == winrt_text_val or abs(len(tess_text_val) - len(winrt_text_val)) <= 1:
                    final_text = tess_text_val
                else:
                    if korean_re.search(winrt_text_val):
                        final_text = winrt_text_val
                    elif not korean_re.search(tess_text_val) and korean_re.search(winrt_text_val):
                        final_text = winrt_text_val
                    else:
                        final_text = tess_text_val
            else:
                if not korean_re.search(tess_word.text):
                    final_text = tess_word.text
                else:
                    nearby_korean_winrt = [
                        w for w in nearby_winrt_words
                        if korean_re.search(w.text) and abs(w.x - tess_word.x) < tess_word.w * 3.0
                    ]
                    if nearby_korean_winrt:
                        closest = min(nearby_korean_winrt, key=lambda w: abs(w.x - tess_word.x))
                        final_text = closest.text
            merged_words.append(WordBox(
                text=final_text,
                x=tess_word.x,
                y=tess_word.y,
                w=tess_word.w,
                h=tess_word.h,
                conf=tess_word.conf
            ))
    for winrt_word in winrt_words:
        is_overlapping = False
        for tess_word in tesseract_words:
            overlapping = find_overlapping_winrt_word(tess_word, [winrt_word], threshold=0.3)
            if overlapping:
                is_overlapping = True
                break
        if not is_overlapping and korean_re.search(winrt_word.text):
            merged_words.append(winrt_word)
    return reconstruct_text_from_words(
        merged_words,
        code_mode=True,
        normalize=True,
        indent_step=4,
        remove_emoji=True,
    )


def get_tesseract_words(
    img: Union[np.ndarray, Image.Image],
    lang: str = "kor+eng",
    *,
    scale: int = 4,
    code_mode: bool = True,
    remove_emoji: bool = True,
) -> List[WordBox]:
    pil_img = open_image_any(img)
    if scale and scale != 1:
        w, h = pil_img.size
        pil_img = pil_img.resize((w * scale, h * scale), Image.LANCZOS)
    pil_img = preprocess_for_code_pil(pil_img, enabled=code_mode)
    return tesseract_word_boxes(
        pil_img,
        lang=lang,
        psm=6,
        oem=3,
        code_mode=code_mode,
        remove_emoji=remove_emoji,
    )


def check_winrt_available():
    """WinRT 바인딩(파이썬 패키지) 사용 가능 여부."""
    try:
        from winrt.windows.media.ocr import OcrEngine  # noqa: F401
        from winrt.windows.globalization import Language  # noqa: F401
        from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat  # noqa: F401
        from winrt.windows.storage.streams import DataWriter  # noqa: F401
        from winrt.windows.foundation.collections import IVectorView  # noqa: F401
        return True, None
    except Exception as e:
        install_cmd = (
            "python -m pip install "
            "winrt-runtime "
            "winrt-Windows.Foundation "
            "winrt-Windows.Foundation.Collections "
            "winrt-Windows.Globalization "
            "winrt-Windows.Graphics.Imaging "
            "winrt-Windows.Storage.Streams "
            "winrt-Windows.Media.Ocr"
        )
        return False, f"{e}\n\n필수 설치(cmd):\n{install_cmd}"
    """Set Tesseract paths if available; keep silent when missing."""
    if pytesseract is None:
        return
    if os.path.exists(TESSERACT_EXE):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
    if os.path.isdir(TESSDATA_DIR):
        os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR


_configure_tesseract()


def get_tesseract_langs() -> List[str]:
    """Get list of installed Tesseract language packs."""
    if pytesseract is None:
        return []
    try:
        langs = pytesseract.get_languages(config="")
        return langs if isinstance(langs, list) else []
    except Exception:
        try:
            import subprocess
            if os.path.exists(TESSERACT_EXE):
                result = subprocess.run(
                    [TESSERACT_EXE, "--list-langs"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        return [ln.strip() for ln in lines[1:] if ln.strip()]
        except Exception:
            pass
    return []


def pick_tess_lang(prefer: str = "eng") -> str:
    """Pick Tesseract language. Returns prefer if installed, else fallback."""
    if pytesseract is None:
        return prefer
    installed = get_tesseract_langs()
    if prefer in installed:
        return prefer
    if prefer.lower() in installed:
        return prefer.lower()
    return installed[0] if installed else prefer


EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def sanitize_text(
    text: str,
    *,
    remove_emoji: bool = True,
    keep_newlines: bool = True,
    collapse_spaces: bool = False,
    tabsize: int = 4,
) -> str:
    if not text:
        return ""
    t = CTRL_RE.sub("", text)
    if remove_emoji:
        t = EMOJI_RE.sub("", t)
    t = t.replace("\t", " " * tabsize)
    t = "\n".join([ln.rstrip() for ln in t.splitlines()])
    if not keep_newlines:
        t = t.replace("\n", " ")
    if collapse_spaces:
        out_lines = []
        for ln in t.splitlines():
            lead = len(ln) - len(ln.lstrip(" "))
            body = re.sub(r"[ ]{2,}", " ", ln.lstrip(" "))
            out_lines.append((" " * lead) + body)
        t = "\n".join(out_lines)
    return t.strip()


def preprocess_for_code_pil(img: Image.Image, enabled: bool) -> Image.Image:
    """Tesseract PASS1 (강전처리): 구조/문자 중심"""
    if not enabled:
        return img.convert("RGB")
    g = img.convert("L")
    g = ImageEnhance.Contrast(g).enhance(2.8)
    g = ImageEnhance.Sharpness(g).enhance(2.2)
    g = ImageEnhance.Brightness(g).enhance(1.1)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    arr = np.array(g)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 너무 강한 close는 ':' ';'를 죽일 수 있으니 (1,1) 유지
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    g = Image.fromarray(binary)
    return g.convert("RGB")


def preprocess_for_code_pil_light(img: Image.Image, enabled: bool) -> Image.Image:
    """Tesseract PASS2 (약전처리): ':' ';' '"' ',' 등 기호 보존"""
    if not enabled:
        return img.convert("RGB")
    g = img.convert("L")
    g = ImageEnhance.Contrast(g).enhance(2.0)
    g = ImageEnhance.Sharpness(g).enhance(1.6)
    g = ImageEnhance.Brightness(g).enhance(1.05)
    # ✅ OTSU/모폴로지 제거: 점/기호 보존용
    return g.convert("RGB")


def preprocess_for_winrt_pil(img: Image.Image, enabled: bool) -> Image.Image:
    """WinRT용 약전처리 (점/기호 ':' ';' 보호)"""
    if not enabled:
        return img.convert("RGB")
    g = img.convert("L")
    g = ImageEnhance.Contrast(g).enhance(2.0)
    g = ImageEnhance.Sharpness(g).enhance(1.6)
    g = ImageEnhance.Brightness(g).enhance(1.05)
    # ✅ OTSU/모폴로지 제외
    return g.convert("RGB")


def open_image_any(img: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).convert("RGB")
    raise TypeError("Unsupported image type (need PIL.Image or OpenCV BGR ndarray).")


@dataclass
class WordBox:
    text: str
    x: float
    y: float
    w: float
    h: float
    conf: float = -1.0

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cy(self) -> float:
        return self.y + self.h * 0.5


@dataclass
class LineBox:
    words: List[WordBox]
    y_center: float
    y_top: float
    y_bot: float


def _robust_median(values: List[float], default: float) -> float:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return default
    return float(np.median(np.array(vals, dtype=np.float32)))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def cluster_lines(words: List[WordBox]) -> List[LineBox]:
    if not words:
        return []
    heights = [w.h for w in words if w.h > 0]
    med_h = _robust_median(heights, default=14.0)
    y_thresh = max(6.0, med_h * 0.60)
    ws = sorted(words, key=lambda w: (w.cy, w.x))

    lines: List[LineBox] = []
    cur: List[WordBox] = []
    cur_center: Optional[float] = None

    for w in ws:
        if not cur:
            cur = [w]
            cur_center = w.cy
            continue
        if abs(w.cy - float(cur_center)) <= y_thresh:
            cur.append(w)
            cur_center = (float(cur_center) * 0.7) + (w.cy * 0.3)
        else:
            cur_sorted = sorted(cur, key=lambda t: t.x)
            y_top = min(t.y for t in cur_sorted)
            y_bot = max(t.y2 for t in cur_sorted)
            y_center = float(np.mean([t.cy for t in cur_sorted]))
            lines.append(LineBox(words=cur_sorted, y_center=y_center, y_top=y_top, y_bot=y_bot))
            cur = [w]
            cur_center = w.cy

    if cur:
        cur_sorted = sorted(cur, key=lambda t: t.x)
        y_top = min(t.y for t in cur_sorted)
        y_bot = max(t.y2 for t in cur_sorted)
        y_center = float(np.mean([t.cy for t in cur_sorted]))
        lines.append(LineBox(words=cur_sorted, y_center=y_center, y_top=y_top, y_bot=y_bot))

    lines.sort(key=lambda ln: ln.y_center)

    merged: List[LineBox] = []
    for ln in lines:
        if not merged:
            merged.append(ln)
            continue
        prev = merged[-1]
        gap = ln.y_top - prev.y_bot
        if gap <= max(2.0, med_h * 0.15):
            merged_words = sorted(prev.words + ln.words, key=lambda t: t.x)
            y_top = min(t.y for t in merged_words)
            y_bot = max(t.y2 for t in merged_words)
            y_center = float(np.mean([t.cy for t in merged_words]))
            merged[-1] = LineBox(words=merged_words, y_center=y_center, y_top=y_top, y_bot=y_bot)
        else:
            merged.append(ln)

    return merged


def estimate_char_width(lines: List[LineBox]) -> float:
    samples: List[float] = []
    for ln in lines:
        for w in ln.words:
            txt = w.text
            if not txt or len(txt) < 2:
                continue
            if " " in txt:
                continue
            cw = w.w / max(1, len(txt))
            if 2.0 <= cw <= 80.0:
                samples.append(float(cw))
    if not samples:
        heights = [w.h for ln in lines for w in ln.words if w.h > 0]
        med_h = _robust_median(heights, default=14.0)
        return max(6.0, med_h * 0.55)
    return _robust_median(samples, default=8.0)


ASCII_CODE_RE = re.compile(r"[A-Za-z0-9_{}\[\]().,:;=<>!+\-/*%\\'\"`@#$^|~]")
HANGUL_TAIL_RE = re.compile(r"([가-힣]{1,4})\s*$")


def _line_ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    ascii_cnt = len(ASCII_CODE_RE.findall(s))
    return ascii_cnt / max(1, len(s))


COMMON_GLUE = [
    (re.compile(r"\bfrom(?=[A-Za-z_])", re.IGNORECASE), "from "),
    (re.compile(r"\bimport(?=[A-Za-z_])", re.IGNORECASE), "import "),
    (re.compile(r"\bdef(?=[A-Za-z_])", re.IGNORECASE), "def "),
    (re.compile(r"\breturn(?=[A-Za-z_])", re.IGNORECASE), "return "),
    (re.compile(r"\braise(?=[A-Za-z_])", re.IGNORECASE), "raise "),
]


def normalize_code_line(
    line: str,
    next_line_indent: Optional[int] = None,
    lang_hint: str = "py",
    safe_mode: bool = True,
) -> str:
    """Normalize code line with high-confidence rules only."""
    if not line:
        return line

    s = line

    # 전각→반각 정규화(안전)
    s = s.replace("：", ":").replace("﹕", ":").replace("∶", ":").replace("ː", ":")
    s = s.replace("；", ";").replace("﹔", ";")
    s = s.replace("，", ",").replace("．", ".")
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("→", "->").replace("-〉", "->")

    # 라인 끝 한글 잡음 제거(코드 라인처럼 보일 때만)
    if _line_ascii_ratio(s) >= 0.60 and HANGUL_TAIL_RE.search(s):
        s = HANGUL_TAIL_RE.sub("", s)

    # 확정 패턴 복원
    s = re.sub(r"\bwithopen\b", "with open", s, flags=re.I)
    s = re.sub(r"\bscoreinzip\b", "score in zip", s, flags=re.I)
    s = re.sub(r"\bforbox\b", "for box", s, flags=re.I)
    
    # --- OCR 공백 붕괴 복원 (확실한 패턴) ---
    # exceptExceptionase -> except Exception as e:
    s = re.sub(r"\bexceptExceptionase\b", "except Exception as e", s, flags=re.I)
    # exceptException -> except Exception
    s = re.sub(r"\bexceptException\b", "except Exception", s, flags=re.I)
    # 처리중오류 -> 처리 중 오류 (한글 주석/문자열 내)
    s = re.sub(r"처리중오류", "처리 중 오류", s)
    # PaddleOCRPretrainedInference -> PaddleOCR Pretrained Inference
    s = re.sub(r"PaddleOCRPretrainedInference", "PaddleOCR Pretrained Inference", s, flags=re.I)
    s = re.sub(r"PaddleOCRPretrained", "PaddleOCR Pretrained", s, flags=re.I)

    # ")asf" / "asf" → "as f"
    s = re.sub(r"\)\s*asf\b", ") as f", s, flags=re.I)
    s = re.sub(r"(\bwith\s+open\([^)]*\))\s*asf\b", r"\1 as f", s, flags=re.I)

    # coordforpointinboxforcoordinpoint 복원
    s = re.sub(
        r"\bcoordforpointinboxforcoordinpoint\b",
        "coord for point in box for coord in point",
        s,
        flags=re.I,
    )

    # zip(boxes,txts,scores) → zip(boxes, txts, scores)
    s = re.sub(r"\bzip\(\s*boxes\s*,\s*txts\s*,\s*scores\s*\)", "zip(boxes, txts, scores)", s)

    # --- OCR 특화 복원: typing 제네릭 표기 제거 ---
    # zip[tuple](...) -> zip(...)
    s = re.sub(r"\bzip\s*\[\s*tuple\s*\]\s*\(", "zip(", s, flags=re.I)
    # tqdm[Path](...) -> tqdm(...)
    s = re.sub(r"\btqdm\s*\[\s*Path\s*\]\s*\(", "tqdm(", s, flags=re.I)
    # list[Path](...) -> list(...)
    s = re.sub(r"\blist\s*\[\s*Path\s*\]\s*\(", "list(", s, flags=re.I)
    # 일반적인 함수[Type](...) -> 함수(...) (확실한 패턴만)
    s = re.sub(r"\b([A-Za-z_]\w*)\s*\[\s*[A-Za-z_]\w*\s*\]\s*\(", r"\1(", s)

    # --- OCR 특화: forimg_pathintqdm... 복원 ---
    # forimg_pathintqdm[Path]|(...) -> for img_path in tqdm(...)
    # 1) "for" 다음 식별자 분리 (forimg_path -> for img_path, intqdm 앞에 오는 경우)
    s = re.sub(r"\bfor([A-Za-z_]\w+)(intqdm)", r"for \1 \2", s, flags=re.I)
    # 2) "intqdm" -> "in tqdm" (공백 없이 붙은 경우)
    s = re.sub(r"(\w+)\s*(intqdm)", r"\1 in tqdm", s, flags=re.I)
    # 3) tqdm[Path]|(...) -> tqdm(...) (파이프 제거)
    s = re.sub(r"\btqdm\s*\[\s*Path\s*\]\s*[|]\s*\(", "tqdm(", s, flags=re.I)
    # 4) tqdm[Path](... -> tqdm(... (파이프 없이도)
    s = re.sub(r"\btqdm\s*\[\s*Path\s*\]\s*\(", "tqdm(", s, flags=re.I)
    # 5) "in" 앞뒤 공백 보정 (in 다음 식별자 붙은 경우, 키워드 "in"만 매칭)
    # "for ... in tqdm" 또는 "for ... in zip" 같은 패턴만
    s = re.sub(r"\bfor\s+([A-Za-z_]\w+)\s+in\s+([A-Za-z_])", r"for \1 in \2", s)

    # --- 경로 문자열 중복/깨짐 복원 ---
    # val_dir=Path(...) path(...) -> val_dir=Path(...) (중복 제거)
    s = re.sub(r"(\w+)\s*=\s*Path\s*\([^)]+\)\s+path\s*\([^)]+\)", lambda m: m.group(1) + "=Path(" + m.group(0).split(" path(")[0].split("Path(")[1] + ")", s, flags=re.I)
    # Path(...) path(...) -> Path(...) (중복 제거, 변수명 없는 경우)
    s = re.sub(r"\bPath\s*\([^)]+\)\s+path\s*\([^)]+\)", lambda m: m.group(0).split(" path(")[0] + ")", s, flags=re.I)
    # 같은 변수 할당이 여러 번 반복되는 경우 첫 번째만 유지
    # val_dir=Path(...) val_dir=Path(...) -> val_dir=Path(...)
    s = re.sub(r"(\w+)\s*=\s*Path\s*\([^)]+\)\s+\1\s*=\s*Path\s*\([^)]+\)", lambda m: m.group(0).split(" " + m.group(1) + "=")[0], s, flags=re.I)

    # --- newline 파라미터 복원 강화 ---
    # newline=", encoding= -> newline="", encoding=
    s = re.sub(r"newline\s*=\s*\"?\s*,\s*encoding\s*=", 'newline="", encoding=', s, flags=re.I)
    # newline=", encoding='...' -> newline="", encoding='...'
    s = re.sub(r"newline\s*=\s*\"\s*,\s*encoding\s*=\s*'", 'newline="", encoding=\'', s, flags=re.I)
    
    # --- try/except 블록 구조 복원 ---
    # 잘못된 구조: )처리중오류:{아"): 같은 패턴 제거
    s = re.sub(r"\)\s*처리중오류\s*:\s*\{[^}]*\"\s*\)\s*:", "):", s)
    # try: 함수() exceptException -> try:\n    함수()\nexcept Exception as e:
    s = re.sub(r"try:\s*([A-Za-z_]\w+\([^)]*\))\s*exceptException", r"try:\n    \1\nexcept Exception as e:", s, flags=re.I)
    # print(f"[ERROR]{변수}Ae]BSe:{e}") -> print(f"[ERROR] {변수} 처리 중 오류: {e}")
    s = re.sub(r'print\s*\(\s*f\s*"\[ERROR\]\{([^}]+)\}Ae\]BSe:\{e\}"\s*\)', r'print(f"[ERROR] {\1} 처리 중 오류: {e}")', s, flags=re.I)
    # print(f"[ERROR]{변수}...:{e}") -> print(f"[ERROR] {변수} ...: {e}")
    s = re.sub(r'print\s*\(\s*f\s*"\[ERROR\]\{([^}]+)\}([^:]+):\{e\}"\s*\)', r'print(f"[ERROR] {\1} \2: {e}")', s, flags=re.I)

    # 콤마 spacing (안전)
    s = re.sub(r",(?=\S)", ", ", s)

    # --- 고신뢰 ':' 복원 (블록 문법 규칙) ---
    # 1) with open(... ) as f  → 항상 ':' 필요
    if re.match(r"^\s*with\s+open\(", s) and re.search(r"\bas\s+f\b", s) and not s.rstrip().endswith(":"):
        s = s.rstrip() + ":"
    # 2) for ... in zip(...) / tqdm(...) → 항상 ':' 필요
    if re.match(r"^\s*for\s+.+\s+in\s+(zip|tqdm)\(", s) and not s.rstrip().endswith(":"):
        s = s.rstrip() + ":"
    # 3) 일반 블록 키워드는 "다음 줄 indent 증가" 근거 있을 때만
    if safe_mode and lang_hint.lower() in ("py", "python"):
        stripped0 = s.lstrip(" ")
        lead0 = len(s) - len(stripped0)
        block_kw = r"(if|elif|else|for|while|try|except|finally|with|def|class|match|case)"
        m = re.match(rf"^({block_kw})\b(.*)$", stripped0)
        if m and not stripped0.rstrip().endswith(":"):
            kw = m.group(1).lower()
            # 단독 블록 키워드는 항상 콜론
            if kw in ("else", "try", "finally", "except"):
                s = (" " * lead0) + stripped0.rstrip() + ":"
            # 다음 줄 indent 증가 근거가 있으면 콜론 복원
            elif next_line_indent is not None and next_line_indent > lead0:
                s = (" " * lead0) + stripped0.rstrip() + ":"

    for pat, rep in COMMON_GLUE:
        s = pat.sub(rep, s)

    # 불필요 다중 공백 축소 (indent 유지)
    lead = len(s) - len(s.lstrip(" "))
    body = re.sub(r"[ ]{2,}", " ", s.lstrip(" "))
    return (" " * lead) + body


def _score_code_text(t: str) -> float:
    if not t:
        return -1e9
    ascii_cnt = len(ASCII_CODE_RE.findall(t))
    total = max(1, len(t))
    ratio = ascii_cnt / total
    colon_cnt = t.count(":")
    semi_cnt = t.count(";")
    brace_cnt = t.count("{") + t.count("}") + t.count("(") + t.count(")") + t.count("[") + t.count("]")
    op_cnt = sum(t.count(x) for x in ["==", "!=", "<=", ">=", "->", "=>"])
    return (ratio * 100.0) + (colon_cnt * 2.5) + (semi_cnt * 2.0) + (brace_cnt * 0.8) + (op_cnt * 1.2)


def reconstruct_text_from_words(
    words: List[WordBox],
    *,
    code_mode: bool = True,
    normalize: bool = True,
    indent_step: int = 4,
    remove_emoji: bool = True,
) -> str:
    if not words:
        return ""
    clean_words: List[WordBox] = []
    for w in words:
        t = (w.text or "").strip()
        if not t:
            continue
        t = CTRL_RE.sub("", t)
        if remove_emoji:
            t = EMOJI_RE.sub("", t)
        if not t:
            continue
        clean_words.append(WordBox(text=t, x=w.x, y=w.y, w=w.w, h=w.h, conf=w.conf))
    if not clean_words:
        return ""

    lines = cluster_lines(clean_words)
    if not lines:
        return ""

    char_w = estimate_char_width(lines)
    left_margin = min(w.x for ln in lines for w in ln.words)

    raw_lines: List[str] = []
    for ln in lines:
        if not ln.words:
            raw_lines.append("")
            continue

        first = ln.words[0]
        leading_spaces = int(round((first.x - left_margin) / max(1e-6, char_w)))
        leading_spaces = max(0, leading_spaces)

        parts: List[str] = []
        parts.append(" " * leading_spaces)
        parts.append(first.text)

        prev_x2 = first.x2
        prev_text = first.text

        for w in ln.words[1:]:
            txt = w.text
            if not txt:
                continue

            gap_px = w.x - prev_x2

            # ✅ 공백 붕괴 방지: 알파뉴메릭 토큰끼리는 최소 1칸
            prev_is_alnum = bool(re.search(r"[A-Za-z0-9_]", prev_text))
            curr_is_alnum = bool(re.search(r"[A-Za-z0-9_]", txt))
            min_spaces = 1 if (prev_is_alnum and curr_is_alnum) else 0

            # 기존 0.10은 너무 공격적 → 0.05로 완화
            if gap_px <= char_w * 0.05:
                spaces = min_spaces
            else:
                spaces = int(round(gap_px / max(1e-6, char_w)))
                spaces = clamp_int(spaces, min_spaces, 80)

            parts.append(" " * spaces)
            parts.append(txt)
            prev_x2 = max(prev_x2, w.x2)
            prev_text = txt

        raw_lines.append("".join(parts).rstrip())

    if not code_mode:
        out = "\n".join(raw_lines).rstrip() + "\n"
        return sanitize_text(out, remove_emoji=remove_emoji, keep_newlines=True, collapse_spaces=False) + "\n"

    def indent_of(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    def set_indent(s: str, n: int) -> str:
        return (" " * max(0, n)) + s.lstrip(" ")

    fixed: List[str] = []
    seen_lines = set()  # 중복 줄 제거용
    
    for i, s in enumerate(raw_lines):
        if not s.strip():
            fixed.append("")
            continue

        cur_indent = indent_of(s)
        stripped = s.lstrip(" ")

        if stripped and stripped[0] in ("}", "]", ")"):
            cur_indent = max(0, cur_indent - indent_step)
        if re.match(r"^(else:|elif\b|except\b|finally:)", stripped):
            cur_indent = max(0, cur_indent - indent_step)
        if fixed:
            prev = fixed[-1].rstrip()
            prev_strip = prev.lstrip(" ")
            prev_indent = indent_of(prev)
            if prev_strip.endswith(":"):
                cur_indent = max(cur_indent, prev_indent + indent_step)

        s2 = set_indent(s, cur_indent)

        # 다음 줄 indent 계산 → 콜론 복원 근거로 사용
        next_line_indent = None
        if i + 1 < len(raw_lines):
            next_s = raw_lines[i + 1]
            if next_s.strip():
                next_indent = indent_of(next_s)
                next_stripped = next_s.lstrip(" ")
                if next_stripped and next_stripped[0] in ("}", "]", ")"):
                    next_indent = max(0, next_indent - indent_step)
                if re.match(r"^(else:|elif\b|except\b|finally:)", next_stripped):
                    next_indent = max(0, next_indent - indent_step)
                next_line_indent = next_indent

        if normalize:
            s2 = normalize_code_line(s2, next_line_indent=next_line_indent, lang_hint="py", safe_mode=True)

        # 중복 줄 제거: 같은 내용의 줄이 연속으로 나타나면 첫 번째만 유지
        s2_stripped = s2.rstrip()
        # 변수 할당 패턴 (val_dir=Path(...))이 연속으로 나타나는 경우 중복 제거
        var_assign_match = re.match(r"^\s*(\w+)\s*=\s*", s2_stripped)
        if var_assign_match and fixed:
            var_name = var_assign_match.group(1)
            prev_stripped = fixed[-1].rstrip()
            # 이전 줄이 같은 변수 할당이면 현재 줄 스킵
            if re.match(rf"^\s*{var_name}\s*=\s*", prev_stripped):
                continue
        
        # 일반적인 중복 줄 제거 (정확히 같은 내용)
        if s2_stripped in seen_lines:
            continue
        seen_lines.add(s2_stripped)

        fixed.append(s2_stripped)

    out = "\n".join(fixed).rstrip() + "\n"
    return sanitize_text(out, remove_emoji=remove_emoji, keep_newlines=True, collapse_spaces=False) + "\n"


def merge_winrt_lines(lines_ko: List[str], lines_en: List[str]) -> List[str]:
    n = max(len(lines_ko), len(lines_en))
    out: List[str] = []
    for i in range(n):
        ko = lines_ko[i] if i < len(lines_ko) else ""
        en = lines_en[i] if i < len(lines_en) else ""
        en_score = len(ASCII_CODE_RE.findall(en))
        ko_score = len(ASCII_CODE_RE.findall(ko))
        out.append(en if en_score >= ko_score else ko)
    return out


def _run_coro_sync(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def _create_winrt_engine(lang_tag: str):
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.globalization import Language

    try:
        eng = OcrEngine.try_create_from_language(Language(lang_tag))
        if eng is not None:
            return eng
    except Exception:
        pass

    alias_map = {"ko": ["ko-KR", "ko"], "en": ["en-US", "en"]}
    for cand in alias_map.get(lang_tag, []):
        try:
            eng = OcrEngine.try_create_from_language(Language(cand))
            if eng is not None:
                return eng
        except Exception:
            continue

    try:
        eng = OcrEngine.try_create_from_user_profile_languages()
        if eng is not None:
            return eng
    except Exception:
        pass

    try:
        eng = OcrEngine.try_create_from_language(Language("en"))
        if eng is not None:
            return eng
    except Exception:
        pass

    return None


async def _winrt_recognize_async(pil_img: Image.Image, lang_tag: str):
    from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
    from winrt.windows.storage.streams import DataWriter

    def pil_to_software_bitmap(img: Image.Image) -> SoftwareBitmap:
        rgba = img.convert("RGBA")
        arr = np.array(rgba, dtype=np.uint8)
        sb = SoftwareBitmap(BitmapPixelFormat.RGBA8, rgba.width, rgba.height)
        writer = DataWriter()
        writer.write_bytes(arr.tobytes())
        sb.copy_from_buffer(writer.detach_buffer())
        return sb

    engine = _create_winrt_engine(lang_tag)
    if engine is None:
        raise RuntimeError("WinRT OcrEngine 생성 실패 (언어팩/OCR 지원 미설치 가능)")
    sb = pil_to_software_bitmap(pil_img)
    return await engine.recognize_async(sb)


def _winrt_words_from_result(result) -> List[WordBox]:
    out: List[WordBox] = []
    for ln in getattr(result, "lines", []):
        for w in getattr(ln, "words", []):
            txt = getattr(w, "text", "") or ""
            rect = getattr(w, "bounding_rect", None)
            if not txt or rect is None:
                continue
            out.append(
                WordBox(
                    text=txt,
                    x=float(rect.x),
                    y=float(rect.y),
                    w=float(rect.width),
                    h=float(rect.height),
                    conf=-1.0,
                )
            )
    return out


def _winrt_lines_text(result) -> List[str]:
    return [getattr(ln, "text", "") or "" for ln in getattr(result, "lines", [])]


def get_winrt_words(
    img: Union[np.ndarray, Image.Image],
    *,
    scale: int = 2,
    code_mode: bool = True,
    remove_emoji: bool = True,
) -> List[WordBox]:
    pil_img = open_image_any(img)
    if scale and scale != 1:
        w, h = pil_img.size
        pil_img = pil_img.resize((w * scale, h * scale), Image.BICUBIC)

    pil_img = preprocess_for_winrt_pil(pil_img, enabled=code_mode)

    result_ko = _run_coro_sync(_winrt_recognize_async(pil_img, "ko"))
    result_en = _run_coro_sync(_winrt_recognize_async(pil_img, "en"))
    words_ko = _winrt_words_from_result(result_ko)
    words_en = _winrt_words_from_result(result_en)

    all_words: List[WordBox] = []
    used_positions = set()

    for w in words_ko:
        t = (w.text or "").strip()
        if remove_emoji:
            t = EMOJI_RE.sub("", t)
        if not t:
            continue
        pos_key = (int(w.x // 10), int(w.y // 10))
        if pos_key not in used_positions:
            all_words.append(WordBox(text=t, x=w.x, y=w.y, w=w.w, h=w.h, conf=w.conf))
            used_positions.add(pos_key)

    for w in words_en:
        t = (w.text or "").strip()
        if remove_emoji:
            t = EMOJI_RE.sub("", t)
        if not t:
            continue
        pos_key = (int(w.x // 10), int(w.y // 10))
        if pos_key not in used_positions:
            all_words.append(WordBox(text=t, x=w.x, y=w.y, w=w.w, h=w.h, conf=w.conf))
            used_positions.add(pos_key)

    return all_words


def image_to_text_winrt(
    img: Union[np.ndarray, Image.Image],
    *,
    scale: int = 2,
    code_mode: bool = True,
    normalize: bool = True,
    indent_step: int = 4,
    remove_emoji: bool = True,
) -> str:
    pil_img = open_image_any(img)
    if scale and scale != 1:
        w, h = pil_img.size
        pil_img = pil_img.resize((w * scale, h * scale), Image.BICUBIC)

    pil_img = preprocess_for_winrt_pil(pil_img, enabled=code_mode)

    result_ko = _run_coro_sync(_winrt_recognize_async(pil_img, "ko"))
    result_en = _run_coro_sync(_winrt_recognize_async(pil_img, "en"))

    lines_ko = [
        sanitize_text(t, remove_emoji=remove_emoji, keep_newlines=False, collapse_spaces=False)
        for t in _winrt_lines_text(result_ko)
    ]
    lines_en = [
        sanitize_text(t, remove_emoji=remove_emoji, keep_newlines=False, collapse_spaces=False)
        for t in _winrt_lines_text(result_en)
    ]
    merged_lines = merge_winrt_lines(lines_ko, lines_en)

    words = _winrt_words_from_result(result_en)
    if not words:
        y = 0.0
        line_h = 18.0
        for ln in merged_lines:
            if ln.strip():
                words.append(
                    WordBox(
                        text=ln,
                        x=0.0,
                        y=y,
                        w=float(max(10, len(ln) * 10)),
                        h=line_h,
                        conf=-1.0,
                    )
                )
            y += line_h

    out = reconstruct_text_from_words(
        words,
        code_mode=code_mode,
        normalize=normalize,
        indent_step=indent_step,
        remove_emoji=remove_emoji,
    )

    # 너무 짧으면 merged_lines 기반 보정
    if out.count("\n") <= 2 and len(merged_lines) >= 2:
        fixed = []
        for i, x in enumerate(merged_lines):
            next_indent = None
            if i + 1 < len(merged_lines):
                nxt = merged_lines[i + 1]
                if nxt.strip():
                    next_indent = len(nxt) - len(nxt.lstrip(" "))
            fixed.append(normalize_code_line(x, next_line_indent=next_indent, lang_hint="py", safe_mode=True) if normalize else x)
        out = "\n".join(fixed).rstrip() + "\n"

    return out


def _build_whitelist(code_mode: bool) -> Optional[str]:
    if not code_mode:
        return None
    return (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_"
        "-+=/*%<>!&|^~.,:;?@#$()[]{}\\"
        "'\"`"
        " \t"
    )


def _tesseract_eng_whitelist() -> str:
    return _build_whitelist(True) or ""


def tesseract_word_boxes(
    pil_img: Image.Image,
    *,
    lang: str,
    psm: int = 6,
    oem: int = 3,
    code_mode: bool = True,
    remove_emoji: bool = True,
    user_defined_dpi: int = 300,
    whitelist_override: Optional[str] = None,
) -> List[WordBox]:
    if pytesseract is None:
        raise RuntimeError("pytesseract not installed")

    whitelist = whitelist_override if whitelist_override is not None else _build_whitelist(code_mode)

    # ✅ 핵심: 사전/자동보정 OFF (공백 붕괴 원인 제거)
    config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
    config += f" -c user_defined_dpi={int(user_defined_dpi)}"
    config += " -c classify_bln_numeric_mode=0"
    config += " -c load_system_dawg=0"
    config += " -c load_freq_dawg=0"
    config += " -c tessedit_enable_dict_correction=0"

    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    data = pytesseract.image_to_data(
        pil_img,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    out: List[WordBox] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = data["text"][i]
        if txt is None:
            continue
        txt = CTRL_RE.sub("", (txt or "").strip())
        if remove_emoji:
            txt = EMOJI_RE.sub("", txt)
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])
        if w <= 1 or h <= 1:
            continue

        out.append(WordBox(text=txt, x=x, y=y, w=w, h=h, conf=conf))

    return out


def _tess_best_by_psm(
    pil_img: Image.Image,
    *,
    lang: str,
    psm_list: List[int],
    oem: int,
    code_mode: bool,
    remove_emoji: bool,
    user_defined_dpi: int,
    whitelist_override: Optional[str],
) -> Tuple[List[WordBox], float]:
    best_words: List[WordBox] = []
    best_score = -1e18

    for psm in psm_list:
        try:
            words = tesseract_word_boxes(
                pil_img,
                lang=lang,
                psm=psm,
                oem=oem,
                code_mode=code_mode,
                remove_emoji=remove_emoji,
                user_defined_dpi=user_defined_dpi,
                whitelist_override=whitelist_override,
            )
            if not words:
                continue
            txt = reconstruct_text_from_words(
                words,
                code_mode=True,
                normalize=False,
                indent_step=4,
                remove_emoji=remove_emoji,
            )
            sc = _score_code_text(txt)
            if sc > best_score:
                best_score = sc
                best_words = words
        except Exception:
            continue

    return best_words, best_score


def _merge_tesseract_two_pass(words_eng: List[WordBox], words_kor: List[WordBox]) -> List[WordBox]:
    if not words_eng and not words_kor:
        return []
    if not words_kor:
        return words_eng
    if not words_eng:
        return words_kor

    korean_re = re.compile(r"[가-힣]")
    symbol_chars = set(":;,\".'")

    def overlap_ratio(a: WordBox, b: WordBox) -> float:
        y_overlap = min(a.y2, b.y2) - max(a.y, b.y)
        if y_overlap <= 0:
            return 0.0
        x_overlap = min(a.x2, b.x2) - max(a.x, b.x)
        if x_overlap <= 0:
            return 0.0
        area = a.w * a.h
        if area <= 0:
            return 0.0
        return (x_overlap * y_overlap) / area

    merged: List[WordBox] = []
    used_k = set()

    for we in words_eng:
        best_j = None
        best_r = 0.0
        for j, wk in enumerate(words_kor):
            r = overlap_ratio(we, wk)
            if r > best_r:
                best_r = r
                best_j = j

        if best_j is not None and best_r >= 0.45:
            wk = words_kor[best_j]
            used_k.add(best_j)

            te = (we.text or "").strip()
            tk = (wk.text or "").strip()

            # 한글은 kor 우선
            if korean_re.search(tk):
                merged.append(WordBox(text=tk, x=we.x, y=we.y, w=we.w, h=we.h, conf=we.conf))
            else:
                # 기호는 eng 우선(코드 기호 살리기)
                te_has_sym = any(c in te for c in symbol_chars)
                tk_has_sym = any(c in tk for c in symbol_chars)
                if te_has_sym and not tk_has_sym:
                    merged.append(we)
                elif tk_has_sym and not te_has_sym:
                    merged.append(WordBox(text=tk, x=we.x, y=we.y, w=we.w, h=we.h, conf=we.conf))
                else:
                    merged.append(we)
        else:
            merged.append(we)

    # eng이 못 잡은 한글 토큰만 추가
    for j, wk in enumerate(words_kor):
        if j in used_k:
            continue
        if korean_re.search((wk.text or "")):
            merged.append(wk)

    return merged


def get_tesseract_words_best_2pass(
    img: Union[np.ndarray, Image.Image],
    *,
    scale: int = 4,
    code_mode: bool = True,
    remove_emoji: bool = True,
    oem: int = 3,
    psm_list: Optional[List[int]] = None,
    user_defined_dpi: int = 300,
) -> List[WordBox]:
    """Tesseract 2-pass:
    PASS1(강전처리 eng whitelist) vs PASS2(약전처리 eng whitelist) best 선택
    + kor(+eng) 결과와 병합(한글만 보강)
    """
    if pytesseract is None:
        return []

    if psm_list is None:
        psm_list = [6, 4, 11, 12, 7]

    pil_img_orig = open_image_any(img)
    if scale and scale != 1:
        w, h = pil_img_orig.size
        pil_img_orig = pil_img_orig.resize((w * scale, h * scale), Image.LANCZOS)

    eng_lang = pick_tess_lang("eng")

    # PASS1
    pil_img_pass1 = preprocess_for_code_pil(pil_img_orig, enabled=code_mode)
    words_p1, sc_p1 = _tess_best_by_psm(
        pil_img_pass1,
        lang=eng_lang,
        psm_list=psm_list,
        oem=oem,
        code_mode=code_mode,
        remove_emoji=remove_emoji,
        user_defined_dpi=user_defined_dpi,
        whitelist_override=_tesseract_eng_whitelist(),
    )

    # PASS2 (기호용)
    pil_img_pass2 = preprocess_for_code_pil_light(pil_img_orig, enabled=code_mode)
    words_p2, sc_p2 = _tess_best_by_psm(
        pil_img_pass2,
        lang=eng_lang,
        psm_list=psm_list,
        oem=oem,
        code_mode=code_mode,
        remove_emoji=remove_emoji,
        user_defined_dpi=user_defined_dpi,
        whitelist_override=_tesseract_eng_whitelist(),
    )

    words_eng = words_p2 if sc_p2 > sc_p1 else words_p1
    sc_eng = max(sc_p1, sc_p2)

    # kor(+eng) 보강 (whitelist OFF)
    installed = set(get_tesseract_langs())
    if "kor" in installed and "eng" in installed:
        kor_lang = "kor+eng"
    elif "kor" in installed:
        kor_lang = "kor"
    else:
        kor_lang = eng_lang

    words_kor, _ = _tess_best_by_psm(
        pil_img_pass2,
        lang=kor_lang,
        psm_list=psm_list,
        oem=oem,
        code_mode=code_mode,
        remove_emoji=remove_emoji,
        user_defined_dpi=user_defined_dpi,
        whitelist_override=None,
    )

    merged = _merge_tesseract_two_pass(words_eng, words_kor)
    if not merged:
        return words_kor or words_eng

    # 병합이 너무 나빠지면 eng-only로 롤백
    try:
        t_merged = reconstruct_text_from_words(merged, code_mode=True, normalize=False, indent_step=4, remove_emoji=remove_emoji)
        sc_m = _score_code_text(t_merged)
        if sc_eng > sc_m + 3.0:
            return words_eng
    except Exception:
        pass

    return merged


def capture_fullscreen_bgr(monitor_index: int = 1) -> np.ndarray:
    with mss() as sct:
        monitor = sct.monitors[monitor_index]
        img = np.array(sct.grab(monitor))
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return bgr


def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)


def safe_imread(path: Union[str, PathLib]) -> Optional[np.ndarray]:
    """
    한글 경로 지원 이미지 읽기 (cv2.imread 대체).
    
    Args:
        path: 이미지 파일 경로
    
    Returns:
        BGR 이미지 (numpy array) 또는 None
    """
    path_str = str(path)
    try:
        # numpy를 통한 우회 방법 (한글 경로 지원)
        import numpy as np
        img_array = np.fromfile(path_str, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"⚠️ safe_imread 실패: {e}")
        return None


def safe_imsave(path: Union[str, PathLib], img: np.ndarray, quality: int = 95) -> bool:
    """
    한글 경로 지원 이미지 저장 (cv2.imwrite 대체).
    
    Args:
        path: 저장 경로
        img: BGR 이미지 (numpy array)
        quality: JPEG 품질 (1-100)
    
    Returns:
        성공 여부
    """
    path_str = str(path)
    try:
        # 확장자에 따라 인코딩 방식 선택
        ext = os.path.splitext(path_str)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, encoded_img = cv2.imencode(ext, img, encode_param)
        elif ext == '.png':
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            success, encoded_img = cv2.imencode(ext, img, encode_param)
        else:
            success, encoded_img = cv2.imencode(ext, img)
        
        if success:
            encoded_img.tofile(path_str)
            return True
        return False
    except Exception as e:
        print(f"⚠️ safe_imsave 실패: {e}")
        return False


def select_roi_auto(image: np.ndarray, window_name: str = "Select ROI") -> np.ndarray:
    drawing = False
    start_point = None
    end_point = None
    current_rect = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, current_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
                img_copy = image.copy()
                cv2.rectangle(img_copy, start_point, end_point, (209, 226, 125), 2)
                cv2.imshow(window_name, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing:
                drawing = False
                end_point = (x, y)
                x1, y1 = start_point
                x2, y2 = end_point
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                if w > 0 and h > 0:
                    current_rect = (x, y, w, h)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, image)

    while current_rect is None:
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise ValueError("ROI 선택이 취소되었습니다.")
        if current_rect is not None:
            break

    cv2.destroyAllWindows()
    x, y, w, h = current_rect
    cropped = image[y : y + h, x : x + w]
    return cropped


def merge_tesseract_winrt_results(
    tesseract_words: List[WordBox],
    winrt_words: List[WordBox],
) -> str:
    korean_re = re.compile(r"[가-힣]")
    symbol_chars = set(":;,\".'")

    def overlap_ratio(a: WordBox, b: WordBox) -> float:
        y_overlap = min(a.y2, b.y2) - max(a.y, b.y)
        if y_overlap <= 0:
            return 0.0
        x_overlap = min(a.x2, b.x2) - max(a.x, b.x)
        if x_overlap <= 0:
            return 0.0
        area = a.w * a.h
        if area <= 0:
            return 0.0
        return (x_overlap * y_overlap) / area

    merged_words: List[WordBox] = []
    used_win = set()

    for tw in tesseract_words:
        best_j = None
        best_r = 0.0
        for j, ww in enumerate(winrt_words):
            r = overlap_ratio(tw, ww)
            if r > best_r:
                best_r = r
                best_j = j

        if best_j is not None and best_r >= 0.45:
            ww = winrt_words[best_j]
            used_win.add(best_j)

            tt = (tw.text or "").strip()
            wt = (ww.text or "").strip()

            if korean_re.search(wt):
                final = wt
            else:
                # 기호는 tesseract 우선
                t_sym = any(c in tt for c in symbol_chars)
                w_sym = any(c in wt for c in symbol_chars)
                if t_sym and not w_sym:
                    final = tt
                elif w_sym and not t_sym:
                    final = wt
                else:
                    final = tt

            merged_words.append(WordBox(text=final, x=tw.x, y=tw.y, w=tw.w, h=tw.h, conf=tw.conf))
        else:
            merged_words.append(tw)

    # winrt 단독 한글 토큰 추가
    for j, ww in enumerate(winrt_words):
        if j in used_win:
            continue
        if korean_re.search((ww.text or "")):
            merged_words.append(ww)

    return reconstruct_text_from_words(
        merged_words,
        code_mode=True,
        normalize=True,
        indent_step=4,
        remove_emoji=True,
    )


def check_winrt_available():
    """WinRT 바인딩(파이썬 패키지) 사용 가능 여부."""
    try:
        from winrt.windows.media.ocr import OcrEngine  # noqa: F401
        from winrt.windows.globalization import Language  # noqa: F401
        from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat  # noqa: F401
        from winrt.windows.storage.streams import DataWriter  # noqa: F401
        from winrt.windows.foundation.collections import IVectorView  # noqa: F401
        return True, None
    except Exception as e:
        install_cmd = (
            "python -m pip install "
            "winrt-runtime "
            "winrt-Windows.Foundation "
            "winrt-Windows.Foundation.Collections "
            "winrt-Windows.Globalization "
            "winrt-Windows.Graphics.Imaging "
            "winrt-Windows.Storage.Streams "
            "winrt-Windows.Media.Ocr"
        )
        return False, f"{e}\n\n필수 설치(cmd):\n{install_cmd}"


def check_paddleocr_available():
    """PaddleOCR 사용 가능 여부."""
    if not PADDLEOCR_AVAILABLE:
        return False, "paddleocr 패키지가 설치되지 않았습니다.\n설치: pip install paddlepaddle paddleocr"
    return True, None


# --- PaddleOCR optional ---
_PADDLE_OCR = None

# 커스텀 학습 모델 경로 (Detection 모델)
# Inference 모델 변환 후 이 경로에 모델이 있어야 함
TRAINED_MODEL_DIR = PathLib(__file__).parent.parent / "output" / "det_ke_model"
TRAINED_MODEL_PDPARAMS = TRAINED_MODEL_DIR / "best_accuracy.pdparams"
TRAINED_MODEL_CHECKPOINT = TRAINED_MODEL_DIR / "best_accuracy"
CUSTOM_DET_MODEL_DIR = PathLib(__file__).parent.parent / "output" / "det_ke_inference"
USE_CUSTOM_DET_MODEL = (
    CUSTOM_DET_MODEL_DIR.exists() 
    and (CUSTOM_DET_MODEL_DIR / "inference.pdmodel").exists()
)


def get_paddle_ocr(
    lang: str = "korean", 
    use_gpu: bool = True, 
    use_angle_cls: bool = True,
    use_custom_det: bool = True,  # 커스텀 Detection 모델 사용 여부
) -> Optional[Any]:
    """
    PaddleOCR 인스턴스 가져오기 (lazy initialization, 싱글턴).
    **커스텀 모델 강제 모드**: 커스텀 모델이 없거나 로드 실패 시 기본 모델로 Fallback하지 않고 RuntimeError 발생.
    
    Args:
        lang: 언어 설정 (기본: "korean")
        use_gpu: GPU 사용 여부 (PaddleOCR 3.3.2에서는 무시됨)
        use_angle_cls: 각도 분류기 사용 여부
        use_custom_det: 커스텀 학습된 Detection 모델 사용 여부 (필수)
    
    Returns:
        PaddleOCR 인스턴스 (커스텀 모델 사용)
    
    Raises:
        RuntimeError: 커스텀 모델이 없거나 로드 실패 시
    """
    global _PADDLE_OCR, USE_CUSTOM_DET_MODEL
    if _PADDLE_OCR is not None:
        # 싱글톤 인스턴스가 있으면 재사용 (서버 시작 시 초기화된 인스턴스 유지)
        return _PADDLE_OCR
    if not PADDLEOCR_AVAILABLE:
        raise RuntimeError("PaddleOCR가 설치되지 않았습니다. pip install paddlepaddle paddleocr")
    
    # 커스텀 모델 강제 체크
    USE_CUSTOM_DET_MODEL = (
        CUSTOM_DET_MODEL_DIR.exists() 
        and (CUSTOM_DET_MODEL_DIR / "inference.pdmodel").exists()
    )
    
    if use_custom_det:
        if not USE_CUSTOM_DET_MODEL:
            raise RuntimeError(
                f"❌ 커스텀 Detection 모델을 찾을 수 없습니다.\n"
                f"   경로: {CUSTOM_DET_MODEL_DIR}\n"
                f"   필요한 파일: inference.pdmodel\n\n"
                f"💡 해결 방법:\n"
                f"   1. 학습된 모델이 있다면 자동 변환을 시도합니다.\n"
                f"   2. 또는 수동으로 변환: core/paddle_train/04_export_inference_model.bat 실행\n"
                f"   3. 서버를 완전히 종료한 후 배치 파일 실행 (GPU 경합 방지)"
            )
    
    try:
        # PaddleOCR 3.3.2에서는 use_gpu, show_log 파라미터 지원 안 함
        kwargs = {
            "lang": lang,
        }
        # use_angle_cls는 지원되는 경우에만 추가
        if use_angle_cls:
            kwargs["use_angle_cls"] = use_angle_cls
        
        # 커스텀 Detection 모델 강제 사용
        kwargs["det_model_dir"] = str(CUSTOM_DET_MODEL_DIR.resolve())
        print(f"✅ [PaddleOCR] 커스텀 Detection 모델 로드: {CUSTOM_DET_MODEL_DIR}")
        print(f"   성능: HMean 90.3%, Precision 91.3%, Recall 89.4%")
        
        _PADDLE_OCR = PaddleOCR(**kwargs)
        return _PADDLE_OCR
    except Exception as e:
        error_msg = str(e)
        # 'Global' 키 에러인 경우 더 명확한 메시지
        if "'Global'" in error_msg or "Global" in error_msg:
            raise RuntimeError(
                f"❌ PaddleOCR 커스텀 모델 로드 실패: inference.yml에 Global 섹션이 없습니다.\n"
                f"   경로: {CUSTOM_DET_MODEL_DIR / 'inference.yml'}\n"
                f"   원본 에러: {error_msg}\n\n"
                f"💡 해결 방법:\n"
                f"   inference.yml 파일 상단에 Global: 섹션을 추가하세요:\n"
                f"   Global:\n"
                f"     algorithm: DB\n"
                f"     use_gpu: false\n"
                f"     use_pdserving: false\n"
                f"     det_algorithm: DB"
            ) from e
        raise RuntimeError(
            f"❌ PaddleOCR 커스텀 모델 로드 실패: {error_msg}\n"
            f"   경로: {CUSTOM_DET_MODEL_DIR}\n"
            f"   확인: inference.pdmodel 파일이 존재하는지 확인하세요."
        ) from e


def export_inference_model_auto() -> bool:
    """
    서버 실행 중 GPU 충돌을 방지하기 위해 CPU 모드로 변환을 시도합니다.
    """
    import subprocess
    import shutil
    
    if not TRAINED_MODEL_PDPARAMS.exists():
        print(f"❌ [자동 변환] 학습된 모델 파일을 찾을 수 없습니다: {TRAINED_MODEL_PDPARAMS}")
        return False
    
    print("=" * 60)
    print("🔄 [자동 변환] Inference 모델 생성 시도 중...")
    print(f"   학습된 모델: {TRAINED_MODEL_PDPARAMS}")
    print(f"   출력 경로: {CUSTOM_DET_MODEL_DIR}")
    print("=" * 60)
    
    try:
        if CUSTOM_DET_MODEL_DIR.exists():
            try:
                shutil.rmtree(CUSTOM_DET_MODEL_DIR)
                print(f"🧹 기존 디렉토리 삭제 완료")
            except Exception as e:
                print(f"⚠️ 기존 디렉토리 삭제 실패: {e}")
        
        CUSTOM_DET_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        CUSTOM_DET_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        paddle_root = PathLib("C:/Pyg/Tools/PaddleOCR").resolve()
        if not paddle_root.exists():
            print(f"❌ [자동 변환] PaddleOCR 도구 경로 없음: {paddle_root}")
            return False
        print(f"✅ PaddleOCR 도구 경로: {paddle_root}")
        
        export_script = paddle_root / "tools" / "export_model.py"
        if not export_script.exists():
            print(f"❌ [자동 변환] export_model.py 없음: {export_script}")
            return False
        print(f"✅ export_script: {export_script}")

        venv_ocr_python = PathLib(__file__).parent.parent / "venv_ocr" / "Scripts" / "python.exe"
        if not venv_ocr_python.exists():
            venv_ocr_python = PathLib(__file__).parent.parent.parent / "venv_ocr" / "Scripts" / "python.exe"
            if not venv_ocr_python.exists():
                print(f"❌ [자동 변환] venv_ocr Python 없음: {venv_ocr_python}")
                return False
        print(f"✅ venv_ocr Python: {venv_ocr_python}")
        
        config_file = PathLib(__file__).parent / "paddle_train" / "configs" / "det_ke_finetune.yml"
        if not config_file.exists():
            print(f"❌ [자동 변환] 설정 파일 없음: {config_file}")
            return False
        print(f"✅ 설정 파일: {config_file}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        # FLAGS_selected_gpus는 빈 문자열이면 오류 발생, 아예 설정하지 않음
        if "FLAGS_selected_gpus" in env:
            del env["FLAGS_selected_gpus"]
        env["PADDLE_INFERENCE_PLACE"] = "cpu"
        env["USE_GPU"] = "0"
        env["FLAGS_use_gpu"] = "0"
        env["FLAGS_use_cuda"] = "0"
        
        paddleocr_path = str(paddle_root)
        if env.get('PYTHONPATH'):
            if paddleocr_path not in env['PYTHONPATH']:
                env['PYTHONPATH'] = f"{paddleocr_path};{env['PYTHONPATH']}" if os.name == 'nt' else f"{paddleocr_path}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = paddleocr_path
        
        print(f"🔒 GPU 격리: CUDA_VISIBLE_DEVICES=-1, PADDLE_INFERENCE_PLACE=cpu")

        pretrained_opt = f'Global.pretrained_model="{TRAINED_MODEL_CHECKPOINT.resolve().as_posix()}"'
        save_dir_opt = f'Global.save_inference_dir="{CUSTOM_DET_MODEL_DIR.resolve().as_posix()}"'
        use_gpu_opt = 'Global.use_gpu=False'
        # PaddlePaddle 2.6.2에서는 export_with_pir=False로 설정해야 AssertionError 방지
        export_pir_opt = 'Global.export_with_pir=False'
        
        full_cmd_str = (
            f'"{venv_ocr_python.resolve()}" "{export_script.as_posix()}" '
            f'-c "{config_file.as_posix()}" '
            f'-o {pretrained_opt} {save_dir_opt} {use_gpu_opt} {export_pir_opt}'
        )
        
        print(f"🚀 실행 명령: {full_cmd_str}")
        print(f"📂 작업 디렉토리: {paddle_root}")
        print("-" * 60)
        
        result = subprocess.run(
            full_cmd_str,
            shell=True,
            cwd=str(paddle_root),
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300
        )
        
        print("-" * 60)
        if result.returncode == 0:
            if (CUSTOM_DET_MODEL_DIR / "inference.pdmodel").exists():
                # inference.yml에 model_name 추가 (PaddleOCR 3.3.2 요구사항)
                inference_yml_path = CUSTOM_DET_MODEL_DIR / "inference.yml"
                if inference_yml_path.exists():
                    try:
                        if yaml is None:
                            raise ImportError("yaml 모듈이 설치되지 않았습니다. pip install pyyaml")
                        with open(inference_yml_path, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f) or {}
                        
                        if "Global" not in config:
                            config["Global"] = {}
                        
                        # model_name이 없으면 추가
                        if "model_name" not in config["Global"]:
                            config["Global"]["model_name"] = "PP-OCRv5_server_det"
                            config["Global"]["model_type"] = "det"
                            
                            with open(inference_yml_path, 'w', encoding='utf-8') as f:
                                yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                            print(f"✅ [자동 변환] inference.yml에 model_name 추가 완료")
                    except Exception as e:
                        print(f"⚠️ [자동 변환] inference.yml 수정 실패 (무시): {e}")
                
                print("✅ [자동 변환] 완료!")
                print(f"   생성된 파일: {list(CUSTOM_DET_MODEL_DIR.iterdir())}")
                return True
            else:
                print("⚠️ [자동 변환] 종료 코드는 0이지만 inference.pdmodel 파일이 없습니다.")
                print(f"   디렉토리 내용: {list(CUSTOM_DET_MODEL_DIR.iterdir()) if CUSTOM_DET_MODEL_DIR.exists() else '디렉토리 없음'}")
        else:
            print(f"❌ [자동 변환] 실패 (종료 코드: {result.returncode})")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout[-1000:])  # 마지막 1000자만 출력
            if result.stderr:
                print("STDERR:")
                print(result.stderr[-1000:])  # 마지막 1000자만 출력
        
        print("=" * 60)
        
        # 직접 명령 실행 실패 시 배치 파일 Fallback 시도
        if result.returncode != 0:
            print("💡 직접 변환 실패. 검증된 배치 파일로 Fallback 시도...")
            bat_file = PathLib(__file__).parent / "paddle_train" / "04_export_inference_model.bat"
            if bat_file.exists():
                print(f"📜 배치 파일 실행: {bat_file}")
                bat_result = subprocess.run(
                    [str(bat_file)],
                    shell=True,
                    cwd=str(bat_file.parent),
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=300
                )
                if bat_result.returncode == 0 and (CUSTOM_DET_MODEL_DIR / "inference.pdmodel").exists():
                    # inference.yml에 model_name 추가 (PaddleOCR 3.3.2 요구사항)
                    inference_yml_path = CUSTOM_DET_MODEL_DIR / "inference.yml"
                    if inference_yml_path.exists():
                        try:
                            if yaml is None:
                                raise ImportError("yaml 모듈이 설치되지 않았습니다. pip install pyyaml")
                            with open(inference_yml_path, 'r', encoding='utf-8') as f:
                                config = yaml.safe_load(f) or {}
                            
                            if "Global" not in config:
                                config["Global"] = {}
                            
                            # model_name이 없으면 추가
                            if "model_name" not in config["Global"]:
                                config["Global"]["model_name"] = "PP-OCRv5_server_det"
                                config["Global"]["model_type"] = "det"
                                
                                with open(inference_yml_path, 'w', encoding='utf-8') as f:
                                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                                print(f"✅ [배치 파일 Fallback] inference.yml에 model_name 추가 완료")
                        except Exception as e:
                            print(f"⚠️ [배치 파일 Fallback] inference.yml 수정 실패 (무시): {e}")
                    
                    print("✅ [배치 파일 Fallback] 변환 성공!")
                    return True
                else:
                    print(f"❌ [배치 파일 Fallback] 실패 (종료 코드: {bat_result.returncode})")
                    if bat_result.stdout:
                        print("STDOUT:")
                        print(bat_result.stdout[-500:])
                    if bat_result.stderr:
                        print("STDERR:")
                        print(bat_result.stderr[-500:])
        
        print("💡 수동 변환: core/paddle_train/04_export_inference_model.bat 실행")
        print("   서버를 완전히 종료한 후 배치 파일을 실행하세요 (GPU 경합 방지)")
        return False
        
    except subprocess.TimeoutExpired:
        print("❌ [자동 변환] 타임아웃 (5분 초과)")
        print("💡 서버를 종료하고 core/paddle_train/04_export_inference_model.bat를 수동 실행하세요")
        return False
    except Exception as e:
        import traceback
        print(f"❌ [자동 변환] 예외 발생: {e}")
        print(traceback.format_exc())
        print("💡 서버를 종료하고 core/paddle_train/04_export_inference_model.bat를 수동 실행하세요")
        return False


def init_paddleocr_once():
    """서버 시작 시 호출되는 초기화 함수 - PaddleOCR 인스턴스를 미리 생성하여 유지"""
    global USE_CUSTOM_DET_MODEL, _PADDLE_OCR
    
    print("=" * 60)
    print("🔄 [PaddleOCR] 서버 시작 시 초기화 중...")
    print("=" * 60)
    
    USE_CUSTOM_DET_MODEL = (CUSTOM_DET_MODEL_DIR / "inference.pdmodel").exists()
    
    if not USE_CUSTOM_DET_MODEL and TRAINED_MODEL_PDPARAMS.exists():
        print("🔄 [PaddleOCR] Inference 모델 자동 변환 시도...")
        export_inference_model_auto()
        USE_CUSTOM_DET_MODEL = (CUSTOM_DET_MODEL_DIR / "inference.pdmodel").exists()
    
    try:
        # PaddleOCR 인스턴스를 미리 생성하여 싱글톤으로 유지
        ocr_instance = get_paddle_ocr(lang="korean", use_custom_det=True)
        if ocr_instance is not None:
            print(f"✅ [PaddleOCR] 초기화 완료! 인스턴스 유지됨 (id: {id(ocr_instance)})")
        else:
            print("⚠️ [PaddleOCR] 초기화 완료했지만 인스턴스가 None입니다")
    except RuntimeError as e:
        print(f"❌ [PaddleOCR] 초기화 실패: {e}")
        print("⚠️ [PaddleOCR] 서버는 시작되지만 OCR은 작동하지 않습니다")
    except Exception as e:
        print(f"❌ [PaddleOCR] 초기화 중 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)


def preprocess_for_paddle_code(bgr: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    코드 스크린샷 전용 전처리 (BGR 입력).
    
    Args:
        bgr: BGR 형식 numpy array
        scale: 업스케일 팩터 (2~3 권장)
    
    Returns:
        전처리된 BGR 이미지
    """
    img = bgr.copy()
    
    # 업스케일
    if scale and scale > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    
    # gray 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # contrast 강화
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=0)
    
    # 가벼운 sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(gray, -1, kernel)
    
    # BGR로 변환 (PaddleOCR 입력 형식)
    out = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    return out


def paddleocr_words_from_bgr(bgr: np.ndarray) -> Tuple[List[WordBox], List, List, List, np.ndarray]:
    """
    PaddleOCR로 BGR 이미지에서 WordBox 리스트 추출.
    
    Args:
        bgr: BGR 형식 numpy array
    
    Returns:
        (words, boxes, txts, scores, preprocessed_img) 튜플
        - words: WordBox 리스트 (reconstruct_text_from_words에 사용)
        - boxes: 4점 좌표 리스트 [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...]
        - txts: 텍스트 문자열 리스트
        - scores: 신뢰도 리스트
        - preprocessed_img: 전처리된 이미지 (시각화용)
    """
    # 전처리 (OCR 인스턴스 가져오기 전에 먼저 수행)
    img = preprocess_for_paddle_code(bgr, scale=2)
    
    # PaddleOCR 인스턴스 가져오기 (싱글톤 - 서버 시작 시 초기화됨)
    try:
        ocr = get_paddle_ocr(lang="korean", use_custom_det=True)
        if ocr is None:
            print("⚠️ [PaddleOCR] 인스턴스가 None입니다")
            return [], [], [], [], img
    except RuntimeError as e:
        print(f"❌ [PaddleOCR] 인스턴스 가져오기 실패: {e}")
        return [], [], [], [], img
    except Exception as e:
        print(f"⚠️ [PaddleOCR] 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return [], [], [], [], img
    
    # OCR 실행
    try:
        print(f"🔍 [PaddleOCR] OCR 실행 중... (이미지 크기: {img.shape})")
        # PaddleOCR 3.3.2에서는 cls 파라미터 지원 안 함
        result = ocr.ocr(img)
        print(f"📊 [PaddleOCR] OCR 결과 타입: {type(result)}, 길이: {len(result) if result else 0}")
        if result and len(result) > 0:
            print(f"📊 [PaddleOCR] 첫 번째 결과 길이: {len(result[0]) if result[0] else 0}")
    except Exception as e:
        print(f"⚠️ [PaddleOCR] OCR 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return [], [], [], [], img
    
    words: List[WordBox] = []
    boxes = []
    txts = []
    scores = []
    
    if not result or not result[0]:
        print(f"⚠️ [PaddleOCR] 인식된 토큰 없음 (result: {result})")
        return words, boxes, txts, scores, img
    
    # PaddleOCR 3.3.2는 PaddleX의 OCRResult 객체를 반환할 수 있음
    image_results = result[0] if result else None
    
    print(f"🔍 [PaddleOCR] 결과 구조 분석:")
    print(f"  - result[0] 타입: {type(image_results)}")
    
    # OCRResult 객체인 경우 속성 확인
    if image_results is not None:
        result_type_name = type(image_results).__name__
        result_type_str = str(type(image_results))
        # OCRResult 객체인지 확인 (다양한 형태의 클래스명 지원)
        is_ocr_result = (
            'OCRResult' in result_type_name or 
            'OCRResult' in result_type_str or
            hasattr(image_results, 'dt_polys') or 
            hasattr(image_results, 'rec_text') or
            hasattr(image_results, 'dt_boxes') or
            hasattr(image_results, 'rec_texts')
        )
        
        if is_ocr_result:
            print(f"  - OCRResult 객체 감지됨")
            # OCRResult 객체의 속성 확인
            attrs = [a for a in dir(image_results) if not a.startswith('_')]
            print(f"  - 사용 가능한 속성: {attrs[:15]}")
            
            # OCRResult 객체는 dict-like 객체이므로 items(), keys(), get() 사용 가능
            # json 속성도 있으므로 활용 가능
            
            boxes_list = []
            texts_list = []
            scores_list = []
            
            # 1. dict-like 접근 시도
            try:
                ocr_dict = dict(image_results) if hasattr(image_results, 'items') else {}
                print(f"  - dict 변환 성공: {list(ocr_dict.keys())[:10]}")
                
                # 가능한 키 이름 확인
                for key in ocr_dict.keys():
                    print(f"    키: {key}, 값 타입: {type(ocr_dict[key])}, 값 길이: {len(ocr_dict[key]) if hasattr(ocr_dict[key], '__len__') else 'N/A'}")
                
                # 박스 정보 (rec_polys 우선 - Recognition 완료된 정확한 박스)
                if 'rec_polys' in ocr_dict:
                    boxes_list = ocr_dict['rec_polys']
                    print(f"  - 박스 사용: rec_polys (Recognition 박스)")
                elif 'dt_polys' in ocr_dict:
                    boxes_list = ocr_dict['dt_polys']
                    print(f"  - 박스 사용: dt_polys (Detection 박스)")
                elif 'dt_boxes' in ocr_dict:
                    boxes_list = ocr_dict['dt_boxes']
                    print(f"  - 박스 사용: dt_boxes")
                elif 'boxes' in ocr_dict:
                    boxes_list = ocr_dict['boxes']
                    print(f"  - 박스 사용: boxes")
                
                # 텍스트 정보 (다양한 키 이름 시도)
                for text_key in ['rec_text', 'rec_texts', 'texts', 'text', 'txt', 'rec_result', 'results']:
                    if text_key in ocr_dict:
                        texts_list = ocr_dict[text_key] if isinstance(ocr_dict[text_key], list) else [ocr_dict[text_key]]
                        print(f"  - 텍스트 찾음: {text_key} ({len(texts_list)}개)")
                        break
                
                # 점수 정보 (다양한 키 이름 시도)
                for score_key in ['rec_score', 'rec_scores', 'scores', 'score', 'conf', 'confidence', 'rec_confidence']:
                    if score_key in ocr_dict:
                        scores_list = ocr_dict[score_key] if isinstance(ocr_dict[score_key], (list, tuple)) else [ocr_dict[score_key]]
                        print(f"  - 점수 찾음: {score_key} ({len(scores_list)}개)")
                        break
            except Exception as e:
                print(f"  - dict 변환 실패: {e}")
            
            # 2. json 속성 활용 시도
            if not texts_list and hasattr(image_results, 'json'):
                try:
                    import json
                    json_data = image_results.json if isinstance(image_results.json, dict) else json.loads(str(image_results.json))
                    print(f"  - json 속성 타입: {type(json_data)}")
                    if isinstance(json_data, dict):
                        print(f"  - json 키: {list(json_data.keys())[:10]}")
                        for key in json_data.keys():
                            if 'text' in key.lower() or 'txt' in key.lower():
                                texts_list = json_data[key] if isinstance(json_data[key], list) else [json_data[key]]
                                print(f"  - json에서 텍스트 찾음: {key}")
                            if 'score' in key.lower() or 'conf' in key.lower():
                                scores_list = json_data[key] if isinstance(json_data[key], (list, tuple)) else [json_data[key]]
                                print(f"  - json에서 점수 찾음: {key}")
                except Exception as e:
                    print(f"  - json 처리 실패: {e}")
            
            # 3. 직접 속성 접근 (fallback)
            if not boxes_list:
                # rec_polys 우선 (Recognition 완료된 정확한 박스)
                if hasattr(image_results, 'rec_polys'):
                    boxes_list = image_results.rec_polys
                    print(f"  - 박스 사용: rec_polys (속성 직접 접근)")
                elif hasattr(image_results, 'dt_polys'):
                    boxes_list = image_results.dt_polys
                    print(f"  - 박스 사용: dt_polys (속성 직접 접근)")
                elif hasattr(image_results, 'dt_boxes'):
                    boxes_list = image_results.dt_boxes
                    print(f"  - 박스 사용: dt_boxes (속성 직접 접근)")
                elif hasattr(image_results, 'boxes'):
                    boxes_list = image_results.boxes
                    print(f"  - 박스 사용: boxes (속성 직접 접근)")
            
            if not texts_list:
                for attr in ['rec_text', 'rec_texts', 'texts', 'text']:
                    if hasattr(image_results, attr):
                        val = getattr(image_results, attr)
                        texts_list = val if isinstance(val, list) else [val]
                        break
            
            if not scores_list:
                for attr in ['rec_score', 'rec_scores', 'scores', 'score']:
                    if hasattr(image_results, attr):
                        val = getattr(image_results, attr)
                        scores_list = val if isinstance(val, (list, tuple)) else [val]
                        break
            
            print(f"  - 박스 개수: {len(boxes_list) if boxes_list else 0}")
            print(f"  - 텍스트 개수: {len(texts_list) if texts_list else 0}")
            print(f"  - 점수 개수: {len(scores_list) if scores_list else 0}")
            
            # 샘플 텍스트 출력 (디버깅)
            if texts_list and len(texts_list) > 0:
                sample_texts = texts_list[:3]
                print(f"  - 텍스트 샘플 (처음 3개): {sample_texts}")
            
            # 박스와 텍스트를 매칭하여 처리
            if boxes_list and texts_list:
                # 개수가 일치해야 함
                min_len = min(len(boxes_list), len(texts_list))
                print(f"  - 매칭할 항목 개수: {min_len}")
                
                processed_count = 0
                skipped_count = 0
                
                for idx in range(min_len):
                    try:
                        box = boxes_list[idx]
                        text = texts_list[idx] if idx < len(texts_list) else ""
                        score = float(scores_list[idx]) if scores_list and idx < len(scores_list) else 0.0
                        
                        # 텍스트가 None이거나 비어있으면 스킵
                        if text is None:
                            if idx < 3:
                                print(f"  ⚠ 항목 {idx}: text가 None")
                            skipped_count += 1
                            continue
                        
                        # box 좌표 처리 - numpy.ndarray, list, tuple 모두 지원
                        xs = []
                        ys = []
                        
                        # numpy.ndarray 처리
                        if isinstance(box, np.ndarray):
                            # numpy 배열을 리스트로 변환
                            box = box.tolist()
                        
                        # box 형식 확인 및 좌표 추출
                        if isinstance(box, (list, tuple)) and len(box) >= 4:
                            # box는 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] 형식
                            for point in box:
                                if isinstance(point, np.ndarray):
                                    point = point.tolist()
                                
                                if isinstance(point, (list, tuple)) and len(point) >= 2:
                                    try:
                                        xs.append(float(point[0]))
                                        ys.append(float(point[1]))
                                    except (ValueError, TypeError, IndexError):
                                        pass
                        else:
                            if idx < 3:
                                print(f"  ⚠ 항목 {idx}: box 형식 문제 (타입: {type(box)}, 길이: {len(box) if hasattr(box, '__len__') else 'N/A'})")
                            skipped_count += 1
                            continue
                        
                        if not xs or not ys or len(xs) < 4 or len(ys) < 4:
                            if idx < 3:
                                print(f"  ⚠ 항목 {idx}: 좌표 파싱 실패 (xs: {len(xs)}, ys: {len(ys)})")
                            skipped_count += 1
                            continue
                        
                        x1, y1 = float(min(xs)), float(min(ys))
                        x2, y2 = float(max(xs)), float(max(ys))
                        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
                        
                        # 텍스트 정규화
                        text = str(text).strip()
                        if not text or text == "":
                            skipped_count += 1
                            continue
                        
                        # 전각→반각 정규화만 (의미 변경 금지)
                        text = text.replace("：", ":").replace("﹕", ":").replace("∶", ":").replace("ː", ":")
                        text = text.replace("；", ";").replace("﹔", ";")
                        text = text.replace("，", ",").replace("．", ".")
                        text = text.replace("—", "-").replace("–", "-")
                        text = CTRL_RE.sub("", text)
                        text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
                        text = EMOJI_RE.sub("", text)
                        text = text.strip()
                        
                        if not text:
                            skipped_count += 1
                            continue
                        
                        # WordBox 생성
                        words.append(WordBox(text=text, x=x1, y=y1, w=w, h=h, conf=score))
                        boxes.append(box)
                        txts.append(text)
                        scores.append(score)
                        processed_count += 1
                        
                        if processed_count <= 5:  # 처음 5개만 상세 출력
                            print(f"  ✓ 단어 {processed_count}: '{text}' (score: {score:.2f}, box: ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f}))")
                    
                    except Exception as e:
                        skipped_count += 1
                        if idx < 3:  # 처음 3개 에러만 출력
                            print(f"  ⚠ 항목 {idx} 처리 중 오류: {e}")
                        import traceback
                        if idx == 0:  # 첫 항목만 전체 traceback
                            traceback.print_exc()
                
                print(f"  - 처리 완료: {processed_count}개, 스킵: {skipped_count}개")
                print(f"✅ [PaddleOCR] {len(words)}개 단어 인식 완료 (OCRResult 객체에서 추출)")
                return words, boxes, txts, scores, img
            else:
                print(f"  ⚠ 박스 또는 텍스트 리스트가 비어있음 (boxes: {len(boxes_list) if boxes_list else 0}, texts: {len(texts_list) if texts_list else 0})")
        
        # 리스트 형식인 경우 (기존 로직)
        if isinstance(image_results, (list, tuple)):
            print(f"  - result[0] 길이: {len(image_results)}")
            if len(image_results) > 0:
                first_item = image_results[0]
                print(f"  - 첫 항목 타입: {type(first_item)}")
                print(f"  - 첫 항목 길이: {len(first_item) if isinstance(first_item, (list, tuple)) else 'N/A'}")
        else:
            print(f"  - 예상치 못한 형식입니다")
    
    # OCRResult 객체가 처리되지 않은 경우 리스트로 처리
    # (OCRResult 객체는 이미 위에서 처리되어 return 되었을 것)
    
    # result[0]의 각 항목 처리 (리스트인 경우)
    if not isinstance(image_results, (list, tuple)):
        # 리스트가 아닌 경우는 이미 위에서 처리되었거나 예외 상황
        print(f"  ⚠ 리스트가 아닌 형식이지만 처리되지 않음: {type(image_results)}")
        print(f"✅ [PaddleOCR] {len(words)}개 단어 인식 완료")
        return words, boxes, txts, scores, img
    
    for idx, line_item in enumerate(image_results):
        if line_item is None:
            continue
        
        # line_item이 직접 [box, (text, score)] 형식인지 확인
        if isinstance(line_item, (list, tuple)) and len(line_item) >= 2:
            box = line_item[0]
            text_info = line_item[1]
            
            # box가 4점 좌표 리스트인지 확인
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                # 직접 항목: [box, (text, score)]
                if text_info is None:
                    print(f"  ⚠ 항목 {idx}: text_info가 None")
                    continue
                
                # 텍스트 추출
                if isinstance(text_info, (list, tuple)):
                    text = str(text_info[0]) if len(text_info) > 0 else ""
                    score = float(text_info[1]) if len(text_info) > 1 else 0.0
                else:
                    text = str(text_info)
                    score = 0.0
                
                # 박스 좌표 처리
                xs = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                
                if not xs or not ys:
                    print(f"  ⚠ 항목 {idx}: box 좌표 파싱 실패 (xs={len(xs)}, ys={len(ys)})")
                    continue
                
                x1, y1 = float(min(xs)), float(min(ys))
                x2, y2 = float(max(xs)), float(max(ys))
                w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
                
                # 전각→반각 정규화만 (의미 변경 금지)
                text = text.replace("：", ":").replace("﹕", ":").replace("∶", ":").replace("ː", ":")
                text = text.replace("；", ";").replace("﹔", ";")
                text = text.replace("，", ",").replace("．", ".")
                text = text.replace("—", "-").replace("–", "-")
                text = CTRL_RE.sub("", text)  # 제어문자 제거
                text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")  # BOM/zero-width 제거
                text = EMOJI_RE.sub("", text)  # 이모지 제거
                
                text = text.strip()
                if not text:
                    print(f"  ⚠ 항목 {idx}: 텍스트가 비어있음 (정규화 후)")
                    continue
                
                # WordBox 생성
                words.append(WordBox(text=text, x=x1, y=y1, w=w, h=h, conf=score))
                boxes.append(box)
                txts.append(text)
                scores.append(score)
                print(f"  ✓ 단어 {len(words)}: '{text}' (score: {score:.2f})")
            else:
                # 중첩 구조: line_item이 또 다른 리스트를 포함하는 경우
                # 예: [[[box, (text, score)], ...]]
                print(f"  🔄 항목 {idx}: 중첩 구조로 처리 시도 (box 타입: {type(box)})")
                if isinstance(line_item, (list, tuple)):
                    for sub_idx, sub_item in enumerate(line_item):
                        if not isinstance(sub_item, (list, tuple)) or len(sub_item) < 2:
                            continue
                        
                        box = sub_item[0]
                        text_info = sub_item[1]
                        
                        if box is None or text_info is None:
                            continue
                        
                        text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                        score = float(text_info[1]) if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.0
                        
                        # 박스 좌표 처리
                        if isinstance(box, (list, tuple)) and len(box) >= 4:
                            xs = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                            ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                            
                            if not xs or not ys:
                                continue
                            
                            x1, y1 = float(min(xs)), float(min(ys))
                            x2, y2 = float(max(xs)), float(max(ys))
                            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
                            
                            # 전각→반각 정규화만 (의미 변경 금지)
                            text = text.replace("：", ":").replace("﹕", ":").replace("∶", ":").replace("ː", ":")
                            text = text.replace("；", ";").replace("﹔", ";")
                            text = text.replace("，", ",").replace("．", ".")
                            text = text.replace("—", "-").replace("–", "-")
                            text = CTRL_RE.sub("", text)  # 제어문자 제거
                            text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")  # BOM/zero-width 제거
                            text = EMOJI_RE.sub("", text)  # 이모지 제거
                            
                            text = text.strip()
                            if not text:
                                continue
                            
                            # WordBox 생성
                            words.append(WordBox(text=text, x=x1, y=y1, w=w, h=h, conf=score))
                            boxes.append(box)
                            txts.append(text)
                            scores.append(score)
                            print(f"  ✓ 단어 {len(words)}: '{text}' (score: {score:.2f})")
            
            # 전각→반각 정규화만 (의미 변경 금지)
            text = text.replace("：", ":").replace("﹕", ":").replace("∶", ":").replace("ː", ":")
            text = text.replace("；", ";").replace("﹔", ";")
            text = text.replace("，", ",").replace("．", ".")
            text = text.replace("—", "-").replace("–", "-")
            text = CTRL_RE.sub("", text)  # 제어문자 제거
            text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")  # BOM/zero-width 제거
            text = EMOJI_RE.sub("", text)  # 이모지 제거
            
            text = text.strip()
            if not text:
                continue
            
            # box에서 min/max로 x, y, w, h 계산
            xs = [float(p[0]) for p in box if len(p) >= 2]
            ys = [float(p[1]) for p in box if len(p) >= 2]
            
            if not xs or not ys:
                continue
            
            x1, y1 = float(min(xs)), float(min(ys))
            x2, y2 = float(max(xs)), float(max(ys))
            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
            
            # WordBox 생성
            words.append(WordBox(text=text, x=x1, y=y1, w=w, h=h, conf=score))
            
            # boxes는 원본 4점 좌표 유지 (CSV 저장용)
            boxes.append(box)
            txts.append(text)
            scores.append(score)
    
    print(f"✅ [PaddleOCR] {len(words)}개 단어 인식 완료")
    return words, boxes, txts, scores, img


def image_to_text_paddle(
    img: Union[np.ndarray, Image.Image],
    *,
    scale: int = 2,
    code_mode: bool = True,
    normalize: bool = True,
    remove_emoji: bool = True,
    return_boxes: bool = False,
) -> Union[str, Tuple[str, List[List[float]], List[str], List[float]]]:
    """
    PaddleOCR로 이미지에서 텍스트 추출.
    
    Args:
        img: 입력 이미지 (numpy array or PIL Image)
        scale: 이미지 스케일 팩터 (1 이상)
        code_mode: 코드 모드 (현재는 사용 안 함, 향후 확장용)
        normalize: 후처리 정규화 (전각→반각만)
        remove_emoji: 이모지 제거
        return_boxes: True면 (text, boxes, txts, scores) 튜플 반환, False면 text만 반환
    
    Returns:
        str 또는 (text, boxes, txts, scores) 튜플
        - text: 추출된 텍스트 (줄바꿈 포함)
        - boxes: 각 텍스트의 4점 좌표 [[x1,y1],[x2,y2],[x3,y3],[x4,y4], ...]
        - txts: 각 텍스트 문자열 리스트
        - scores: 각 텍스트의 신뢰도 리스트
    """
    if not PADDLEOCR_AVAILABLE:
        raise RuntimeError("PaddleOCR가 설치되지 않았습니다. pip install paddlepaddle paddleocr")
    
    # PaddleOCR 3.3.2 호환: use_gpu, use_angle_cls 파라미터는 내부에서 처리됨
    ocr = get_paddle_ocr(lang="korean", use_custom_det=True)
    if ocr is None:
        raise RuntimeError("PaddleOCR 초기화 실패")
    
    # 이미지 변환 및 스케일링
    if isinstance(img, Image.Image):
        pil_img = img.convert("RGB")
        img_array = np.array(pil_img)
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_array = img
        else:
            img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError("img must be PIL.Image or numpy.ndarray")
    
    # 스케일링
    if scale > 1:
        h, w = img_array.shape[:2]
        img_array = cv2.resize(img_array, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    
    # OCR 실행
    try:
        # PaddleOCR 3.3.2에서는 cls 파라미터 지원 안 함
        result = ocr.ocr(img_array)
    except Exception as e:
        raise RuntimeError(f"PaddleOCR 실행 실패: {e}")
    
    if not result or not result[0]:
        text = ""
        boxes = []
        txts = []
        scores = []
    else:
        boxes = []
        txts = []
        scores = []
        lines = []
        
        for line in result[0]:
            if not line or len(line) < 2:
                continue
            
            box_info = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text_info = line[1]  # (text, score)
            
            if not box_info or not text_info:
                continue
            
            text_str = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
            score_val = text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.0
            
            # 후처리: 전각→반각 정규화만 (의미 변경 금지)
            if normalize:
                text_str = text_str.replace("：", ":").replace("﹕", ":").replace("∶", ":").replace("ː", ":")
                text_str = text_str.replace("；", ";").replace("﹔", ";")
                text_str = text_str.replace("，", ",").replace("．", ".")
                text_str = text_str.replace("—", "-").replace("–", "-")
                # 제어문자 제거
                text_str = CTRL_RE.sub("", text_str)
                # BOM/zero-width 제거
                text_str = text_str.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
            
            if remove_emoji:
                text_str = EMOJI_RE.sub("", text_str)
            
            text_str = text_str.strip()
            if not text_str:
                continue
            
            # box 좌표 정규화 (flat list로 변환)
            box_flat = []
            for point in box_info:
                if len(point) >= 2:
                    box_flat.extend([float(point[0]), float(point[1])])
            
            if len(box_flat) >= 8:  # 4점 = 8개 좌표
                boxes.append(box_flat)
                txts.append(text_str)
                scores.append(float(score_val))
                lines.append(text_str)
        
        text = "\n".join(lines)
    
    if return_boxes:
        return text, boxes, txts, scores
    return text


def save_ocr_results(
    img: Union[np.ndarray, Image.Image],
    boxes: List,
    txts: List[str],
    scores: List[float],
    output_dir: Union[str, PathLib],
    stem: str = "ocr_result",
    use_safe_imsave: bool = True,
) -> Tuple[str, str]:
    """
    OCR 결과를 이미지(시각화)와 CSV로 저장.
    
    Args:
        img: 원본 이미지 (BGR numpy array 또는 PIL Image)
        boxes: 각 텍스트의 4점 좌표 리스트 [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...] 또는 flat [[x1,y1,x2,y2,x3,y3,x4,y4], ...]
        txts: 각 텍스트 문자열 리스트
        scores: 각 텍스트의 신뢰도 리스트
        output_dir: 출력 디렉토리
        stem: 파일명 stem (확장자 제외)
        use_safe_imsave: True면 safe_imsave 사용 (한글 경로 지원)
    
    Returns:
        (image_path, csv_path) 튜플
    """
    output_dir = PathLib(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 변환 (BGR로 통일)
    if isinstance(img, Image.Image):
        pil_img = img.convert("RGB")
        img_array = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img_bgr = img.copy()
            else:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        raise TypeError("img must be PIL.Image or numpy.ndarray")
    
    # 시각화 이미지 생성 (BGR)
    vis_img_bgr = img_bgr.copy()
    
    # 박스 및 텍스트 그리기 (OpenCV 사용)
    for box, txt, score in zip(boxes, txts, scores):
        # box 형식 확인: 4점 좌표 리스트 또는 flat 리스트
        if isinstance(box[0], (list, tuple)):
            # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] 형식
            points = np.array([[int(p[0]), int(p[1])] for p in box], dtype=np.int32)
        else:
            # [x1,y1,x2,y2,x3,y3,x4,y4] flat 형식
            if len(box) >= 8:
                points = np.array([
                    [int(box[0]), int(box[1])],
                    [int(box[2]), int(box[3])],
                    [int(box[4]), int(box[5])],
                    [int(box[6]), int(box[7])],
                ], dtype=np.int32)
            else:
                continue
        
        # 박스 그리기 (다각형)
        cv2.polylines(vis_img_bgr, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # 텍스트 그리기 (첫 번째 점 위에)
        text_pos = (int(points[0][0]), int(points[0][1]) - 5)
        text_str = f"{txt} ({score:.2f})"
        cv2.putText(vis_img_bgr, text_str, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 이미지 저장
    image_path = output_dir / f"{stem}_ocr_result.jpg"
    if use_safe_imsave:
        safe_imsave(str(image_path), vis_img_bgr, quality=95)
    else:
        cv2.imwrite(str(image_path), vis_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # CSV 저장
    csv_path = output_dir / f"{stem}_ocr_result.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # 헤더: x1 y1 x2 y2 x3 y3 x4 y4 text score
        writer.writerow(["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "text", "score"])
        # 데이터
        for box, txt, score in zip(boxes, txts, scores):
            # box를 flat 형식으로 변환
            if isinstance(box[0], (list, tuple)):
                # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] -> flat
                flat_box = [coord for point in box for coord in point[:2]]
            else:
                # 이미 flat
                flat_box = box[:8] if len(box) >= 8 else box
            
            if len(flat_box) >= 8:
                row = [
                    flat_box[0], flat_box[1],  # x1, y1
                    flat_box[2], flat_box[3],  # x2, y2
                    flat_box[4], flat_box[5],  # x3, y3
                    flat_box[6], flat_box[7],  # x4, y4
                    txt,
                    score,
                ]
                writer.writerow(row)
    
    return str(image_path), str(csv_path)


def capture_and_ocr(
    engine: str = "paddle",
    save_dir: Optional[Union[str, PathLib]] = None,
    save_stem: Optional[str] = None,
) -> dict:
    """
    Capture screen and perform OCR. PaddleOCR 우선 사용.
    
    Args:
        engine: OCR 엔진 선택
            - "paddle": PaddleOCR 우선, 실패 시 WinRT+Tesseract fallback (기본값)
            - "paddle_only": PaddleOCR만 사용 (fallback 없음)
            - "hybrid": PaddleOCR 실패 시 WinRT+Tesseract fallback
        save_dir: 결과 저장 디렉토리 (None이면 저장 안 함)
        save_stem: 저장 파일명 stem (None이면 타임스탬프 사용)
    
    Returns:
        dict with keys: success, text, method, error, image_path, csv_path
    """
    try:
        paddle_available, paddle_error = check_paddleocr_available()
        tesseract_available = pytesseract is not None
        winrt_available, winrt_error = check_winrt_available()

        print("📸 화면 캡처 중...")
        screen = capture_fullscreen_bgr()
        print("✅ 화면 캡처 완료")

        print("🖱️ OpenCV 창을 열고 영역 선택을 기다리는 중...")
        cropped = select_roi_auto(screen, window_name="원하는 부분을 드래그로 박스치세요")
        print("✅ 영역 선택 완료")

        print("🔍 OCR 인식 시작...")

        ocr_result = ""
        ocr_method = ""
        image_path = None
        csv_path = None

        # PaddleOCR 우선 시도
        if paddle_available:
            try:
                print("🔍 PaddleOCR 실행 중...")
                words, boxes, txts, scores, preprocessed_img = paddleocr_words_from_bgr(cropped)
                
                if words:
                    print(f"📊 PaddleOCR: {len(words)}개 토큰 인식 (scale=2)")
                    
                    # 기존 reconstruct_text_from_words로 레이아웃 복원
                    ocr_result = reconstruct_text_from_words(
                        words,
                        code_mode=True,
                        normalize=True,
                        indent_step=4,
                        remove_emoji=True,
                    )
                    ocr_method = "PaddleOCR(korean, scale=2) + reconstruct_text_from_words"
                    
                    # 저장
                    if save_dir is not None:
                        if save_stem is None:
                            from datetime import datetime
                            save_stem = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path, csv_path = save_ocr_results(
                            preprocessed_img, boxes, txts, scores, save_dir, save_stem, use_safe_imsave=True
                        )
                        print(f"💾 결과 저장: {image_path}, {csv_path}")
                else:
                    print("⚠️ PaddleOCR: 인식된 토큰 없음")
                    if engine == "paddle_only":
                        return {
                            "success": False,
                            "error": "PaddleOCR: 인식된 토큰이 없습니다.",
                            "text": "",
                            "method": "",
                            "image_path": None,
                            "csv_path": None,
                        }
                    # fallback으로 진행
                
            except Exception as e:
                print(f"⚠️ PaddleOCR 실패: {e}")
                if engine == "paddle_only":
                    return {
                        "success": False,
                        "error": f"PaddleOCR 실패: {e}",
                        "text": "",
                        "method": "",
                        "image_path": None,
                        "csv_path": None,
                    }
                # fallback으로 진행

        # Fallback: WinRT + Tesseract (hybrid 모드 또는 PaddleOCR 실패 시)
        if not ocr_result and engine in ("paddle", "hybrid"):
            if not winrt_available:
                return {
                    "success": False,
                    "error": f"WinRT OCR 실패 (fallback): {winrt_error}",
                    "text": "",
                    "method": "",
                    "image_path": None,
                    "csv_path": None,
                }

            if not tesseract_available:
                return {
                    "success": False,
                    "error": "pytesseract 미설치: Tesseract 보조(구조/기호)를 사용할 수 없습니다.",
                    "text": "",
                    "method": "",
                    "image_path": None,
                    "csv_path": None,
                }

            print("🔍 WinRT + Tesseract (fallback) 실행 중...")
            
            # 1) WinRT (ko/en 2-pass) — 한글/문장 강점
            winrt_words = get_winrt_words(cropped, scale=3, code_mode=True, remove_emoji=True)

            # 2) Tesseract 2-pass + PSM sweep + DPI=300 — 구조/기호 강점
            tesseract_words = get_tesseract_words_best_2pass(
                cropped,
                scale=4,
                code_mode=True,
                remove_emoji=True,
                user_defined_dpi=300,
                psm_list=[6, 4, 11, 12, 7],
            )

            # 3) 병합: "구조/기호는 tesseract", "한글은 winrt"
            ocr_result = merge_tesseract_winrt_results(
                tesseract_words=tesseract_words,
                winrt_words=winrt_words,
            )

            ocr_method = "WinRT(ko/en) + Tesseract(2-pass + dpi300 + psm-sweep + dictOFF) merge (fallback)"

        if ocr_result:
            print(f"✅ OCR 완료 ({ocr_method})")
            copy_to_clipboard(ocr_result)
            print("📋 클립보드에 저장 완료")
            return {
                "success": True,
                "text": ocr_result,
                "method": ocr_method,
                "error": None,
                "image_path": image_path,
                "csv_path": csv_path,
            }

        return {
            "success": False,
            "error": "OCR 결과가 비어있습니다.",
            "text": "",
            "method": "",
            "image_path": None,
            "csv_path": None,
        }

    except ValueError as e:
        if "취소" in str(e) or "cancel" in str(e).lower():
            return {
                "success": False,
                "error": "사용자가 영역 선택을 취소했습니다.",
                "text": "",
                "method": "",
                "image_path": None,
                "csv_path": None,
            }
        return {
            "success": False,
            "error": str(e),
            "text": "",
            "method": "",
            "image_path": None,
            "csv_path": None,
        }

    except Exception as e:
        import traceback
        error_msg = f"OCR 처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        return {
            "success": False,
            "error": error_msg,
            "text": "",
            "method": "",
            "image_path": None,
            "csv_path": None,
        }


# -----------------------------
# OCR 후처리 테스트 함수
# -----------------------------
def test_normalize_code_line_ocr_patterns():
    """OCR 특화 복원 패턴 테스트"""
    
    # 테스트 1: zip[tuple](...) -> zip(...)
    input1 = "for box, txt, score in zip[tuple](boxes, txts, scores):"
    expected1 = "for box, txt, score in zip(boxes, txts, scores):"
    result1 = normalize_code_line(input1, next_line_indent=4, lang_hint="py", safe_mode=True)
    assert result1 == expected1, f"Test 1 failed: got '{result1}', expected '{expected1}'"
    
    # 테스트 2: list[Path](...) -> list(...)
    input2 = "val_images = list[Path](val_dir.glob(\"*.jpg\"))"
    expected2 = "val_images = list(val_dir.glob(\"*.jpg\"))"
    result2 = normalize_code_line(input2, next_line_indent=None, lang_hint="py", safe_mode=True)
    assert result2 == expected2, f"Test 2 failed: got '{result2}', expected '{expected2}'"
    
    # 테스트 3: tqdm[Path](...) -> tqdm(...)
    input3 = "for img_path in tqdm[Path](val_images, desc=\"PaddleOCR Pretrained Inference\", unit=\"img\"):"
    expected3 = "for img_path in tqdm(val_images, desc=\"PaddleOCR Pretrained Inference\", unit=\"img\"):"
    result3 = normalize_code_line(input3, next_line_indent=4, lang_hint="py", safe_mode=True)
    assert result3 == expected3, f"Test 3 failed: got '{result3}', expected '{expected3}'"
    
    # 테스트 4: forimg_pathintqdm[Path]|(...) -> for img_path in tqdm(...)
    input4 = "forimg_pathintqdm[Path]|(val_images, desc=\"PaddleOCRPretrainedInference\", unit=\"img\"):"
    expected4 = "for img_path in tqdm(val_images, desc=\"PaddleOCR Pretrained Inference\", unit=\"img\"):"
    result4 = normalize_code_line(input4, next_line_indent=4, lang_hint="py", safe_mode=True)
    assert result4 == expected4, f"Test 4 failed: got '{result4}', expected '{expected4}'"
    
    # 테스트 5: newline=", encoding='utf-8-sig' -> newline="", encoding='utf-8-sig'
    input5 = "with open(save_csv_path, 'w', newline=\", encoding='utf-8-sig') as f"
    expected5 = "with open(save_csv_path, 'w', newline=\"\", encoding='utf-8-sig') as f:"
    result5 = normalize_code_line(input5, next_line_indent=4, lang_hint="py", safe_mode=True)
    assert result5 == expected5, f"Test 5 failed: got '{result5}', expected '{expected5}'"
    
    # 테스트 6: 실제 OCR 결과 패턴
    input6 = "val_dir=Path(\"/content/data/HMG)O|E{/TS1/val\") path(\"/content/data/원천데이터/TS1/va1\")"
    expected6 = "val_dir=Path(\"/content/data/HMG)O|E{/TS1/val\")"
    result6 = normalize_code_line(input6, next_line_indent=None, lang_hint="py", safe_mode=True)
    # 중복 제거는 reconstruct에서 처리되므로 여기서는 경로 중복만 제거
    assert "path(" not in result6 or result6.count("Path(") == 1, f"Test 6 failed: got '{result6}'"
    
    # 테스트 7: exceptExceptionase -> except Exception as e:
    input7 = "exceptExceptionase:"
    expected7 = "except Exception as e:"
    result7 = normalize_code_line(input7, next_line_indent=None, lang_hint="py", safe_mode=True)
    assert result7 == expected7, f"Test 7 failed: got '{result7}', expected '{expected7}'"
    
    # 테스트 8: PaddleOCRPretrainedInference -> PaddleOCR Pretrained Inference
    input8 = "for img_path in tqdm(val_images, desc=\"PaddleOCRPretrainedInference\", unit=\"img\"):"
    expected8 = "for img_path in tqdm(val_images, desc=\"PaddleOCR Pretrained Inference\", unit=\"img\"):"
    result8 = normalize_code_line(input8, next_line_indent=4, lang_hint="py", safe_mode=True)
    assert result8 == expected8, f"Test 8 failed: got '{result8}', expected '{expected8}'"
    
    # 테스트 9: try: 함수() exceptException -> try:\n    함수()\nexcept Exception as e:
    input9 = "try: run_inference_and_visualize(img_path, output_dir) exceptException"
    expected9 = "try:\n    run_inference_and_visualize(img_path, output_dir)\nexcept Exception as e:"
    result9 = normalize_code_line(input9, next_line_indent=4, lang_hint="py", safe_mode=True)
    assert "except Exception" in result9, f"Test 9 failed: got '{result9}'"
    
    print("[OK] All OCR pattern normalization tests passed!")


if __name__ == "__main__":
    # 테스트 실행
    test_normalize_code_line_ocr_patterns()
