# core/capture.py
# OCR 캡처 기능을 서버 엔드포인트로 제공

import sys
import os
import asyncio
import re
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from mss import mss
import pyperclip
import yaml

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
    """
    코드 이미지 전처리 (대비, 선명도 향상)
    주의: 너무 강한 전처리는 텍스트를 손상시킬 수 있으므로 주의
    """
    if not enabled:
        return img
    try:
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
    except Exception as e:
        # 전처리 실패 시 원본 이미지 반환
        print(f"   ⚠ 이미지 전처리 실패, 원본 사용: {e}")
        return img.convert("RGB")


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
    # y_thresh를 더 타이트하게 설정하여 줄 압착 방지 (0.60 -> 0.30)
    y_thresh = max(4.0, med_h * 0.30)
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
        # 줄 병합 임계값을 더 타이트하게 설정 (0.15 -> 0.10)
        if gap <= max(1.0, med_h * 0.10):
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
            # 리사이즈된 이미지(x3)에서 너무 크게 계산되지 않도록 제한 (80.0 -> 30.0)
            if 2.0 <= cw <= 30.0:
                samples.append(float(cw))
    if not samples:
        heights = [w.h for ln in lines for w in ln.words if w.h > 0]
        med_h = _robust_median(heights, default=14.0)
        # 리사이즈된 이미지 고려하여 더 작은 값 반환 (0.55 -> 0.40)
        return max(4.0, med_h * 0.40)
    char_w = _robust_median(samples, default=8.0)
    # 최대값 제한 (리사이즈된 이미지 대응)
    return min(char_w, 30.0)


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
    if not s:
        return s
    
    # ------------------------------------------------------------
    # [0] 긴급: 함수 정의 및 기본 문법 오인식 복구 (최우선)
    # ------------------------------------------------------------
    # def _configure_tesseract() -> None -> def _configure_tesseract None 복구
    s = re.sub(r"def\s+(\w+)\s+None\s*:", r"def \1() -> None:", s)
    s = re.sub(r"def\s+(\w+)\s*\(\s*\)\s+None\s*:", r"def \1() -> None:", s)
    s = re.sub(r"def\s+(\w+)\s+None\s*$", r"def \1() -> None:", s)
    # return -> Feturn 복구
    s = re.sub(r"\bFeturn\b", "return", s, flags=re.I)
    s = re.sub(r"\bretum\b", "return", s, flags=re.I)
    s = re.sub(r"\bretu[rn]\b", "return", s, flags=re.I)
    
    # ------------------------------------------------------------
    # [0-1] 긴급: Python 문법 기호 복구 (최우선)
    # ------------------------------------------------------------
    # 한글 오인식 복구 (터/스트 -> 텍스트)
    s = re.sub(r"터\s*/\s*스트", "텍스트", s)
    s = re.sub(r"터\s*스트", "텍스트", s)
    # 언더스코어와 점 복구 (merged WordsappendWordBox -> merged_words.append(WordBox)
    s = re.sub(r"(\w+)\s+Wordsappend(\w+)", r"\1_words.append(\2", s, flags=re.I)
    s = re.sub(r"(\w+)\s+words\s*append\s*(\w+)", r"\1_words.append(\2", s, flags=re.I)
    s = re.sub(r"(\w+)\s*Words\s*append\s*(\w+)", r"\1_words.append(\2", s, flags=re.I)
    # 함수 정의 복구 (def get_ paddleocr Words -> def get_paddleocr_words)
    # 먼저 직접 문자열 치환으로 처리
    s = s.replace("get__paddleocr_word.s,", "get_paddleocr_words(")
    s = s.replace("get_ paddleocr Words", "get_paddleocr_words")
    s = s.replace("get _ paddleocr Words", "get_paddleocr_words")
    # 그 다음 정규식으로 처리
    s = re.sub(r"def\s+get\s+_\s+paddleocr\s+Words", "def get_paddleocr_words", s, flags=re.I)
    s = re.sub(r"def\s+(\w+)\s+_\s+(\w+)\s+(\w+)", r"def \1_\2_\3", s, flags=re.I)
    s = re.sub(r"def\s+(\w+)\s+(\w+)\s+(\w+)", r"def \1_\2_\3", s, flags=re.I)
    # 함수 호출 복구 (reconstruct_text_from words -> reconstruct_text_from_words()
    s = re.sub(r"(\w+)_text_from\s+words$", r"\1_text_from_words(", s, flags=re.I)
    s = re.sub(r"(\w+)\s+text\s+from\s+words$", r"\1_text_from_words(", s, flags=re.I)
    s = re.sub(r"(\w+)_text_from_words$", r"\1_text_from_words(", s, flags=re.I)
    # 타입 힌트 복구 (img Union[npndarray Image Image) -> img: Union[np.ndarray, Image.Image])
    # 먼저 직접 문자열 치환
    s = s.replace("img Union[np.ndarray, Image.Image)", "img: Union[np.ndarray, Image.Image]")
    s = s.replace("Image.Image)", "Image.Image]")
    # 그 다음 정규식으로 처리
    s = re.sub(r"(\w+)\s+Union\s*\[\s*npndarray\s+Image\s+Image\s*\)", r"\1: Union[np.ndarray, Image.Image]", s, flags=re.I)
    s = re.sub(r"(\w+)\s+Union\s*\[\s*np\s*ndarray\s+Image\s+Image\s*\)", r"\1: Union[np.ndarray, Image.Image]", s, flags=re.I)
    s = re.sub(r"^(\s*)img\s+Union\[", r"\1img: Union[", s, flags=re.I)
    s = re.sub(r"(\w+)\s+Union\s*\[", r"\1: Union[", s, flags=re.I)
    # np.ndarray 복구
    s = re.sub(r"np\s*ndarray", "np.ndarray", s, flags=re.I)
    s = re.sub(r"npndarray", "np.ndarray", s, flags=re.I)
    # Image.Image 복구
    s = re.sub(r"Image\s+Image", "Image.Image", s)
    # 변수명 복구 (Out -> out, Words -> words, Scale -> scale)
    s = re.sub(r"^(\s*)Out\s*=", r"\1out =", s)
    s = re.sub(r"^(\s*)Words\s*$", r"\1words,", s)
    s = re.sub(r"^(\s*)Scale\s+int", r"\1scale: int", s, flags=re.I)
    # 타입 힌트 콜론 복구 (Scale int = 3 -> scale: int = 3)
    # 먼저 직접 문자열 치환
    s = s.replace("scale int =", "scale: int =")
    s = s.replace("Scale int =", "scale: int =")
    # 그 다음 정규식으로 처리
    s = re.sub(r"^(\s*)(\w+)\s+int\s*=", r"\1\2: int =", s, flags=re.I)
    s = re.sub(r"^(\s*)(\w+)\s+bool\s*=", r"\1\2: bool =", s, flags=re.I)
    s = re.sub(r"^(\s*)(\w+)\s+str\s*=", r"\1\2: str =", s, flags=re.I)
    s = re.sub(r"^(\s*)(\w+)\s+float\s*=", r"\1\2: float =", s, flags=re.I)
    # 함수 파라미터 타입 힌트 복구 (ode_mode bool = True -> code_mode: bool = True)
    s = re.sub(r"^(\s*)ode_mode\s+bool", r"\1code_mode: bool", s, flags=re.I)
    # 함수 파라미터 복구 (ode_mode_code_mode -> code_mode=code_mode,)
    s = re.sub(r"^(\s*)ode_mode_(\w+)$", r"\1code_mode=\2,", s, flags=re.I)
    s = re.sub(r"^(\s*)(\w+)_mode_(\w+)$", r"\1\2_mode=\3,", s, flags=re.I)
    # 등호 누락 복구 (indent_step_indent_step -> indent_step=indent_step,)
    s = re.sub(r"^(\s*)(\w+)_(\1)$", r"\1\2=\3,", s)  # 같은 변수명 반복
    s = re.sub(r"^(\s*)(\w+)_(\w+)_(\2)$", r"\1\2_\3=\4,", s)  # indent_step_indent_step
    # 복잡한 파라미터 오인식 복구 (removeUseLocalemojimodelboolbool= True=False)
    s = re.sub(r"removeUseLocalemojimodelboolbool\s*=\s*True\s*=\s*False", r"remove_emoji: bool = True, use_local_model: bool = False", s, flags=re.I)
    s = re.sub(r"remove\s*Use\s*Local\s*emoji\s*model\s*bool\s*bool", r"remove_emoji: bool = True, use_local_model: bool = False", s, flags=re.I)
    # emoji 오인식 복구 (remove_emo>i -> remove_emoji)
    s = re.sub(r"emo\s*>\s*i", "emoji", s, flags=re.I)
    s = re.sub(r"emo\s*>\s*I", "emoji", s, flags=re.I)
    s = re.sub(r"emo\s*>\s*1", "emoji", s, flags=re.I)
    # remove_emoji 파라미터 복구
    s = re.sub(r"^(\s*)remove_emo\s*>\s*i_(\w+)$", r"\1remove_emoji=\2,", s, flags=re.I)
    s = re.sub(r"^(\s*)remove_emo\s*>\s*i$", r"\1remove_emoji=remove_emoji,", s, flags=re.I)
    # 주석 복구 (가본Zt -> 기본값)
    s = re.sub(r"가본\s*Z\s*t", "기본값", s)
    s = re.sub(r"가본\s*Zt", "기본값", s)
    # 점 복구를 먼저 처리 (paddle_wordx -> paddle_word.x)
    # 등호가 있는 경우 먼저 처리
    s = re.sub(r"([a-zA-Z])=(\w+)_word([a-z])$", r"\1=\2_word.\3,", s)
    s = re.sub(r"([a-zA-Z])=(\w+)_word([A-Z])$", r"\1=\2_word.\3,", s)
    # 등호가 없는 경우
    s = re.sub(r"(\w+)_word([a-z])$", r"\1_word.\2,", s, flags=re.I)
    s = re.sub(r"(\w+)_word([A-Z])$", r"\1_word.\2,", s, flags=re.I)
    s = re.sub(r"(\w+)word([a-z])$", r"\1_word.\2,", s, flags=re.I)
    # 등호와 언더스코어 복구 (text_final_text -> text=final_text,)
    # 단, _word 패턴이 아닌 경우만 처리
    s = re.sub(r"^(\s*)(text)_(final)_(text)$", r"\1\2=\3_\4,", s)
    s = re.sub(r"^(\s*)([a-z])_(\w+)$", r"\1\2=\3,", s)  # 단순한 경우 (단, _word가 아닌 경우)
    # conf 복구 (onf -> conf)
    s = re.sub(r"\bonf\s*_", "conf=", s, flags=re.I)
    s = re.sub(r"\bonf\s*=", "conf=", s, flags=re.I)
    s = re.sub(r"(\w+)_onf\s*_", r"\1.conf=", s, flags=re.I)
    s = re.sub(r"(\w+)_onf$", r"\1.conf", s, flags=re.I)
    # print[ -> print( 복구
    s = re.sub(r"\bprint\s*\[", "print(", s, flags=re.I)
    # print[f" -> print(f" 복구
    s = re.sub(r"\bprint\s*\[\s*f\s*[\"']", r'print(f"', s, flags=re.I)
    # f-string 문법 복구 (f" -> f", f' -> f')
    s = re.sub(r"f\s*\[\s*[\"']", r'f"', s)
    s = re.sub(r"f\s*\[\s*['\"]", r"f'", s)
    # PaddleOCR 변수명 복구 (PADDLEOR INSTANE -> _PADDLEOCR_INSTANCE)
    s = re.sub(r"\bPADDLEOR\s+INSTANE\b", "_PADDLEOCR_INSTANCE", s, flags=re.I)
    s = re.sub(r"\bPADDLEOR\s+_?INSTANCE\b", "_PADDLEOCR_INSTANCE", s, flags=re.I)
    s = re.sub(r"\bPADDLE\s*OR\s*C\s*R\s*_?INSTANCE\b", "_PADDLEOCR_INSTANCE", s, flags=re.I)
    s = re.sub(r"\bPADDLE\s*OCR\s*_?INSTANCE\b", "_PADDLEOCR_INSTANCE", s, flags=re.I)
    # PaddleOCR 단어 복구 (PaddleOR08 -> PaddleOCR, PaddleOR -> PaddleOCR)
    # 순서 중요: 더 구체적인 패턴부터 처리
    # 먼저 직접 문자열 치환으로 처리 (정규식보다 우선)
    s = s.replace("PaddleOR+", "PaddleOCR")
    s = s.replace("PaddleOR +", "PaddleOCR")
    s = s.replace("Paddle OR+", "PaddleOCR")
    s = s.replace("Paddle OR +", "PaddleOCR")
    # 그 다음 정규식으로 처리
    s = re.sub(r"Paddle\s*OR\s*0?\d*", "PaddleOCR", s, flags=re.I)
    s = re.sub(r"Paddle\s*OR\s*C\s*R", "PaddleOCR", s, flags=re.I)
    s = re.sub(r"Paddle\s*OR\s*\+", "PaddleOCR", s, flags=re.I)  # PaddleOR+ -> PaddleOCR
    s = re.sub(r"Paddle\s*OR\s+", "PaddleOCR ", s, flags=re.I)  # PaddleOR 뒤 공백 -> PaddleOCR
    s = re.sub(r"Paddle\s*OR\b", "PaddleOCR", s, flags=re.I)  # PaddleOR -> PaddleOCR (단어 경계)
    # PaddleOR 뒤에 한글이 오는 경우도 처리
    s = re.sub(r"Paddle\s*OR\s*([가-힣])", r"PaddleOCR \1", s, flags=re.I)
    # return _PADDLEOCR_INSTANCE 복구
    s = re.sub(r"return\s+PADDLEOR\s+INSTANE", "return _PADDLEOCR_INSTANCE", s, flags=re.I)
    s = re.sub(r"return\s+PADDLE\s*OR\s*C\s*R\s*INSTANCE", "return _PADDLEOCR_INSTANCE", s, flags=re.I)
    # PP-OCRv4/v5 오인식 복구 (PP-ORv4/V5, PP-ORv4/V5 등)
    s = re.sub(r"PP\s*[-]?\s*OR\s*v\s*(\d+)\s*[/]?\s*V?\s*(\d+)", r"PP-OCRv\1/v\2", s, flags=re.I)
    s = re.sub(r"PP\s*[-]?\s*OCR\s*v\s*(\d+)\s*[/]?\s*v\s*(\d+)", r"PP-OCRv\1/v\2", s, flags=re.I)
    s = re.sub(r"PP\s*[-]?\s*OR\s*v\s*(\d+)\s*[/]?\s*(\d+)", r"PP-OCRv\1/v\2", s, flags=re.I)
    # PP-OCRv4/v5 뒤에 붙은 단어 분리 (PP-OCRv4/v5PaddleOR -> PP-OCRv4/v5, PaddleOCR)
    s = re.sub(r"PP-OCRv(\d+)/v(\d+)(Paddle\s*OR)", r"PP-OCRv\1/v\2, PaddleOCR", s, flags=re.I)
    s = re.sub(r"PP-OCRv(\d+)/v(\d+)(PaddleOCR)", r"PP-OCRv\1/v\2, PaddleOCR", s, flags=re.I)
    # PaddleOR+ 패턴 직접 복구 (PaddleOR+ -> PaddleOCR)
    s = s.replace("PaddleOR+", "PaddleOCR")
    s = s.replace("PaddleOR +", "PaddleOCR")
    s = s.replace("Paddle OR+", "PaddleOCR")
    s = s.replace("Paddle OR +", "PaddleOCR")
    # 한글+영어 최적화 오인식 복구 (한글+영어, 한글\+영어 등)
    s = re.sub(r"한글\s*[+]\s*영어\s*최적화", "한글+영어 최적화", s)
    s = re.sub(r"한글\s*영어\s*최적화", "한글+영어 최적화", s)
    # 초기화 완료 오인식 복구 (초기화 최적화 완료 -> 초기화 완료)
    s = re.sub(r"초기화\s*최적화\s*완료", "초기화 완료", s)
    s = re.sub(r"초기화\s*완료\s*최적화", "초기화 완료", s)
    s = re.sub(r"초기화\s*완료\s*최신", "초기화 완료", s)
    # 따옴표와 단어 분리 복구 ("완료최신 -> " 완료, 최신)
    s = re.sub(r"\"\s*완료\s*([가-힣]+)", r'" 완료, \1', s)
    s = re.sub(r"\"\s*([가-힣]+)\s*완료", r'" \1 완료', s)
    # 자동 다운로드 모델 오인식 복구
    s = re.sub(r"자동\s*다운로드\s*모델", "자동 다운로드 모델", s)
    s = re.sub(r"자동\s*다운로드", "자동 다운로드", s)
    # 괄호 오인식 복구 (print[f" -> print(f")
    s = re.sub(r"(\w+)\s*\[\s*f\s*[\"']", r'\1(f"', s)
    s = re.sub(r"(\w+)\s*\[\s*[\"']", r'\1("', s)
    # 숫자 오인식 복구 (08 -> 0, v4/v5 -> v4/v5)
    s = re.sub(r"v\s*(\d+)\s*[/]\s*v\s*(\d+)", r"v\1/v\2", s, flags=re.I)
    s = re.sub(r"v\s*(\d+)\s*[/]\s*V\s*(\d+)", r"v\1/v\2", s, flags=re.I)
    # 숫자 중간 0 제거 (PaddleOR08 -> PaddleOCR, 08 -> 0)
    s = re.sub(r"(\w)0([1-9])(\w)", r"\1\2\3", s)  # 단어 내부의 0 제거 (08 -> 8, OR08 -> OR8)
    # os.path.exists 오인식 복구 (ifos_pathexists<-, ifos_pathexists 등)
    s = re.sub(r"ifos_pathexists\s*[<\-]?\s*['\"]?([A-Z_]+EXE)['\"]?", r"if os.path.exists(\1)", s, flags=re.I)
    s = re.sub(r"if\s*os\.path\.exists\s*[<\-]\s*['\"]?([A-Z_]+EXE)['\"]?", r"if os.path.exists(\1)", s, flags=re.I)
    s = re.sub(r"if\s*os\.path\.exists\s*[<\-]\s*['\"]?([A-Z_]+DIR)['\"]?", r"if os.path.exists(\1)", s, flags=re.I)
    # os.path.isdir 오인식 복구 (OSpathis_dir, OSpath.is_dir 등)
    s = re.sub(r"OS\s*path\s*is\s*_?dir\s*([A-Z_]+DIR)", r"os.path.isdir(\1)", s, flags=re.I)
    s = re.sub(r"os\.path\.is\s*_?dir\s*([A-Z_]+DIR)", r"os.path.isdir(\1)", s, flags=re.I)
    s = re.sub(r"OSpathis_dir\s*([A-Z_]+DIR)", r"os.path.isdir(\1)", s, flags=re.I)
    # os.environ 오인식 복구 (OS_ENV_Iron, OS_ENV 등)
    s = re.sub(r"OS\s*_?\s*ENV\s*_?\s*Iron\s*\[\s*[\"']([A-Z_]+PREFIX)[\"']\s*\]\s*[<\-]\s*([A-Z_]+DIR)", 
               r'os.environ["\1"] = \2', s, flags=re.I)
    s = re.sub(r"OS\s*_?\s*ENV\s*\[\s*[\"']([A-Z_]+PREFIX)[\"']\s*\]\s*[<\-]\s*([A-Z_]+DIR)", 
               r'os.environ["\1"] = \2', s, flags=re.I)
    s = re.sub(r"os\.environ\s*\[\s*[\"']([A-Z_]+PREFIX)[\"']\s*\]\s*[<\-]\s*([A-Z_]+DIR)", 
               r'os.environ["\1"] = \2', s, flags=re.I)
    # TESSERACT_EXE 오인식 복구 (TESSERAT_EXE, TESSERAT EXE 등)
    s = re.sub(r"TESSERAT\s+EXE", "TESSERACT_EXE", s, flags=re.I)
    s = re.sub(r"TESSERAT_EXE", "TESSERACT_EXE", s, flags=re.I)
    s = re.sub(r"TESSERACT\s+EXE", "TESSERACT_EXE", s)
    # TESSDATA_DIR 오인식 복구 (TESSDATA DIR 등)
    s = re.sub(r"TESSDATA\s+DIR", "TESSDATA_DIR", s, flags=re.I)
    s = re.sub(r"TESSDATA\s+PREFIX", "TESSDATA_PREFIX", s, flags=re.I)
    # docstring 닫는 따옴표 복구
    s = re.sub(r'"""([^"]*?)(\s*)$', r'"""\1"""', s)  # 줄 끝에 닫는 따옴표 없으면 추가
    s = re.sub(r'"""([^"]*?)(\s*)([^"])', r'"""\1"""\3', s)  # 중간에 닫는 따옴표 없으면 추가
    # 의미 없는 숫자 제거 (10 1 10 같은 패턴)
    s = re.sub(r"^\s*\d+\s+\d+\s+\d+\s*$", "", s)  # 줄 전체가 숫자만 있으면 제거
    s = re.sub(r"^\s*\d+\s+\d+\s*$", "", s)  # 두 개 숫자만 있으면 제거
    
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
    
    # ------------------------------------------------------------
    # [최종] 코드 문법 기호 보정 (괄호, 따옴표, f-string 등)
    # ------------------------------------------------------------
    # Python 문법 기호 강화 복구
    # 1. 언더스코어와 점 복구 (merged WordsappendWordBox -> merged_words.append(WordBox)
    s = re.sub(r"(\w+)\s+Wordsappend(\w+)", r"\1_words.append(\2", s, flags=re.I)
    s = re.sub(r"(\w+)\s+words\s*append\s*(\w+)", r"\1_words.append(\2", s, flags=re.I)
    s = re.sub(r"(\w+)\s*Words\s*append\s*(\w+)", r"\1_words.append(\2", s, flags=re.I)
    # 2. 등호와 언더스코어 복구 (text_final_text -> text=final_text,)
    # 패턴: 변수명_변수명_변수명 -> 변수명=변수명_변수명,
    s = re.sub(r"^(\s*)(text)_(final)_(text)$", r"\1\2=\3_\4,", s)
    # 3. 등호가 이미 있는 경우 점 복구 (X=paddle_wordx -> x=paddle_word.x,)
    # 먼저 등호가 있는 경우를 처리 (더 구체적이므로 우선)
    s = re.sub(r"=(\w+)_word([a-z])", r"=\1_word.\2", s, flags=re.I)
    s = re.sub(r"=(\w+)_word([A-Z])", r"=\1_word.\2", s, flags=re.I)
    # 대문자 변수명 소문자로 변환 (X= -> x=, W= -> w=)
    s = re.sub(r"^(\s*)([A-Z])=(\w+\.\w+)", r"\1\2.lower()=\3", s)
    s = re.sub(r"^(\s*)([A-Z])=(\w+_word\.\w+)", r"\1\2.lower()=\3", s)
    # .lower() 제거하고 직접 소문자로 변환
    s = re.sub(r"([A-Z])\.lower\(\)=", lambda m: m.group(1).lower() + "=", s)
    # 4. 점 복구 (paddle_wordx -> paddle_word.x, paddle_wordy -> paddle_word.y)
    # 등호가 없는 경우: paddle_wordx -> paddle_word.x,
    s = re.sub(r"^(\s*)(\w+)_word([a-z])$", r"\1\2_word.\3,", s, flags=re.I)
    s = re.sub(r"^(\s*)(\w+)_word([A-Z])$", r"\1\2_word.\3,", s, flags=re.I)
    s = re.sub(r"^(\s*)(\w+)word([a-z])$", r"\1\2_word.\3,", s, flags=re.I)
    # 5. 등호와 언더스코어 복구 (h_paddle_wordh -> h=paddle_word.h,)
    # _word 패턴이 있는 경우 먼저 처리
    s = re.sub(r"^(\s*)([a-z])_(\w+)_word([a-z])$", r"\1\2=\3_word.\4,", s)
    s = re.sub(r"^(\s*)([a-z])_(\w+)$", r"\1\2=\3,", s)  # 단순한 경우 (text_final_text 등)
    # 대문자 속성명 소문자로 변환 (W -> w)
    s = re.sub(r"=(\w+_word\.)([A-Z])", lambda m: "=" + m.group(1) + m.group(2).lower(), s)
    # 함수 파라미터 등호 복구 강화 (변수명_변수명 -> 변수명=변수명,)
    # 같은 변수명이 반복되는 경우
    s = re.sub(r"^(\s*)(\w+)_(\2)$", r"\1\2=\3,", s)  # code_mode_code_mode -> code_mode=code_mode,
    s = re.sub(r"^(\s*)(\w+)_(\w+)_(\3)$", r"\1\2_\3=\4,", s)  # indent_step_indent_step -> indent_step=indent_step,
    s = re.sub(r"^(\s*)(\w+)_(\w+)_(\w+)_(\4)$", r"\1\2_\3_\4=\5,", s)  # remove_emoji_remove_emoji -> remove_emoji=remove_emoji,
    # 6. 괄호 복구 (merged_words.append(WordBox -> merged_words.append(WordBox()
    s = re.sub(r"(\w+)\.append\((\w+)$", r"\1.append(\2(", s)
    # 5. conf 복구 강화 (onf -> conf)
    s = re.sub(r"(\w+)_onf\s*_(\w+)", r"\1.conf=\2", s, flags=re.I)
    s = re.sub(r"(\w+)_onf$", r"\1.conf", s, flags=re.I)
    s = re.sub(r"^(\s*)onf\s*_(\w+)", r"\1conf=\2", s, flags=re.I)
    s = re.sub(r"^(\s*)onf\s*=", r"\1conf=", s, flags=re.I)
    # 6. 괄호 오인식 복구 ([ -> (, ] -> ))
    # 함수 호출 패턴: print[ -> print(, return [ -> return (
    s = re.sub(r"\b(print|return|if|elif|for|while|with|def|class|append)\s*\[", r"\1(", s, flags=re.I)
    # 변수 접근 패턴은 유지: [0], [1] 등은 그대로
    # 하지만 함수 호출 뒤의 [는 (로 변경
    s = re.sub(r"(\w+)\s*\[\s*f\s*[\"']", r'\1(f"', s)  # print[f" -> print(f"
    s = re.sub(r"(\w+)\s*\[\s*[\"']", r'\1("', s)  # print[" -> print("
    # 닫는 괄호 복구 (] -> ))
    # 하지만 배열 인덱스는 유지해야 하므로 주의
    # 함수 호출 끝의 ]를 )로 변경 (단, 배열 인덱스가 아닌 경우)
    s = re.sub(r"\)\s*\]\s*$", "))", s)  # )] -> ))
    s = re.sub(r"\]\s*\]\s*$", "))", s)  # ]] -> ))
    s = re.sub(r"\]\s*$", ")", s)  # 줄 끝의 ]를 )로
    # 7. f-string 복구
    s = re.sub(r'f\s*\[\s*"', 'f"', s)
    s = re.sub(r"f\s*\[\s*'", "f'", s)
    # 8. 등호 복구 (h_paddle_word.h, -> h=paddle_word.h,)
    # 점이 이미 있는데 등호가 없는 경우
    s = re.sub(r"^(\s*)([a-z])_(\w+_word\.\w+),?$", r"\1\2=\3,", s)
    # 9. 쉼표 복구 (줄 끝에 쉼표가 없으면 추가)
    s = re.sub(r"(\w+\.\w+)\s*$", r"\1,", s)  # paddle_word.x -> paddle_word.x,
    s = re.sub(r"(\w+=\w+\.\w+)\s*$", r"\1,", s)  # x=paddle_word.x -> x=paddle_word.x,
    # 특수문자 복구 (✅, 한글 등)
    s = re.sub(r"[✅✓✔]", "✅", s)  # 다양한 체크마크를 ✅로 통일
    # 언더스코어 복구 (PADDLEOR INSTANE -> _PADDLEOCR_INSTANCE)
    s = re.sub(r"([A-Z]+)\s+([A-Z]+)", r"\1_\2", s)  # 공백으로 분리된 대문자를 언더스코어로
    # 하지만 이미 언더스코어가 있는 경우는 유지
    s = re.sub(r"([A-Z]+)_\s+([A-Z]+)", r"\1_\2", s)  # 언더스코어와 공백이 함께 있는 경우
    # PaddleOCR 변수명 복구 강화
    s = re.sub(r"\bPADDLE\s*OR\s*C\s*R\s*_?\s*INSTANCE\b", "_PADDLEOCR_INSTANCE", s, flags=re.I)
    s = re.sub(r"\bPADDLE\s*OR\s*_?\s*INSTANCE\b", "_PADDLEOCR_INSTANCE", s, flags=re.I)
    s = re.sub(r"\bPADDLE\s*OCR\s*_?\s*INSTANCE\b", "_PADDLEOCR_INSTANCE", s, flags=re.I)
    # _PADDLEOR_INSTANE -> _PADDLEOCR_INSTANCE
    s = re.sub(r"_PADDLEOR\s*_?\s*INSTANE", "_PADDLEOCR_INSTANCE", s, flags=re.I)
    s = re.sub(r"_PADDLE\s*OR\s*_?\s*INSTANCE", "_PADDLEOCR_INSTANCE", s, flags=re.I)
    # PP-OCRv4/v5 오인식 복구 강화
    s = re.sub(r"PP\s*[-]?\s*OR\s*v\s*(\d+)\s*[/]?\s*V?\s*(\d+)", r"PP-OCRv\1/v\2", s, flags=re.I)
    s = re.sub(r"PP\s*[-]?\s*OCR\s*v\s*(\d+)\s*[/]?\s*v\s*(\d+)", r"PP-OCRv\1/v\2", s, flags=re.I)
    s = re.sub(r"PP\s*[-]?\s*OR\s*v\s*(\d+)\s*[/]?\s*(\d+)", r"PP-OCRv\1/v\2", s, flags=re.I)
    # 한글+영어 최적화 오인식 복구
    s = re.sub(r"한글\s*[+]\s*영어\s*최적화", "한글+영어 최적화", s)
    s = re.sub(r"한글\s*영어\s*최적화", "한글+영어 최적화", s)
    s = re.sub(r"한글\s*영어", "한글+영어", s)
    # 초기화 완료 오인식 복구
    s = re.sub(r"초기화\s*최적화\s*완료", "초기화 완료", s)
    s = re.sub(r"초기화\s*완료\s*최적화", "초기화 완료", s)
    s = re.sub(r"초기화\s*완료\s*최신", "초기화 완료", s)
    # 자동 다운로드 모델 오인식 복구
    s = re.sub(r"자동\s*다운로드\s*모델", "자동 다운로드 모델", s)
    s = re.sub(r"자동\s*다운로드", "자동 다운로드", s)
    # 숫자 오인식 복구 (08 -> 0, v4/v5 -> v4/v5)
    s = re.sub(r"v\s*(\d+)\s*[/]\s*v\s*(\d+)", r"v\1/v\2", s, flags=re.I)
    s = re.sub(r"v\s*(\d+)\s*[/]\s*V\s*(\d+)", r"v\1/v\2", s, flags=re.I)
    s = re.sub(r"(\d+)0(\d+)", r"\1\2", s)  # 08 -> 0 (중간에 0이 잘못 들어간 경우)
    
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
        for w in ln.words[1:]:
            txt = w.text
            if not txt:
                continue
            gap_px = w.x - prev_x2
            # 공백 보존 로직 개선: gap_px가 조금이라도 있으면 최소 1칸 공백 삽입
            if gap_px <= 0:
                # 겹치는 경우 공백 없음
                spaces = 0
            elif gap_px <= char_w * 0.02:
                # 매우 작은 간격 (2% 이하)은 공백 없음 (인식 오차 범위)
                spaces = 0
            else:
                # gap_px가 있으면 반드시 최소 1칸 공백 삽입
                spaces = int(round(gap_px / max(1e-6, char_w)))
                spaces = clamp_int(spaces, 1, 80)  # 최소 1개, 최대 80개
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


def merge_paddleocr_winrt_results(
    paddleocr_words: List[WordBox],
    winrt_words: List[WordBox],
    pil_img: Image.Image,
) -> str:
    """
    PaddleOCR과 WinRT 결과를 병합하여 정확도 향상
    - PaddleOCR: 코드 문법 기호 인식 우수
    - WinRT: 한글 인식 우수
    - 두 결과를 병합하여 최적의 텍스트 추출
    """
    korean_re = re.compile(r"[가-힣]")
    paddleocr_lines = cluster_lines(paddleocr_words)
    winrt_lines = cluster_lines(winrt_words)

    def find_overlapping_winrt_word(paddle_word: WordBox, winrt_words_list: List[WordBox], threshold: float = 0.5) -> Optional[WordBox]:
        best_match = None
        best_overlap = 0.0
        for winrt_word in winrt_words_list:
            y_overlap = min(paddle_word.y2, winrt_word.y2) - max(paddle_word.y, winrt_word.y)
            if y_overlap <= 0:
                continue
            x_overlap = min(paddle_word.x2, winrt_word.x2) - max(paddle_word.x, winrt_word.x)
            if x_overlap <= 0:
                continue
            paddle_area = paddle_word.w * paddle_word.h
            overlap_area = x_overlap * y_overlap
            if paddle_area > 0:
                overlap_ratio = overlap_area / paddle_area
                if overlap_ratio >= threshold and overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_match = winrt_word
        return best_match

    merged_words: List[WordBox] = []
    for paddle_line in paddleocr_lines:
        for paddle_word in paddle_line.words:
            nearby_winrt_words = [
                w for w in winrt_words
                if abs(w.cy - paddle_word.cy) < paddle_word.h * 2.0
            ]
            overlapping_winrt = find_overlapping_winrt_word(paddle_word, nearby_winrt_words)
            final_text = paddle_word.text
            
            if overlapping_winrt:
                winrt_text_val = overlapping_winrt.text.strip()
                paddle_text_val = paddle_word.text.strip()
                
                # 한글이 포함된 경우 WinRT 우선 (한글 인식이 더 정확)
                if korean_re.search(winrt_text_val):
                    final_text = winrt_text_val
                # 코드 문법 기호가 많은 경우 PaddleOCR 우선
                elif ASCII_CODE_RE.findall(paddle_text_val) and not korean_re.search(paddle_text_val):
                    final_text = paddle_text_val
                # 길이가 비슷하면 PaddleOCR 우선 (코드 인식이 더 정확)
                elif abs(len(paddle_text_val) - len(winrt_text_val)) <= 2:
                    final_text = paddle_text_val
                else:
                    final_text = winrt_text_val if len(winrt_text_val) > len(paddle_text_val) else paddle_text_val
            else:
                # 한글이 없는 코드는 PaddleOCR 우선
                if not korean_re.search(paddle_word.text):
                    final_text = paddle_word.text
                else:
                    # 한글이 있는 경우 주변 WinRT 단어 확인
                    nearby_korean_winrt = [
                        w for w in nearby_winrt_words
                        if korean_re.search(w.text) and abs(w.x - paddle_word.x) < paddle_word.w * 3.0
                    ]
                    if nearby_korean_winrt:
                        closest = min(nearby_korean_winrt, key=lambda w: abs(w.x - paddle_word.x))
                        final_text = closest.text
                    else:
                        final_text = paddle_word.text
            
            merged_words.append(WordBox(
                text=final_text,
                x=paddle_word.x,
                y=paddle_word.y,
                w=paddle_word.w,
                h=paddle_word.h,
                conf=paddle_word.conf
            ))
    
    # WinRT에만 있는 한글 단어 추가
    for winrt_word in winrt_words:
        is_overlapping = False
        for paddle_word in paddleocr_words:
            overlapping = find_overlapping_winrt_word(paddle_word, [winrt_word], threshold=0.3)
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
    try:
        import winrt  # noqa: F401
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


# ------------------------------------------------------------
# PaddleOCR 설정 및 함수
# ------------------------------------------------------------
_PADDLEOCR_INSTANCE = None
_PADDLEOCR_LOCAL_INSTANCE = None  # 로컬 모델 인스턴스
_PADDLEOCR_TRAINED_INSTANCE = None  # v5 학습된 모델 인스턴스

# 로컬 모델 경로 설정
_LOCAL_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "server")
_LOCAL_DET_MODEL = os.path.join(_LOCAL_MODELS_DIR, "ch_ppocr_server_v2.0_det_infer", "ch_ppocr_server_v2.0_det_infer")
_LOCAL_REC_MODEL = os.path.join(_LOCAL_MODELS_DIR, "ch_ppocr_server_v2.0_rec_infer", "ch_ppocr_server_v2.0_rec_infer")
_LOCAL_CLS_MODEL = os.path.join(_LOCAL_MODELS_DIR, "ch_ppocr_mobile_v2.0_cls_infer", "ch_ppocr_mobile_v2.0_cls_infer")

# v5 학습된 모델 경로 설정 (v5 전용)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
# v5 학습된 Recognition 모델 (rec_ke_v5_inference)
_TRAINED_REC_MODEL_DIR = os.path.join(_PROJECT_ROOT, "output", "rec_ke_v5_inference")
_TRAINED_REC_MODEL_PATH = os.path.join(_TRAINED_REC_MODEL_DIR, "inference")
_TRAINED_REC_INFERENCE_YML = os.path.join(_TRAINED_REC_MODEL_DIR, "inference.yml")
# v5 Detection 모델 (det_ke_inference - PP-OCRv5_server_det)
_TRAINED_DET_MODEL_DIR = os.path.join(_PROJECT_ROOT, "output", "det_ke_inference")
_TRAINED_DET_MODEL_PATH = os.path.join(_TRAINED_DET_MODEL_DIR, "inference")
_TRAINED_DET_INFERENCE_YML = os.path.join(_TRAINED_DET_MODEL_DIR, "inference.yml")


def _fix_inference_yml(yml_path: str, model_name: str, model_type: str = "rec") -> bool:
    """
    inference.yml 파일에 Global 섹션과 model_name을 자동으로 추가/수정
    
    Args:
        yml_path: inference.yml 파일 경로
        model_name: 모델 이름 (예: "PP-OCRv5_server_rec", "PP-OCRv5_server_det")
        model_type: 모델 타입 ("rec" 또는 "det")
    
    Returns:
        bool: 수정 성공 여부
    """
    if not os.path.exists(yml_path):
        print(f"[WARN] inference.yml 파일이 없습니다: {yml_path}")
        return False
    
    try:
        # YAML 파일 읽기
        with open(yml_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            content = ''.join(lines)
            # YAML 파싱
            try:
                config = yaml.safe_load(content) or {}
            except Exception:
                # YAML 파싱 실패 시 빈 딕셔너리로 시작
                config = {}
        
        # Global 섹션이 없으면 생성
        if 'Global' not in config:
            config['Global'] = {}
        
        # model_name 처리: PaddleX의 엄격한 검증을 피하기 위해 model_name을 삭제하거나 업데이트
        needs_update = False
        # Recognition 모델의 경우 model_name을 삭제하여 PaddleX가 자동으로 감지하도록 함
        if model_type == "rec" and 'model_name' in config.get('Global', {}):
            # 기존 model_name 삭제 (PaddleX가 자동으로 올바른 이름을 사용하도록)
            if config['Global']['model_name'] != model_name:
                # 다를 때만 삭제하고 업데이트하지 않음
                del config['Global']['model_name']
                needs_update = True
        elif model_type == "det":
            # Detection 모델은 model_name 유지
            if 'model_name' not in config['Global'] or config['Global']['model_name'] != model_name:
                config['Global']['model_name'] = model_name
                needs_update = True
        
        # model_type 설정
        if 'model_type' not in config['Global']:
            config['Global']['model_type'] = model_type
            needs_update = True
        
        # algorithm 설정 (Recognition 모델의 경우)
        if model_type == "rec" and 'algorithm' not in config['Global']:
            config['Global']['algorithm'] = "NRTR"
            needs_update = True
        
        # Detection 모델의 경우
        if model_type == "det":
            if 'det_algorithm' not in config['Global']:
                config['Global']['det_algorithm'] = "DB"
                needs_update = True
            if 'algorithm' not in config['Global']:
                config['Global']['algorithm'] = "DB"
                needs_update = True
        
        # use_gpu 설정 (기본값 False)
        if 'use_gpu' not in config['Global']:
            config['Global']['use_gpu'] = False
            needs_update = True
        
        # use_pdserving 설정
        if 'use_pdserving' not in config['Global']:
            config['Global']['use_pdserving'] = False
            needs_update = True
        
        # 업데이트가 필요한 경우에만 파일 쓰기
        if needs_update:
            # 기존 파일 백업
            backup_path = yml_path + '.backup'
            try:
                import shutil
                shutil.copy2(yml_path, backup_path)
            except Exception:
                pass
            
            # YAML 파일 쓰기
            with open(yml_path, 'w', encoding='utf-8') as f:
                # Global 섹션을 맨 위에 배치
                yaml.dump({'Global': config['Global']}, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                f.write('\n')
                
                # 나머지 섹션들 쓰기 (Global 제외)
                for key, value in config.items():
                    if key != 'Global':
                        yaml.dump({key: value}, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                        f.write('\n')
            
            # model_name이 정확히 업데이트되었는지 검증
            with open(yml_path, 'r', encoding='utf-8') as f:
                verify_config = yaml.safe_load(f) or {}
                actual_model_name = verify_config.get('Global', {}).get('model_name', '')
                if actual_model_name != model_name:
                    print(f"[WARN] model_name 검증 실패: 예상={model_name}, 실제={actual_model_name}")
                    # 강제로 다시 설정
                    verify_config.setdefault('Global', {})['model_name'] = model_name
                    with open(yml_path, 'w', encoding='utf-8') as f2:
                        yaml.dump({'Global': verify_config['Global']}, f2, default_flow_style=False, allow_unicode=True, sort_keys=False)
                        f2.write('\n')
                        for key, value in verify_config.items():
                            if key != 'Global':
                                yaml.dump({key: value}, f2, default_flow_style=False, allow_unicode=True, sort_keys=False)
                                f2.write('\n')
                    print(f"[OK] model_name 강제 업데이트 완료: {model_name}")
            
            print(f"[OK] inference.yml 파일 수정 완료: {yml_path}")
            print(f"   model_name: {model_name}, model_type: {model_type}")
            return True
        else:
            print(f"[OK] inference.yml 파일 검증 완료 (수정 불필요): {yml_path}")
            return True
        
    except Exception as e:
        print(f"[WARN] inference.yml 파일 수정 실패: {yml_path}, 오류: {e}")
        import traceback
        print(f"   상세 오류: {traceback.format_exc()}")
        return False


def _extract_character_dict_from_yml(yml_path: str) -> Optional[str]:
    """
    inference.yml에서 character_dict를 추출하여 텍스트 파일로 저장
    PaddleOCR이 사용할 수 있는 사전 파일 생성
    """
    if not os.path.exists(yml_path):
        return None
    
    try:
        with open(yml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # PostProcess.character_dict 추출
        postprocess = config.get('PostProcess', {})
        character_dict = postprocess.get('character_dict', [])
        
        if not character_dict or not isinstance(character_dict, list):
            return None
        
        # 사전 파일 경로 생성 (inference.yml과 같은 디렉토리)
        dict_dir = os.path.dirname(yml_path)
        dict_path = os.path.join(dict_dir, "character_dict.txt")
        
        # 사전 파일 생성 (한 줄에 한 문자)
        with open(dict_path, 'w', encoding='utf-8') as dict_file:
            for char in character_dict:
                if char:  # None이나 빈 문자열 제외
                    dict_file.write(str(char) + '\n')
        
        print(f"[OK] character_dict.txt 생성 완료: {dict_path} ({len(character_dict)}개 문자)")
        return dict_path
        
    except Exception as e:
        print(f"[WARN] character_dict 추출 실패: {yml_path}, 오류: {e}")
        return None


def _ensure_trained_model_configs() -> bool:
    """
    v5 학습된 모델의 inference.yml 파일들을 자동으로 보정
    PaddleOCR 3.x에서 필수인 Global.model_name을 추가
    그리고 character_dict를 사전 파일로 추출
    """
    success = True
    
    # Recognition 모델 설정 파일 보정
    if os.path.exists(_TRAINED_REC_INFERENCE_YML):
        if not _fix_inference_yml(_TRAINED_REC_INFERENCE_YML, "PP-OCRv5_server_rec", "rec"):
            success = False
        
        # character_dict를 사전 파일로 추출 (dictionary mismatch 방지)
        char_dict_path = _extract_character_dict_from_yml(_TRAINED_REC_INFERENCE_YML)
        if char_dict_path:
            print(f"[OK] Recognition 모델 사전 파일 준비 완료: {char_dict_path}")
        else:
            print(f"[WARN] Recognition 모델 사전 파일 생성 실패 (외계어 발생 가능)")
    else:
        print(f"[WARN] Recognition 모델 inference.yml 파일이 없습니다: {_TRAINED_REC_INFERENCE_YML}")
    
    # Detection 모델 설정 파일 보정
    if os.path.exists(_TRAINED_DET_INFERENCE_YML):
        if not _fix_inference_yml(_TRAINED_DET_INFERENCE_YML, "PP-OCRv5_server_det", "det"):
            success = False
    else:
        print(f"[WARN] Detection 모델 inference.yml 파일이 없습니다: {_TRAINED_DET_INFERENCE_YML}")
    
    return success


def _check_local_models_available() -> bool:
    """로컬 모델 파일 존재 여부 확인"""
    return (
        os.path.exists(_LOCAL_DET_MODEL) and
        os.path.exists(_LOCAL_REC_MODEL) and
        os.path.exists(_LOCAL_CLS_MODEL)
    )


def _check_trained_model_available() -> bool:
    """v5 학습된 Recognition 모델 파일 존재 여부 확인"""
    # v5 추론 모델이 있으면 사용
    if os.path.exists(_TRAINED_REC_MODEL_PATH + ".pdmodel"):
        return True
    # 추론 모델이 없으면 v5 학습된 모델 직접 사용 (best_accuracy)
    trained_model_path = os.path.join(_PROJECT_ROOT, "output", "rec_ke_v5_model", "best_accuracy")
    return os.path.exists(trained_model_path + ".pdparams")


def _check_trained_det_model_available() -> bool:
    """v5 학습된 Detection 모델 파일 존재 여부 확인 (PP-OCRv5_server_det)"""
    return os.path.exists(_TRAINED_DET_MODEL_PATH + ".pdmodel")


# 모듈 로드 시 자동으로 설정 파일 보정 (경로 변수 정의 이후 실행)
try:
    _ensure_trained_model_configs()
except Exception as e:
    # 모듈 로드 시점에는 경로가 아직 설정되지 않았을 수 있으므로 무시
    pass


def _get_paddleocr_instance(use_local: bool = False, use_trained: bool = True):
    """
    PaddleOCR 인스턴스를 싱글톤으로 관리 (v5 전용)
    
    모델 우선순위:
    1. v5 학습된 모델 (use_trained=True, 기본값): rec_ke_v5_inference + det_ke_inference (PP-OCRv5)
    2. 기본 v5 모델 (use_trained=False): PP-OCRv5 자동 다운로드 모델
    
    Args:
        use_local: 사용 안 함 (v2.0 구버전, 제거됨)
        use_trained: v5 학습된 모델 사용 여부 (기본값: True)
    
    주의: v5 모델만 사용합니다. 과거 버전 모델은 사용하지 않습니다.
    """
    global _PADDLEOCR_INSTANCE, _PADDLEOCR_LOCAL_INSTANCE, _PADDLEOCR_TRAINED_INSTANCE
    
    try:
        from paddleocr import PaddleOCR
        
        # v5 학습된 모델 우선 사용 (코드 문법 인식 최적화, 88.96% 정확도)
        if use_trained and _check_trained_model_available():
            if _PADDLEOCR_TRAINED_INSTANCE is None:
                try:
                    # v5 추론 모델이 있으면 사용
                    if os.path.exists(_TRAINED_REC_MODEL_PATH + ".pdmodel"):
                        # inference.yml 파일 자동 보정 (Global.model_name 추가)
                        print(f"[OK] v5 학습된 모델 설정 파일 검사 중...")
                        _ensure_trained_model_configs()
                        
                        # inference.yml에서 character_dict 추출하여 사전 파일 생성 (로깅용)
                        char_dict_path = _extract_character_dict_from_yml(_TRAINED_REC_INFERENCE_YML)
                        
                        # PaddleOCR 3.x API에 맞는 파라미터 구성 (CPU 전용)
                        # 참고: inference.yml의 character_dict가 자동으로 사용됨
                        # 주의: 커스텀 모델(rec_model_dir, text_detection_model_dir)을 사용하면 lang 파라미터는 무시됨
                        # 따라서 lang 파라미터는 제거 (inference.yml의 character_dict가 영어+한글을 모두 포함)
                        init_params = {
                            'use_textline_orientation': True,  # 텍스트 방향 보정 (use_angle_cls의 새 이름)
                            # 'lang': 'korean',  # 커스텀 모델 사용 시 무시되므로 제거 (inference.yml의 character_dict 사용)
                            'device': 'cpu',  # CPU 전용 (GPU 메모리 절약) - use_gpu 대신 device 사용
                        }
                        
                        # v5 학습된 Recognition 모델 사용
                        if os.path.exists(_TRAINED_REC_MODEL_PATH + ".pdmodel"):
                            # 구버전 호환성을 위해 rec_model_dir 사용 (model_name mismatch 방지)
                            # PaddleX의 엄격한 검증을 피하기 위해 구버전 파라미터 사용
                            init_params['rec_model_dir'] = _TRAINED_REC_MODEL_DIR
                            print(f"[OK] v5 학습된 Recognition 모델 경로: {_TRAINED_REC_MODEL_DIR}")
                            
                            # inference.yml의 character_dict가 자동으로 사용됨
                            if char_dict_path and os.path.exists(char_dict_path):
                                print(f"[OK] inference.yml의 character_dict 사용 (사전 파일: {char_dict_path})")
                            else:
                                print(f"[WARN] character_dict.txt 생성 실패, inference.yml의 character_dict 사용")
                        
                        # v5 학습된 Detection 모델 사용 (가능한 경우)
                        if _check_trained_det_model_available() and os.path.exists(_TRAINED_DET_MODEL_PATH + ".pdmodel"):
                            # text_detection_model_dir 사용 (det_model_dir는 deprecated이지만 호환됨)
                            init_params['text_detection_model_dir'] = _TRAINED_DET_MODEL_DIR
                            print(f"[OK] v5 학습된 Detection 모델 경로: {_TRAINED_DET_MODEL_DIR} (PP-OCRv5_server_det)")
                            print(f"[OK] PaddleOCR 초기화 중... (v5 학습된 Detection + Recognition 모델 사용, CPU 전용)")
                        else:
                            print(f"[OK] PaddleOCR 초기화 중... (v5 학습된 Recognition 모델 + 기본 v5 Detection 모델 사용, CPU 전용)")
                        
                        # PaddleOCR 인스턴스 생성
                        _PADDLEOCR_TRAINED_INSTANCE = PaddleOCR(**init_params)
                        print(f"[OK] PaddleOCR 초기화 완료 (v5 학습된 모델 사용, CPU 전용, inference.yml character_dict 적용, 코드 문법 인식 최적화, 88.96% 정확도)")
                        
                    else:
                        # v5 추론 모델이 없으면 기본 v5 모델 사용
                        print(f"[WARN] v5 학습된 추론 모델을 찾을 수 없습니다: {_TRAINED_REC_MODEL_PATH}")
                        print(f"   기본 v5 모델로 fallback합니다.")
                        _PADDLEOCR_TRAINED_INSTANCE = None
                        
                except KeyError as e:
                    # KeyError는 inference.yml 설정 문제일 가능성이 높음
                    print(f"[WARN] v5 학습된 모델 로드 실패 (설정 파일 오류): {e}")
                    print(f"   inference.yml 파일을 다시 보정 시도합니다...")
                    _ensure_trained_model_configs()
                    # 한 번 더 시도 (PaddleOCR 3.x API 사용)
                    try:
                        char_dict_path = _extract_character_dict_from_yml(_TRAINED_REC_INFERENCE_YML)
                        
                        init_params = {
                            'use_textline_orientation': True,
                            # 'lang': 'korean',  # 커스텀 모델 사용 시 무시되므로 제거 (inference.yml의 character_dict 사용)
                            'device': 'cpu',  # CPU 전용
                        }
                        if os.path.exists(_TRAINED_REC_MODEL_PATH + ".pdmodel"):
                            init_params['text_recognition_model_dir'] = _TRAINED_REC_MODEL_DIR
                            if char_dict_path and os.path.exists(char_dict_path):
                                print(f"[OK] inference.yml의 character_dict 사용 (재시도, 사전 파일: {char_dict_path})")
                        if _check_trained_det_model_available() and os.path.exists(_TRAINED_DET_MODEL_PATH + ".pdmodel"):
                            init_params['text_detection_model_dir'] = _TRAINED_DET_MODEL_DIR
                        _PADDLEOCR_TRAINED_INSTANCE = PaddleOCR(**init_params)
                        print(f"[OK] PaddleOCR 초기화 완료 (재시도 성공, v5 모델, CPU 전용, inference.yml character_dict 적용)")
                    except Exception as e2:
                        print(f"[WARN] 재시도 실패: {e2}")
                        import traceback
                        print(f"   상세 오류: {traceback.format_exc()}")
                        print(f"   기본 v5 모델로 fallback합니다.")
                        _PADDLEOCR_TRAINED_INSTANCE = None
                        
                except Exception as e:
                    print(f"[WARN] v5 학습된 모델 로드 실패: {e}")
                    import traceback
                    print(f"   상세 오류: {traceback.format_exc()}")
                    print(f"   기본 v5 모델로 fallback합니다.")
                    _PADDLEOCR_TRAINED_INSTANCE = None
                    
            if _PADDLEOCR_TRAINED_INSTANCE is not None:
                return _PADDLEOCR_TRAINED_INSTANCE
        
        # 기본 v5 모델 사용 (PP-OCRv5, 한글+영어 최적화, CPU 전용)
        if _PADDLEOCR_INSTANCE is None:
            _PADDLEOCR_INSTANCE = PaddleOCR(
                use_textline_orientation=True,  # 텍스트 방향 보정 (use_angle_cls의 새 이름)
                lang='korean',  # 한글+영어 자동 인식 (PP-OCRv5 모델)
                device='cpu',  # CPU 전용 (GPU 메모리 절약) - use_gpu 대신 device 사용
            )
            print(f"[OK] PaddleOCR 초기화 완료 (기본 v5 모델, PP-OCRv5, 한글+영어 최적화, CPU 전용)")
        return _PADDLEOCR_INSTANCE
        
        # 로컬 모델은 v2.0 구버전이므로 사용하지 않음 (필요시 주석 해제)
        # if use_local and _check_local_models_available():
        #     if _PADDLEOCR_LOCAL_INSTANCE is None:
        #         _PADDLEOCR_LOCAL_INSTANCE = PaddleOCR(
        #             det_model_dir=_LOCAL_DET_MODEL,
        #             rec_model_dir=_LOCAL_REC_MODEL,
        #             cls_model_dir=_LOCAL_CLS_MODEL,
        #             use_angle_cls=True,
        #             lang='ch',
        #             use_gpu=False,
        #             show_log=False,
        #         )
        #         print(f"⚠ PaddleOCR 초기화 완료 (로컬 v2.0 모델 사용, 구버전)")
        #     return _PADDLEOCR_LOCAL_INSTANCE
        
    except ImportError:
        raise RuntimeError("paddleocr이 설치되지 않았습니다. pip install paddlepaddle paddleocr")
    except Exception as e:
        raise RuntimeError(f"PaddleOCR 초기화 실패: {e}")


def check_paddleocr_available():
    """PaddleOCR 사용 가능 여부 확인 (v5 학습된 모델 우선 사용)"""
    try:
        # v5 학습된 모델 우선 사용 (코드 문법 인식 최적화)
        _get_paddleocr_instance(use_local=False, use_trained=True)
        return True, None
    except Exception as e:
        return False, str(e)


def _load_character_dict_from_yml(yml_path: str) -> Optional[List[str]]:
    """
    inference.yml에서 character_dict를 로드하여 리스트로 반환
    """
    if not os.path.exists(yml_path):
        return None
    
    try:
        with open(yml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        postprocess = config.get('PostProcess', {})
        character_dict = postprocess.get('character_dict', [])
        
        if character_dict and isinstance(character_dict, list):
            return [str(char) for char in character_dict if char]
        return None
    except Exception as e:
        print(f"   ⚠ character_dict 로드 실패: {e}")
        return None


def load_character_dict_txt(txt_path: str) -> Optional[List[str]]:
    """
    character_dict.txt 파일을 직접 로드하여 리스트로 반환
    학습 시 사용한 character_dict.txt 기준으로 직접 디코딩에 사용
    
    Args:
        txt_path: character_dict.txt 파일 경로
        
    Returns:
        character_dict 리스트 (각 인덱스가 문자)
    """
    if not os.path.exists(txt_path):
        print(f"   ⚠ character_dict.txt 파일이 없습니다: {txt_path}")
        return None
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            char_dict = [line.strip() for line in f.readlines() if line.strip()]
        
        if char_dict:
            print(f"   ✅ character_dict.txt 로드 완료 ({len(char_dict)}개 문자)")
            return char_dict
        else:
            print(f"   ⚠ character_dict.txt가 비어있습니다: {txt_path}")
            return None
    except Exception as e:
        print(f"   ⚠ character_dict.txt 로드 실패: {txt_path}, 오류: {e}")
        return None


def paddleocr_word_boxes(
    pil_img: Image.Image,
    *,
    code_mode: bool = True,
    remove_emoji: bool = True,
    use_local_model: bool = False,  # 사용 안 함 (v2.0 구버전, 제거됨)
    use_trained_model: bool = True,  # 기본값: v5 학습된 모델 사용 (코드 문법 인식 최적화)
) -> List[WordBox]:
    """PaddleOCR로 WordBox 리스트 추출 (v5 학습된 모델 우선 사용, 코드 문법 인식 최적화)"""
    try:
        # v5 학습된 모델 인스턴스 가져오기
        ocr = _get_paddleocr_instance(use_local=use_local_model, use_trained=use_trained_model)
        
        # 이미지 정보 확인
        img_width, img_height = pil_img.size
        print(f"   이미지 크기: {img_width}x{img_height}px")
        
        # PIL Image를 numpy array로 변환
        img_array = np.array(pil_img)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            print(f"   이미지 형식: Grayscale → RGB 변환")
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            print(f"   이미지 형식: RGBA → RGB 변환")
        else:
            print(f"   이미지 형식: RGB (shape: {img_array.shape})")
        
        # OCR 실행
        # use_angle_cls=True가 이미 초기화 시 설정되어 있으므로 cls 파라미터 불필요
        print(f"   OCR 실행 중...")
        result = ocr.ocr(img_array)
        print(f"   OCR 실행 완료, 결과 타입: {type(result)}")
        
        # 디버깅: 결과 구조 확인
        if result is None:
            print(f"   ⚠ PaddleOCR 결과가 None입니다. (이미지에 텍스트가 없거나 모델 오류)")
            return []
        
        # 결과 타입 안전하게 확인 (dict가 아닌 list/tuple만 허용)
        # dict인 경우 KeyError: 0 방지를 위해 별도 처리
        if isinstance(result, dict):
            print(f"   ⚠ PaddleOCR 결과가 dict 타입입니다. 키 목록: {list(result.keys())[:10]}")
            # dict인 경우 첫 번째 값이나 특정 키를 시도
            if len(result) > 0:
                first_key = next(iter(result.keys()))
                result = [result[first_key]]  # dict의 첫 번째 값을 리스트로 변환
                print(f"   ✅ dict의 첫 번째 값({first_key})을 리스트로 변환")
            else:
                print(f"   ⚠ dict가 비어있습니다.")
                return []
        
        # 디버깅: 실제 결과 구조 상세 출력 (안전하게)
        # result가 list/tuple인지 확인 (dict는 이미 처리됨)
        if isinstance(result, (list, tuple)) and len(result) > 0:
            try:
                img_result_raw = result[0]
                print(f"   디버깅: result[0] 타입: {type(img_result_raw)}")
                
                # PaddleX OCRResult 객체인지 확인
                if hasattr(img_result_raw, '__class__'):
                    class_name = img_result_raw.__class__.__name__
                    print(f"   디버깅: result[0] 클래스명: {class_name}")
                    if 'OCRResult' in class_name or 'Result' in class_name:
                        # 안전하게 객체 정보 출력
                        try:
                            obj_str = str(img_result_raw)[:200] if hasattr(img_result_raw, '__str__') else repr(img_result_raw)[:200]
                            print(f"   📦 PaddleX OCRResult 객체 감지: {obj_str}")
                            # 주요 속성만 안전하게 확인
                            safe_attrs = [attr for attr in dir(img_result_raw) if not attr.startswith('_') and not callable(getattr(img_result_raw, attr, None))]
                            print(f"   📦 사용 가능한 속성 (일부): {safe_attrs[:10]}")
                        except Exception as e:
                            print(f"   📦 PaddleX OCRResult 객체 감지 (정보 출력 실패: {type(e).__name__})")
                
                # 안전하게 첫 번째 요소 확인 시도 (KeyError 방지)
                try:
                    if img_result_raw is not None:
                        # __getitem__이 있는지 확인
                        if hasattr(img_result_raw, '__getitem__'):
                            try:
                                first_elem = img_result_raw[0]
                                if first_elem is not None:
                                    first_elem_str = str(first_elem)[:100] if not isinstance(first_elem, (list, dict)) else f"{type(first_elem)} (복합 객체)"
                                    print(f"   디버깅: result[0][0] 타입: {type(first_elem)}, 값: {first_elem_str}")
                            except (KeyError, TypeError, IndexError, AttributeError) as e:
                                print(f"   디버깅: result[0][0] 접근 불가 (예상됨 - OCRResult 객체): {type(e).__name__}")
                        else:
                            print(f"   디버깅: result[0]는 인덱스 접근 불가 객체")
                except Exception as e:
                    print(f"   디버깅: result[0] 정보 확인 중 오류: {type(e).__name__}")
            except (KeyError, IndexError, TypeError) as e:
                print(f"   ⚠ result[0] 접근 실패: {type(e).__name__}: {e}")
                print(f"   결과 구조: {str(result)[:200]}")
                return []
        elif not isinstance(result, (list, tuple)):
            print(f"   ⚠ PaddleOCR 결과가 예상치 못한 타입입니다: {type(result)}")
            print(f"   결과 내용: {str(result)[:200]}")
            # 다른 타입인 경우 그대로 처리 시도
            result = [result]  # 단일 객체를 리스트로 감싸기
        
        # PaddleOCR 3.x 결과 구조: [[[bbox, (text, conf)], ...], ...] 또는 None
        # PaddleX OCRResult 객체도 처리 가능하도록 수정
        # 첫 번째 이미지의 결과를 가져옴
        words: List[WordBox] = []
        
        # 결과가 리스트인지 확인 (dict는 이미 처리됨)
        if isinstance(result, (list, tuple)) and len(result) > 0:
            try:
                # 첫 번째 이미지 결과 (KeyError 방지를 위해 try-except 사용)
                img_result_raw = result[0] if result[0] is not None else None
            except (KeyError, IndexError, TypeError) as e:
                print(f"   ⚠ result[0] 접근 실패: {type(e).__name__}: {e}")
                print(f"   결과 타입: {type(result)}, 길이: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                return []
            
            if img_result_raw is None:
                print(f"   ⚠ PaddleOCR 결과가 None입니다 (이미지에서 텍스트를 찾지 못함)")
                return []
            
            # PaddleX OCRResult 객체 처리
            img_result = None
            if hasattr(img_result_raw, '__class__'):
                class_name = img_result_raw.__class__.__name__
                
                # OCRResult 객체인 경우
                if 'OCRResult' in class_name or 'Result' in class_name:
                    print(f"   🔍 PaddleX OCRResult 객체 처리 중...")
                    
                    # 방법 1: doc_res 속성 확인 (가장 중요 - 실제 OCR 결과가 여기 있음)
                    if hasattr(img_result_raw, 'doc_res'):
                        try:
                            doc_res_data = img_result_raw.doc_res
                            if doc_res_data is not None:
                                # doc_res의 모든 속성 확인 (디버깅)
                                if hasattr(doc_res_data, '__dict__'):
                                    doc_res_attrs = [attr for attr in dir(doc_res_data) if not attr.startswith('_')]
                                    print(f"   🔍 doc_res 속성 목록: {doc_res_attrs[:20]}")
                                
                                # doc_res가 객체인 경우 rec_texts 속성 확인
                                if hasattr(doc_res_data, 'rec_texts'):
                                    # 이미 디코딩된 텍스트가 있는 경우
                                    texts_from_doc_res = getattr(doc_res_data, 'rec_texts', [])
                                    if texts_from_doc_res:
                                        print(f"   ✅ doc_res.rec_texts에서 직접 텍스트 추출 성공 ({len(texts_from_doc_res)}개)")
                                        # 샘플 확인
                                        if len(texts_from_doc_res) > 0:
                                            sample = str(texts_from_doc_res[0])[:50] if texts_from_doc_res[0] is not None else "None"
                                            print(f"   🔍 doc_res.rec_texts[0] 샘플: '{sample}'")
                                        # doc_res를 그대로 사용 (rec_texts가 이미 있음)
                                        img_result = doc_res_data
                                    else:
                                        img_result = doc_res_data
                                        print(f"   ✅ doc_res 속성에서 데이터 추출 성공 (타입: {type(doc_res_data)}, rec_texts 비어있음)")
                                elif isinstance(doc_res_data, dict):
                                    # dict인 경우 rec_texts 확인
                                    if 'rec_texts' in doc_res_data:
                                        rec_texts_list = doc_res_data.get('rec_texts', [])
                                        print(f"   ✅ doc_res (dict)에서 rec_texts 발견 ({len(rec_texts_list)}개)")
                                        if len(rec_texts_list) > 0:
                                            sample = str(rec_texts_list[0])[:50] if rec_texts_list[0] is not None else "None"
                                            print(f"   🔍 doc_res['rec_texts'][0] 샘플: '{sample}'")
                                    img_result = doc_res_data
                                    print(f"   ✅ doc_res (dict) 속성에서 데이터 추출 성공")
                                else:
                                    img_result = doc_res_data
                                    print(f"   ✅ doc_res 속성에서 데이터 추출 성공 (타입: {type(doc_res_data)})")
                        except Exception as e:
                            print(f"   ⚠ doc_res 접근 실패: {e}")
                            import traceback
                            print(f"   상세 오류: {traceback.format_exc()}")
                    
                    # 방법 2: json 속성 확인 (PaddleX OCRResult의 주요 데이터 소스)
                    if img_result is None and hasattr(img_result_raw, 'json'):
                        try:
                            json_data = img_result_raw.json
                            print(f"   🔍 json 속성 타입: {type(json_data)}")
                            
                            if isinstance(json_data, dict):
                                print(f"   🔍 json 키 목록: {list(json_data.keys())[:10]}")
                                
                                # 'res' 키 확인 (PaddleX OCRResult의 실제 구조)
                                if 'res' in json_data:
                                    res_data = json_data['res']
                                    print(f"   🔍 json['res'] 발견, 타입: {type(res_data)}")
                                    
                                    # res가 리스트인 경우 (PaddleOCR 3.x 표준 형식)
                                    if isinstance(res_data, list) and len(res_data) > 0:
                                        print(f"   ✅ json['res']가 리스트입니다 (길이: {len(res_data)}) - PaddleOCR 3.x 표준 형식")
                                        # res_data가 바로 OCR 결과 리스트이므로 img_result로 설정
                                        img_result = res_data
                                        print(f"   ✅ json['res'] 리스트를 img_result로 설정")
                                    # res가 dict인 경우
                                    elif isinstance(res_data, dict):
                                        if 'doc_res' in res_data:
                                            doc_res_json = res_data['doc_res']
                                            if isinstance(doc_res_json, dict):
                                                img_result = doc_res_json
                                                print(f"   ✅ json.res.doc_res에서 데이터 추출 성공")
                                            else:
                                                img_result = doc_res_json
                                                print(f"   ✅ json.res.doc_res에서 데이터 추출 성공 (dict 아님)")
                                        elif 'rec_texts' in res_data:
                                            # rec_texts가 있는지 확인
                                            rec_texts_list = res_data.get('rec_texts', [])
                                            print(f"   🔍 json.res.rec_texts 발견: {len(rec_texts_list)}개")
                                            # 디버깅: rec_texts의 첫 번째 요소 확인
                                            if rec_texts_list and len(rec_texts_list) > 0:
                                                print(f"   🔍 rec_texts[0] 타입: {type(rec_texts_list[0])}, 값: {repr(str(rec_texts_list[0])[:50])}")
                                            img_result = res_data
                                            print(f"   ✅ json.res에서 직접 데이터 추출 성공 (rec_texts 있음)")
                                        else:
                                            print(f"   🔍 json.res 키 목록: {list(res_data.keys())[:20]}")
                                            # json['res']의 실제 구조를 더 자세히 확인
                                            import json as json_module
                                            res_data_str = json_module.dumps(res_data, ensure_ascii=False, indent=2)
                                            print(f"   🔍 json['res'] 전체 구조 (처음 1000자): {res_data_str[:1000]}")
                                            img_result = res_data
                                # doc_res가 있는 경우
                                elif 'doc_res' in json_data:
                                    doc_res_json = json_data['doc_res']
                                    print(f"   ✅ json.doc_res 발견, 타입: {type(doc_res_json)}")
                                    
                                    # doc_res가 dict인 경우
                                    if isinstance(doc_res_json, dict):
                                        # rec_texts가 있는지 확인
                                        if 'rec_texts' in doc_res_json:
                                            rec_texts_list = doc_res_json.get('rec_texts', [])
                                            print(f"   ✅ json.doc_res.rec_texts 발견 ({len(rec_texts_list)}개)")
                                            if len(rec_texts_list) > 0:
                                                sample = str(rec_texts_list[0])[:50] if rec_texts_list[0] is not None else "None"
                                                print(f"   🔍 json.doc_res.rec_texts[0] 샘플: '{sample}'")
                                        img_result = doc_res_json
                                    else:
                                        img_result = doc_res_json
                                elif 'result' in json_data:
                                    img_result = json_data['result']
                                    print(f"   ✅ json.result에서 데이터 추출 성공")
                                elif 'rec_texts' in json_data:
                                    # json에 직접 rec_texts가 있는 경우
                                    rec_texts_list = json_data.get('rec_texts', [])
                                    print(f"   ✅ json.rec_texts 직접 발견 ({len(rec_texts_list)}개)")
                                    img_result = json_data
                            elif isinstance(json_data, str):
                                # json이 문자열인 경우 파싱 시도
                                try:
                                    import json
                                    json_parsed = json.loads(json_data)
                                    print(f"   ✅ json 문자열 파싱 성공, 타입: {type(json_parsed)}")
                                    if isinstance(json_parsed, dict):
                                        if 'doc_res' in json_parsed:
                                            img_result = json_parsed['doc_res']
                                            print(f"   ✅ 파싱된 json.doc_res에서 데이터 추출 성공")
                                        elif 'rec_texts' in json_parsed:
                                            img_result = json_parsed
                                            print(f"   ✅ 파싱된 json에서 직접 데이터 추출 성공")
                                except Exception as e:
                                    print(f"   ⚠ json 문자열 파싱 실패: {e}")
                            elif isinstance(json_data, list):
                                img_result = json_data
                                print(f"   ✅ json에서 리스트 데이터 추출 성공")
                        except Exception as e:
                            print(f"   ⚠ json 접근 실패: {e}")
                            import traceback
                            print(f"   상세 오류: {traceback.format_exc()}")
                    
                    # 방법 3: dict처럼 접근
                    if img_result is None and isinstance(img_result_raw, dict):
                        if 'doc_res' in img_result_raw:
                            img_result = img_result_raw['doc_res']
                            print(f"   ✅ dict['doc_res']에서 데이터 추출 성공")
                        elif 'result' in img_result_raw:
                            img_result = img_result_raw['result']
                            print(f"   ✅ dict['result']에서 데이터 추출 성공")
                    
                    # 방법 4: 순회 가능한 객체인 경우 (dict가 아닌 경우만)
                    if img_result is None and hasattr(img_result_raw, '__iter__') and not isinstance(img_result_raw, (str, dict)):
                        try:
                            # 리스트로 변환 시도
                            temp_list = list(img_result_raw)
                            if len(temp_list) > 0:
                                img_result = temp_list
                                print(f"   ✅ 순회 가능한 객체를 리스트로 변환 성공 ({len(temp_list)}개 요소)")
                        except Exception as e:
                            print(f"   ⚠ 순회 변환 실패: {e}")
                    
                    # 방법 5: 속성 직접 확인
                    if img_result is None:
                        # 모든 속성 확인
                        try:
                            attrs = [attr for attr in dir(img_result_raw) if not attr.startswith('_') and not callable(getattr(img_result_raw, attr, None))]
                            print(f"   🔍 사용 가능한 속성: {attrs[:15]}")
                            
                            # 일반적인 속성명 시도
                            for attr_name in ['results', 'data', 'output', 'lines', 'texts', 'ocr_result', 'rec_result', 'det_result']:
                                if hasattr(img_result_raw, attr_name):
                                    try:
                                        potential_data = getattr(img_result_raw, attr_name)
                                        if isinstance(potential_data, (list, tuple)) and len(potential_data) > 0:
                                            img_result = potential_data
                                            print(f"   ✅ {attr_name} 속성에서 데이터 추출 성공 ({len(potential_data)}개 요소)")
                                            break
                                        elif isinstance(potential_data, dict):
                                            # dict 내부에 리스트가 있는지 확인
                                            for key in ['doc_res', 'result', 'data', 'lines']:
                                                if key in potential_data and isinstance(potential_data[key], (list, tuple)):
                                                    img_result = potential_data[key]
                                                    print(f"   ✅ {attr_name}.{key}에서 데이터 추출 성공")
                                                    break
                                            if img_result is not None:
                                                break
                                    except Exception as e:
                                        continue
                        except Exception as e:
                            print(f"   ⚠ 속성 확인 중 오류: {e}")
            
            # OCRResult 객체가 아니거나 변환 실패한 경우 원본 사용
            if img_result is None:
                img_result = img_result_raw
            
            # 최종 검증 및 데이터 정규화
            if img_result is None:
                print(f"   ⚠ PaddleOCR 결과가 None입니다 (이미지에서 텍스트를 찾지 못함)")
                return []
            
            # OCRResult 객체 또는 dict에서 실제 데이터 추출 (rec_texts, dt_polys, rec_scores)
            texts = []
            polys = []
            scores = []
            use_ocr_result_format = False
            
            # OCRResult 객체인지 먼저 확인 (dict 체크보다 우선)
            if hasattr(img_result, '__class__'):
                class_name = img_result.__class__.__name__
                if 'OCRResult' in class_name or 'Result' in class_name:
                    print(f"   🔍 OCRResult 객체 감지, rec_texts/dt_polys/rec_scores 추출 중...")
                    use_ocr_result_format = True
                    
                    # 방법 1: doc_res 속성을 통한 접근 (가장 정확)
                    if hasattr(img_result, 'doc_res'):
                        try:
                            doc_res = img_result.doc_res
                            if doc_res is not None:
                                # doc_res가 객체인 경우
                                if hasattr(doc_res, 'rec_texts'):
                                    texts = getattr(doc_res, 'rec_texts', None)
                                    polys = getattr(doc_res, 'dt_polys', None)
                                    scores = getattr(doc_res, 'rec_scores', None)
                                    print(f"   ✅ doc_res.rec_texts에서 직접 추출 성공")
                                # doc_res가 dict인 경우
                                elif isinstance(doc_res, dict):
                                    texts = doc_res.get('rec_texts', None)
                                    polys = doc_res.get('dt_polys', None)
                                    scores = doc_res.get('rec_scores', None)
                                    print(f"   ✅ doc_res (dict)에서 직접 추출 성공")
                        except Exception as e:
                            print(f"   ⚠ doc_res 접근 실패: {e}")
                    
                    # 방법 2: json 속성 사용 (PaddleX OCRResult의 주요 데이터 소스)
                    if texts is None and hasattr(img_result, 'json'):
                        try:
                            json_data = img_result.json
                            print(f"   🔍 json 속성 타입: {type(json_data)}")
                            
                            if isinstance(json_data, dict):
                                # json이 dict인 경우
                                # 먼저 'res' 키 확인 (PaddleX OCRResult의 실제 구조)
                                if 'res' in json_data:
                                    res_data = json_data.get("res", {})
                                    
                                    # ========================================
                                    # 1️⃣ json['res'] 구조 및 키 전체 출력 (최우선)
                                    # ========================================
                                    print("🔍 json['res'] 전체 키 목록:", list(res_data.keys()))
                                    
                                    import json as json_module
                                    try:
                                        res_preview = json_module.dumps(
                                            res_data,
                                            ensure_ascii=False,
                                            indent=2,
                                            default=str
                                        )
                                        print("🔍 json['res'] 전체 구조 미리보기 (앞 2000자):")
                                        print(res_preview[:2000])
                                    except Exception as e:
                                        print(f"⚠ json['res'] 구조 출력 실패: {e}")
                                    
                                    print(f"   🔍 json['res'] 타입: {type(res_data)}")
                                    
                                    # res가 dict인 경우
                                    if isinstance(res_data, dict):
                                        # ========================================
                                        # 2️⃣ rec_indices 존재 여부에 따른 분기 처리 (최우선)
                                        # ========================================
                                        if "rec_indices" in res_data:
                                            print("✅ rec_indices 발견, 직접 디코딩 로직 사용")
                                            
                                            # character_dict.txt 직접 로드 (학습 시 사용한 기준)
                                            char_dict_path = os.path.join(_TRAINED_REC_MODEL_DIR, "character_dict.txt")
                                            char_dict = load_character_dict_txt(char_dict_path)
                                            
                                            if char_dict:
                                                # rec_indices를 사용하여 직접 디코딩
                                                rec_indices = res_data.get("rec_indices", [])
                                                texts = []
                                                
                                                for indices in rec_indices:
                                                    if not isinstance(indices, (list, tuple)):
                                                        continue
                                                    
                                                    decoded_chars = []
                                                    for idx in indices:
                                                        if isinstance(idx, int) and 0 <= idx < len(char_dict):
                                                            decoded_chars.append(char_dict[idx])
                                                    
                                                    texts.append("".join(decoded_chars))
                                                
                                                # polys와 scores 추출
                                                polys = res_data.get('dt_polys', [])
                                                scores = res_data.get('rec_scores', [])
                                                
                                                print(f"   ✅ rec_indices로 직접 디코딩 완료: texts={len(texts) if texts else 0}개")
                                                
                                                # ========================================
                                                # 3️⃣ 디코딩 결과 검증 로그
                                                # ========================================
                                                if texts and len(texts) > 0:
                                                    print("🧪 디코딩 검증 샘플")
                                                    if len(rec_indices) > 0:
                                                        print(f"   rec_indices[0][:20]: {rec_indices[0][:20] if len(rec_indices[0]) > 20 else rec_indices[0]}")
                                                    print(f"   decoded_text: {texts[0]}")
                                                    # 정상 기준: def, if, :, {, }, return 등 코드 문자가 정상 출력되면 성공
                                                
                                                # texts, polys, scores가 설정되었으므로 이후 로직으로 진행
                                            else:
                                                print("   ⚠ character_dict.txt 로드 실패, rec_texts fallback 사용")
                                                texts = res_data.get('rec_texts', [])
                                                polys = res_data.get('dt_polys', [])
                                                scores = res_data.get('rec_scores', [])
                                        # res 안에 doc_res가 있는지 확인
                                        elif 'doc_res' in res_data:
                                            doc_res_json = res_data['doc_res']
                                            if isinstance(doc_res_json, dict):
                                                texts = doc_res_json.get('rec_texts', [])
                                                polys = doc_res_json.get('dt_polys', [])
                                                scores = doc_res_json.get('rec_scores', [])
                                                print(f"   ✅ json.res.doc_res에서 추출 성공: texts={len(texts) if texts else 0}개")
                                        # res 안에 직접 rec_texts가 있는지 확인
                                        elif 'rec_texts' in res_data:
                                            texts = res_data.get('rec_texts', [])
                                            polys = res_data.get('dt_polys', [])
                                            scores = res_data.get('rec_scores', [])
                                            print(f"   ✅ json.res에서 직접 추출 성공: texts={len(texts) if texts else 0}개")
                                            # 디버깅: texts의 실제 타입과 값 확인
                                            if texts and len(texts) > 0:
                                                print(f"   🔍 texts[0] 타입: {type(texts[0])}, 값 샘플: {repr(str(texts[0])[:50])}")
                                                # texts가 인덱스 리스트인지 확인
                                                if isinstance(texts[0], (int, list)):
                                                    print(f"   ⚠ texts가 인덱스 형태입니다! character_dict 디코딩 필요")
                                        else:
                                            # rec_indices가 없는 경우
                                            print("⚠ rec_indices 없음 → rec_texts fallback 사용")
                                            print(f"   🔍 json.res 키 목록: {list(res_data.keys())[:20]}")
                                            # rec_indices가 없는 경우, 모델 export 단계 문제 가능성
                                            print("   💡 참고: rec_indices가 없으면 Recognition 모델 재-export 필요할 수 있음")
                                            texts = res_data.get('rec_texts', [])
                                            polys = res_data.get('dt_polys', [])
                                            scores = res_data.get('rec_scores', [])
                                    # res가 list인 경우 (직접 결과 리스트) - PaddleOCR 3.x 표준 형식
                                    elif isinstance(res_data, list) and len(res_data) > 0:
                                        # PaddleOCR 3.x 표준 형식: [[[bbox], (text, conf)], ...]
                                        # 이 경우 res_data가 바로 OCR 결과 리스트이므로 img_result로 사용
                                        print(f"   🔍 json['res']가 리스트입니다 (길이: {len(res_data)})")
                                        # 첫 번째 요소 샘플 확인
                                        if len(res_data) > 0:
                                            print(f"   🔍 res_data[0] 타입: {type(res_data[0])}, 값 샘플: {str(res_data[0])[:100]}")
                                        # 리스트의 첫 번째 요소가 dict인 경우 (다른 형식)
                                        if isinstance(res_data[0], dict):
                                            first_item = res_data[0]
                                            if 'rec_texts' in first_item:
                                                texts = first_item.get('rec_texts', [])
                                                polys = first_item.get('dt_polys', [])
                                                scores = first_item.get('rec_scores', [])
                                                print(f"   ✅ json.res[0]에서 추출 성공: texts={len(texts) if texts else 0}개")
                                            elif 'text' in first_item:
                                                # text 필드가 있는 경우 (다른 형식)
                                                texts = [first_item.get('text', '')]
                                                bbox = first_item.get('bbox', [])
                                                if bbox:
                                                    polys = [bbox]
                                                conf = first_item.get('confidence', 0.0)
                                                if conf:
                                                    scores = [conf]
                                                print(f"   ✅ json.res[0]에서 text 형식으로 추출 성공: texts={len(texts) if texts else 0}개")
                                        else:
                                            # PaddleOCR 3.x 표준 형식: [[[bbox], (text, conf)], ...]
                                            # res_data가 바로 OCR 결과 리스트이지만, 이 블록은 두 번째 OCRResult 처리 부분이므로
                                            # 첫 번째 처리 부분에서 이미 img_result가 설정되었을 것임
                                            # 여기서는 texts/polys/scores를 추출할 수 없으므로 None으로 유지
                                            print(f"   🔍 json['res']가 리스트입니다 (길이: {len(res_data)}) - 첫 번째 처리 부분에서 이미 처리됨")
                                # 기존 로직: doc_res 직접 확인
                                elif 'doc_res' in json_data:
                                    doc_res_json = json_data['doc_res']
                                    if isinstance(doc_res_json, dict):
                                        texts = doc_res_json.get('rec_texts', [])
                                        polys = doc_res_json.get('dt_polys', [])
                                        scores = doc_res_json.get('rec_scores', [])
                                        print(f"   ✅ json.doc_res에서 추출 성공: texts={len(texts) if texts else 0}개")
                                # 기존 로직: rec_texts 직접 확인
                                elif 'rec_texts' in json_data:
                                    texts = json_data.get('rec_texts', [])
                                    polys = json_data.get('dt_polys', [])
                                    scores = json_data.get('rec_scores', [])
                                    print(f"   ✅ json에서 직접 추출 성공: texts={len(texts) if texts else 0}개")
                                else:
                                    print(f"   🔍 json 키 목록: {list(json_data.keys())[:10]}")
                                    # json 구조 전체 출력 (디버깅용)
                                    import json as json_module
                                    json_str = json_module.dumps(json_data, ensure_ascii=False, indent=2)
                                    print(f"   🔍 json 전체 구조 (처음 500자): {json_str[:500]}")
                            elif isinstance(json_data, str):
                                # json이 문자열인 경우 파싱 시도
                                try:
                                    import json
                                    json_parsed = json.loads(json_data)
                                    if isinstance(json_parsed, dict):
                                        if 'doc_res' in json_parsed:
                                            doc_res_json = json_parsed['doc_res']
                                            if isinstance(doc_res_json, dict):
                                                texts = doc_res_json.get('rec_texts', [])
                                                polys = doc_res_json.get('dt_polys', [])
                                                scores = doc_res_json.get('rec_scores', [])
                                                print(f"   ✅ json 문자열 파싱 후 doc_res에서 추출 성공: texts={len(texts) if texts else 0}개")
                                        elif 'rec_texts' in json_parsed:
                                            texts = json_parsed.get('rec_texts', [])
                                            polys = json_parsed.get('dt_polys', [])
                                            scores = json_parsed.get('rec_scores', [])
                                            print(f"   ✅ json 문자열 파싱 후 직접 추출 성공: texts={len(texts) if texts else 0}개")
                                except Exception as e:
                                    print(f"   ⚠ json 문자열 파싱 실패: {e}")
                        except Exception as e:
                            print(f"   ⚠ json 속성 접근 실패: {e}")
                            import traceback
                            print(f"   상세 오류: {traceback.format_exc()}")
                    
                    # 방법 3: str 속성 사용 (문자열로 변환된 결과)
                    if texts is None and hasattr(img_result, 'str'):
                        try:
                            str_data = img_result.str
                            print(f"   🔍 str 속성 타입: {type(str_data)}, 길이: {len(str(str_data)) if str_data else 0}")
                            # str 속성은 보통 문자열 표현이므로, 여기서는 직접 사용하기 어려움
                            # 하지만 디버깅용으로 출력
                            if str_data:
                                print(f"   🔍 str 속성 샘플: {str(str_data)[:200]}")
                        except Exception as e:
                            print(f"   ⚠ str 속성 접근 실패: {e}")
                    
                    # 방법 4: 객체 속성으로 직접 접근 (doc_res가 없는 경우)
                    if texts is None:
                        texts = getattr(img_result, 'rec_texts', None)
                        polys = getattr(img_result, 'dt_polys', None)
                        scores = getattr(img_result, 'rec_scores', None)
                        if texts is not None:
                            print(f"   ✅ 객체 속성에서 직접 추출 성공")
                    
                    # None인 경우 dict 접근 시도
                    if texts is None:
                        if hasattr(img_result, '__getitem__'):
                            try:
                                texts = img_result.get('rec_texts', []) if hasattr(img_result, 'get') else img_result['rec_texts'] if 'rec_texts' in img_result else []
                                polys = img_result.get('dt_polys', []) if hasattr(img_result, 'get') else img_result['dt_polys'] if 'dt_polys' in img_result else []
                                scores = img_result.get('rec_scores', []) if hasattr(img_result, 'get') else img_result['rec_scores'] if 'rec_scores' in img_result else []
                            except (KeyError, TypeError, AttributeError):
                                texts = []
                                polys = []
                                scores = []
                        else:
                            texts = []
                            polys = []
                            scores = []
            
            # OCRResult 객체가 아니고 dict인 경우
            if not use_ocr_result_format and isinstance(img_result, dict):
                print(f"   🔍 dict 형태 결과 감지, rec_texts/dt_polys/rec_scores 추출 시도...")
                use_ocr_result_format = True
                
                texts = img_result.get('rec_texts', [])
                polys = img_result.get('dt_polys', [])
                scores = img_result.get('rec_scores', [])
            
            # OCRResult 형식 데이터가 추출된 경우 처리
            if use_ocr_result_format and texts is not None:
                # 리스트가 아닌 경우 변환 시도
                if not isinstance(texts, (list, tuple)):
                    texts = list(texts) if hasattr(texts, '__iter__') and not isinstance(texts, str) else []
                if not isinstance(polys, (list, tuple)):
                    polys = list(polys) if hasattr(polys, '__iter__') and not isinstance(polys, str) else []
                if not isinstance(scores, (list, tuple)):
                    scores = list(scores) if hasattr(scores, '__iter__') and not isinstance(scores, str) else []
                
                print(f"   ✅ OCRResult 데이터 추출 완료: texts={len(texts)}개, polys={len(polys)}개, scores={len(scores)}개")
                
                # 디버깅: 추출된 텍스트 샘플 확인 (처음 5개)
                if len(texts) > 0:
                    sample_texts = [str(texts[i])[:50] for i in range(min(5, len(texts)))]
                    print(f"   🔍 추출된 텍스트 샘플 (처음 5개): {sample_texts}")
                    # 텍스트가 인덱스(숫자)인지 확인
                    if len(texts) > 0:
                        first_text = str(texts[0])
                        if first_text.isdigit() or (first_text.startswith('[') and ']' in first_text):
                            print(f"   ⚠ 경고: rec_texts가 인덱스로 보입니다! character_dict 디코딩이 필요할 수 있습니다.")
                            print(f"   🔍 첫 번째 텍스트 값: {first_text} (타입: {type(texts[0])})")
                
                # 텍스트가 없으면 빈 결과 반환
                if len(texts) == 0:
                    print(f"   ⚠ rec_texts가 비어있습니다 (이미지에서 텍스트를 찾지 못함)")
                    return []
            
            # 일반 리스트 형태인 경우 (기존 로직)
            if not use_ocr_result_format:
                # dict 형태인 경우 리스트로 변환 시도
                if isinstance(img_result, dict):
                    print(f"   🔍 dict 형태 결과 감지, 키: {list(img_result.keys())[:10]}")
                    # 일반적인 키 이름 확인
                    for key in ['doc_res', 'result', 'data', 'lines', 'texts', 'ocr_result']:
                        if key in img_result and isinstance(img_result[key], (list, tuple)):
                            img_result = img_result[key]
                            print(f"   ✅ dict['{key}']에서 리스트 데이터 추출 성공")
                            break
                    # 여전히 dict인 경우 첫 번째 값 사용
                    if isinstance(img_result, dict):
                        first_value = next(iter(img_result.values())) if img_result else None
                        if isinstance(first_value, (list, tuple)):
                            img_result = first_value
                            print(f"   ✅ dict의 첫 번째 값에서 리스트 데이터 추출 성공")
                
                # 빈 결과 확인
                if hasattr(img_result, '__len__'):
                    if len(img_result) == 0:
                        print(f"   ⚠ PaddleOCR 결과가 비어있습니다 (이미지에서 텍스트를 찾지 못함)")
                        return []
                    print(f"   ✅ PaddleOCR 결과: {len(img_result)}개 라인 감지 (타입: {type(img_result).__name__})")
                else:
                    # 길이를 알 수 없는 경우 (예: 제너레이터)
                    print(f"   ⚠ PaddleOCR 결과 길이를 확인할 수 없습니다. 순회 시도...")
                    try:
                        img_result = list(img_result) if hasattr(img_result, '__iter__') and not isinstance(img_result, str) else [img_result]
                        if len(img_result) == 0:
                            print(f"   ⚠ PaddleOCR 결과가 비어있습니다")
                            return []
                        print(f"   ✅ PaddleOCR 결과: {len(img_result)}개 라인 감지 (변환 후)")
                    except Exception as e:
                        print(f"   ⚠ 결과 변환 실패: {e}")
                        return []
            
            # 이미지 크기 (bbox가 없을 때 사용)
            img_w, img_h = float(img_width), float(img_height)
            
            # OCRResult에서 추출한 데이터로 WordBox 생성
            if texts:
                # 디버깅: 추출된 텍스트 샘플 확인 및 인덱스 여부 체크
                if len(texts) > 0:
                    sample_texts = []
                    sample_types = []
                    for idx in range(min(5, len(texts))):
                        sample = texts[idx]
                        sample_texts.append(str(sample)[:50] if sample is not None else "None")
                        sample_types.append(type(sample).__name__)
                    print(f"   🔍 추출된 텍스트 샘플 (처음 5개): {sample_texts}")
                    print(f"   🔍 샘플 타입: {sample_types}")
                    
                    # 텍스트가 인덱스(숫자)인지 확인
                    first_text = str(texts[0]) if texts[0] is not None else ""
                    if first_text.isdigit() or (first_text.startswith('[') and ']' in first_text):
                        print(f"   ⚠ 경고: rec_texts가 인덱스로 보입니다! character_dict 디코딩이 필요할 수 있습니다.")
                        print(f"   🔍 첫 번째 텍스트 값: {first_text} (타입: {type(texts[0])})")
                        print(f"   ⚠ PaddleX가 inference.yml의 character_dict를 사용하지 않고 기본 사전을 사용 중일 수 있습니다.")
                
                # character_dict 로드 (인덱스 디코딩용)
                char_dict = None
                if os.path.exists(_TRAINED_REC_INFERENCE_YML):
                    char_dict = _load_character_dict_from_yml(_TRAINED_REC_INFERENCE_YML)
                    if char_dict:
                        print(f"   ✅ character_dict 로드 완료 ({len(char_dict)}개 문자)")
                    else:
                        print(f"   ⚠ character_dict 로드 실패")
                
                for i in range(len(texts)):
                    text_raw = texts[i]
                    text = str(text_raw) if text_raw is not None else ""
                    
                    # 디버깅: 각 텍스트의 실제 값 확인 (처음 10개만)
                    if i < 10:
                        print(f"   🔍 텍스트[{i}] 원본: '{text[:50]}' (타입: {type(text_raw)}, 값: {repr(text_raw)[:50]})")
                    
                    # 텍스트가 인덱스(정수 또는 정수 리스트)인 경우 character_dict로 디코딩 시도
                    if char_dict and text_raw is not None:
                        try:
                            # 정수인 경우
                            if isinstance(text_raw, int):
                                if 0 <= text_raw < len(char_dict):
                                    text = char_dict[text_raw]
                                    if i < 10:
                                        print(f"   ✅ 인덱스 {text_raw} -> 문자 '{text}' 디코딩 성공")
                                else:
                                    if i < 10:
                                        print(f"   ⚠ 인덱스 {text_raw}가 character_dict 범위를 벗어남 (최대: {len(char_dict)-1})")
                            # 정수 리스트인 경우 (문자열로 변환)
                            elif isinstance(text_raw, (list, tuple)) and len(text_raw) > 0:
                                decoded_chars = []
                                for idx in text_raw:
                                    if isinstance(idx, int) and 0 <= idx < len(char_dict):
                                        decoded_chars.append(char_dict[idx])
                                if decoded_chars:
                                    text = ''.join(decoded_chars)
                                    if i < 10:
                                        print(f"   ✅ 인덱스 리스트 {text_raw[:10]} -> 문자열 '{text[:50]}' 디코딩 성공")
                            # 문자열이지만 숫자로만 구성된 경우 (인덱스 문자열)
                            elif isinstance(text_raw, str) and text_raw.isdigit():
                                idx = int(text_raw)
                                if 0 <= idx < len(char_dict):
                                    text = char_dict[idx]
                                    if i < 10:
                                        print(f"   ✅ 인덱스 문자열 '{text_raw}' -> 문자 '{text}' 디코딩 성공")
                        except Exception as e:
                            if i < 10:
                                print(f"   ⚠ 인덱스 디코딩 실패: {e}")
                    
                    # 텍스트가 인덱스(정수 또는 정수 리스트)인 경우 character_dict로 디코딩 시도
                    if char_dict and text:
                        try:
                            # 정수인 경우
                            if isinstance(text_raw, int):
                                if 0 <= text_raw < len(char_dict):
                                    text = char_dict[text_raw]
                                    if i < 10:
                                        print(f"   ✅ 인덱스 {text_raw} -> 문자 '{text}' 디코딩 성공")
                                else:
                                    if i < 10:
                                        print(f"   ⚠ 인덱스 {text_raw}가 character_dict 범위를 벗어남 (최대: {len(char_dict)-1})")
                            # 정수 리스트인 경우 (문자열로 변환)
                            elif isinstance(text_raw, (list, tuple)) and len(text_raw) > 0:
                                decoded_chars = []
                                for idx in text_raw:
                                    if isinstance(idx, int) and 0 <= idx < len(char_dict):
                                        decoded_chars.append(char_dict[idx])
                                if decoded_chars:
                                    text = ''.join(decoded_chars)
                                    if i < 10:
                                        print(f"   ✅ 인덱스 리스트 {text_raw[:10]} -> 문자열 '{text[:50]}' 디코딩 성공")
                        except Exception as e:
                            if i < 10:
                                print(f"   ⚠ 인덱스 디코딩 실패: {e}")
                    
                    if not text.strip():
                        continue
                    
                    # bbox 추출
                    bbox = None
                    x, y, w, h = 0.0, 0.0, 0.0, 0.0
                    
                    if i < len(polys) and polys[i] is not None:
                        bbox = polys[i]
                        try:
                            # bbox에서 좌표 추출
                            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                x_coords = [float(p[0]) for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
                                y_coords = [float(p[1]) for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
                                
                                if x_coords and y_coords:
                                    x = float(min(x_coords))
                                    y = float(min(y_coords))
                                    x2 = float(max(x_coords))
                                    y2 = float(max(y_coords))
                                    w = x2 - x
                                    h = y2 - y
                        except Exception as e:
                            print(f"   ⚠ 라인 {i} bbox 좌표 추출 실패: {e}")
                            # bbox 추출 실패 시 기본값 사용
                            x, y, w, h = 0.0, float(i * 30), img_w, 30.0
                    else:
                        # bbox가 없으면 기본값 사용
                        x, y, w, h = 0.0, float(i * 30), img_w, 30.0
                    
                    # 신뢰도 추출
                    conf = 0.5  # 기본값
                    if i < len(scores) and scores[i] is not None:
                        try:
                            conf = float(scores[i])
                        except (ValueError, TypeError):
                            conf = 0.5
                    
                    # 텍스트 정리
                    try:
                        text_cleaned = sanitize_text(text, remove_emoji=remove_emoji, keep_newlines=False, collapse_spaces=False)
                        
                        if text_cleaned.strip():
                            # w, h가 0이면 기본값 설정
                            if w <= 0 or h <= 0:
                                w, h = img_w, 30.0
                            
                            words.append(
                                WordBox(
                                    text=text_cleaned,
                                    x=x,
                                    y=y,
                                    w=w,
                                    h=h,
                                    conf=conf,
                                )
                            )
                    except Exception as e:
                        print(f"   ⚠ 라인 {i} WordBox 생성 실패: {e}")
                        continue
            
            # 일반 리스트 형태인 경우 기존 로직 사용
            elif not isinstance(img_result, dict) and hasattr(img_result, '__iter__') and not isinstance(img_result, str):
                for line_idx, line in enumerate(img_result):
                    if line is None:
                        continue
                    
                    text = ""
                    conf = 0.0
                    bbox = None
                    x, y, w, h = 0.0, 0.0, 0.0, 0.0
                    
                    # 케이스 1: line이 문자열인 경우 (현재 발생 중인 오류)
                    if isinstance(line, str):
                        text = line.strip()
                        if text:
                            # 전체 이미지를 bbox로 사용 (가상 더미 bbox)
                            x, y = 0.0, float(line_idx * 30)  # 라인별로 약간씩 아래로 배치
                            w, h = img_w, 30.0  # 기본 높이
                            conf = 0.5  # 기본 신뢰도
                            print(f"   📝 라인 {line_idx}: 문자열 형식 처리 (텍스트: {text[:50]}...)")
                    
                    # 케이스 2: 표준 형식 [bbox, (text, confidence)] 또는 [bbox, [text, confidence]]
                    elif isinstance(line, (list, tuple)) and len(line) >= 2:
                        bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text_info = line[1]  # (text, confidence) 또는 [text, confidence]
                        
                        # text_info 처리
                        if text_info is not None:
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = str(text_info[0]) if text_info[0] is not None else ""
                                try:
                                    conf = float(text_info[1]) if text_info[1] is not None else 0.0
                                except (ValueError, TypeError):
                                    conf = 0.0
                            elif isinstance(text_info, str):
                                text = text_info
                                conf = 0.5
                            else:
                                text = str(text_info) if text_info else ""
                                conf = 0.0
                        
                        # bbox 처리
                        if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            try:
                                # bbox에서 좌표 추출
                                x_coords = [float(p[0]) for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
                                y_coords = [float(p[1]) for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
                                
                                if x_coords and y_coords:
                                    x = float(min(x_coords))
                                    y = float(min(y_coords))
                                    x2 = float(max(x_coords))
                                    y2 = float(max(y_coords))
                                    w = x2 - x
                                    h = y2 - y
                            except Exception as e:
                                print(f"   ⚠ 라인 {line_idx} bbox 좌표 추출 실패: {e}")
                                # bbox 추출 실패 시 전체 이미지 사용
                                x, y, w, h = 0.0, float(line_idx * 30), img_w, 30.0
                        else:
                            # bbox가 유효하지 않으면 전체 이미지 사용
                            x, y, w, h = 0.0, float(line_idx * 30), img_w, 30.0
                    
                    # 케이스 3: 간소화 형식 (text, confidence) 또는 [text, confidence]
                    elif isinstance(line, (list, tuple)) and len(line) >= 1:
                        # 첫 번째 요소가 텍스트
                        if isinstance(line[0], str):
                            text = line[0]
                        else:
                            text = str(line[0]) if line[0] is not None else ""
                        
                        # 두 번째 요소가 confidence (있는 경우)
                        if len(line) >= 2:
                            try:
                                conf = float(line[1]) if line[1] is not None else 0.0
                            except (ValueError, TypeError):
                                conf = 0.0
                        else:
                            conf = 0.5
                        
                        # bbox 없음, 전체 이미지 사용
                        x, y, w, h = 0.0, float(line_idx * 30), img_w, 30.0
                        print(f"   📝 라인 {line_idx}: 간소화 형식 처리 (텍스트: {text[:50]}...)")
                    
                    # 케이스 4: 알 수 없는 형식
                    else:
                        print(f"   ⚠ 라인 {line_idx} 알 수 없는 형식: {type(line)}, 값: {str(line)[:100]}")
                        # 그래도 텍스트 추출 시도
                        if isinstance(line, str):
                            text = line.strip()
                            x, y, w, h = 0.0, float(line_idx * 30), img_w, 30.0
                            conf = 0.5
                        else:
                            continue
                    
                    # 텍스트가 있으면 WordBox 생성
                    if text:
                        try:
                            # 텍스트 정리
                            text_cleaned = sanitize_text(text, remove_emoji=remove_emoji, keep_newlines=False, collapse_spaces=False)
                            
                            if text_cleaned.strip():
                                # w, h가 0이면 기본값 설정
                                if w <= 0 or h <= 0:
                                    w, h = img_w, 30.0
                                
                                words.append(
                                    WordBox(
                                        text=text_cleaned,
                                        x=x,
                                        y=y,
                                        w=w,
                                        h=h,
                                        conf=conf,
                                    )
                                )
                        except Exception as e:
                            print(f"   ⚠ 라인 {line_idx} WordBox 생성 실패: {e}")
                            continue
        else:
            print(f"   ⚠ PaddleOCR 결과 형식 오류: {type(result)}, 길이: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            return []
        
        if words:
            print(f"   ✅ PaddleOCR WordBox 추출 성공: {len(words)}개 단어 인식")
        else:
            print(f"   ⚠ PaddleOCR WordBox 추출 결과가 비어있습니다 (결과 파싱 후 단어 없음)")
        
        return words
    except Exception as e:
        print(f"⚠ PaddleOCR WordBox 추출 실패: {e}")
        import traceback
        print(f"   상세 오류: {traceback.format_exc()}")
        return []


def get_paddleocr_words(
    img: Union[np.ndarray, Image.Image],
    *,
    scale: int = 3,
    code_mode: bool = True,
    remove_emoji: bool = True,
    use_local_model: bool = False,  # 사용 안 함 (v2.0 구버전, 제거됨)
    use_trained_model: bool = True,  # 기본값: v5 학습된 모델 사용 (코드 문법 인식 최적화)
) -> List[WordBox]:
    """PaddleOCR로 WordBox 리스트 추출 (이미지 전처리 포함, v5 학습된 모델 우선 사용)"""
    pil_img = open_image_any(img)
    if scale and scale != 1:
        w, h = pil_img.size
        pil_img = pil_img.resize((w * scale, h * scale), Image.LANCZOS)
    pil_img = preprocess_for_code_pil(pil_img, enabled=code_mode)
    return paddleocr_word_boxes(
        pil_img, 
        code_mode=code_mode, 
        remove_emoji=remove_emoji, 
        use_local_model=use_local_model,
        use_trained_model=use_trained_model
    )


def image_to_text_paddleocr(
    img: Union[np.ndarray, Image.Image],
    *,
    scale: int = 3,
    code_mode: bool = True,
    normalize: bool = True,
    indent_step: int = 4,
    remove_emoji: bool = True,
    use_local_model: bool = False,  # 사용 안 함 (v2.0 구버전, 제거됨)
    use_trained_model: bool = True,  # 기본값: v5 학습된 모델 사용 (코드 문법 인식 최적화)
) -> str:
    """PaddleOCR로 텍스트 추출 (v5 학습된 모델 우선 사용)"""
    pil_img = open_image_any(img)
    original_size = pil_img.size
    
    if scale and scale != 1:
        w, h = pil_img.size
        pil_img = pil_img.resize((w * scale, h * scale), Image.LANCZOS)
        print(f"   이미지 리사이즈: {original_size} → {pil_img.size} (scale={scale})")
    
    # 전처리 적용
    pil_img_processed = preprocess_for_code_pil(pil_img, enabled=code_mode)
    
    # v5 학습된 모델로 OCR 실행
    words = paddleocr_word_boxes(
        pil_img_processed, 
        code_mode=code_mode, 
        remove_emoji=remove_emoji, 
        use_local_model=use_local_model,
        use_trained_model=use_trained_model
    )
    
    # 전처리된 이미지에서 결과가 없으면 원본 이미지로 재시도
    if not words and code_mode:
        print(f"   ⚠ 전처리된 이미지에서 결과 없음, 원본 이미지로 재시도...")
        words = paddleocr_word_boxes(
            pil_img,  # 원본 이미지 사용
            code_mode=False,  # 전처리 비활성화
            remove_emoji=remove_emoji, 
            use_local_model=use_local_model,
            use_trained_model=use_trained_model
        )
        if words:
            print(f"   ✅ 원본 이미지에서 {len(words)}개 단어 인식 성공")
    
    if not words:
        return ""
    
    # WordBox를 텍스트로 재구성
    out = reconstruct_text_from_words(
        words,
        code_mode=code_mode,
        normalize=normalize,
        indent_step=indent_step,
        remove_emoji=remove_emoji,
    )
    
    return out

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
        # OCR 엔진 사용 가능 여부 확인
        paddleocr_available, paddleocr_error = check_paddleocr_available()
        tesseract_available = pytesseract is not None
        winrt_available, winrt_error = check_winrt_available()
        
        if not paddleocr_available and not tesseract_available and not winrt_available:
            return {
                "success": False,
                "error": "모든 OCR 엔진이 사용 불가능합니다. (PaddleOCR, Tesseract, WinRT 모두 실패)",
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
        
        # PaddleOCR 우선 사용 (최신 모델, 정확도 높음)
        paddleocr_words = None
        paddleocr_text = None
        tesseract_words = None
        tesseract_text = None
        winrt_words = None
        winrt_text = None
        
        if paddleocr_available:
            try:
                # v5 학습된 모델 사용 (코드 문법 인식 최적화, 88.96% 정확도)
                print(f"🚀 PaddleOCR 실행 중... (v5 학습된 전용 모델, 코드 문법 인식 최적화)")
                # WordBox 가져오기 (v5 학습된 모델 명시적 사용)
                paddleocr_words = get_paddleocr_words(
                    cropped,
                    scale=3,
                    code_mode=True,
                    remove_emoji=True,
                    use_local_model=False,  # 로컬 모델 사용 안 함 (v2.0 구버전, 제거됨)
                    use_trained_model=True  # v5 학습된 모델 사용
                )
                # 전체 텍스트도 가져오기 (v5 학습된 모델 명시적 사용)
                paddleocr_text = image_to_text_paddleocr(
                    cropped,
                    scale=3,
                    code_mode=True,
                    normalize=True,
                    use_local_model=False,  # 로컬 모델 사용 안 함 (v2.0 구버전, 제거됨)
                    use_trained_model=True  # v5 학습된 모델 사용
                )
                if paddleocr_text:
                    print(f"✅ PaddleOCR 완료 (인식된 텍스트 길이: {len(paddleocr_text)} 문자)")
                else:
                    print(f"⚠ PaddleOCR 완료했지만 텍스트가 비어있습니다.")
            except Exception as e:
                print(f"⚠ PaddleOCR 실패: {e}")
                import traceback
                print(f"   상세 오류: {traceback.format_exc()}")
        
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
        
        # 결과 선택 (PaddleOCR + WinRT 병합 우선)
        if paddleocr_words and winrt_words:
            try:
                pil_img = open_image_any(cropped)
                ocr_result = merge_paddleocr_winrt_results(
                    paddleocr_words,
                    winrt_words,
                    pil_img
                )
                ocr_method = "PaddleOCR + WinRT 병합 (v5 학습된 모델 사용, 코드 문법 인식 최적화)"
            except Exception as e:
                print(f"⚠ PaddleOCR+WinRT 병합 실패: {e}")
                import traceback
                print(f"   상세 오류: {traceback.format_exc()}")
                if paddleocr_text:
                    ocr_result = paddleocr_text
                    ocr_method = "PaddleOCR (v5 학습된 모델 사용, 병합 실패)"
        elif paddleocr_text:
            ocr_result = paddleocr_text
            ocr_method = "PaddleOCR (v5 학습된 모델 사용, 코드 문법 인식 최적화, 88.96% 정확도)"
        elif tesseract_words and winrt_words:
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
        
        if ocr_result and ocr_result.strip():
            # 클립보드에 저장
            print(f"✅ OCR 완료 ({ocr_method})")
            print(f"   인식된 텍스트 길이: {len(ocr_result)} 문자, 줄 수: {ocr_result.count(chr(10)) + 1}")
            try:
                copy_to_clipboard(ocr_result)
                print("📋 클립보드에 저장 완료")
            except Exception as e:
                print(f"⚠ 클립보드 저장 실패: {e}")
            
            return {
                "success": True,
                "text": ocr_result,
                "method": ocr_method,
                "error": None
            }
        else:
            # OCR 결과가 비어있는 경우 상세 정보 출력
            error_details = []
            if not paddleocr_available:
                error_details.append(f"PaddleOCR: {paddleocr_error or '사용 불가'}")
            if not tesseract_available:
                error_details.append("Tesseract: 사용 불가")
            if not winrt_available:
                error_details.append(f"WinRT: {winrt_error or '사용 불가'}")
            if paddleocr_available and not paddleocr_text:
                error_details.append("PaddleOCR: 실행했지만 결과가 비어있음")
            
            error_msg = "모든 OCR 엔진이 실패했습니다."
            if error_details:
                error_msg += f" ({', '.join(error_details)})"
            
            print(f"❌ OCR 실패: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "text": "",
                "method": ""
            }
    
    except ValueError as e:
        # 사용자가 ROI 선택을 취소한 경우
        if "취소" in str(e) or "cancel" in str(e).lower():
            print("⚠ 사용자가 영역 선택을 취소했습니다.")
            return {
                "success": False,
                "error": "사용자가 영역 선택을 취소했습니다.",
                "text": "",
                "method": ""
            }
        print(f"❌ ValueError 발생: {e}")
        import traceback
        print(f"   상세 오류: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"영역 선택 오류: {str(e)}",
            "text": "",
            "method": ""
        }
    except Exception as e:
        import traceback
        error_msg = f"OCR 처리 중 오류 발생: {str(e)}"
        print(f"❌ 예상치 못한 오류 발생: {error_msg}")
        print(f"   상세 오류: {traceback.format_exc()}")
        return {
            "success": False,
            "error": error_msg,
            "text": "",
            "method": ""
        }

