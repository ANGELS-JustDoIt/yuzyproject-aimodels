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
    if not s:
        return s
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
            if gap_px <= char_w * 0.10:
                spaces = 0
            else:
                spaces = int(round(gap_px / max(1e-6, char_w)))
                spaces = clamp_int(spaces, 1, 80)
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
    try:
        import winrt  # noqa: F401
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

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
        tesseract_available = pytesseract is not None
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

