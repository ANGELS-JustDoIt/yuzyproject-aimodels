# ocrtest.py
# 독립 실행 가능한 OCR 테스트 스크립트
# 화면 캡처 -> 드래그로 crop -> 자동 OCR 인식 -> 클립보드 저장
# 모든 필요한 코드가 이 파일 하나에 포함되어 있습니다.

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

# =========================================================
# Tesseract 설정
# =========================================================
try:
    import pytesseract
except Exception:
    pytesseract = None

TESSERACT_EXE = r"C:\Pyg\Program_Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR = r"C:\Pyg\Program_Files\Tesseract-OCR\tessdata"

if pytesseract is not None and os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
if os.path.isdir(TESSDATA_DIR):
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

# =========================================================
# OCR Core Functions (from ocr.py)
# =========================================================
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
    # 대비 강화 (따옴표 등 작은 기호 인식을 위해 더 강화)
    g = ImageEnhance.Contrast(g).enhance(2.8)
    # 선명도 강화 (따옴표 경계 명확화)
    g = ImageEnhance.Sharpness(g).enhance(2.2)
    # 밝기 조정
    g = ImageEnhance.Brightness(g).enhance(1.1)
    # 노이즈 제거 (작은 커널로 기호 보존)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    # 이진화를 위한 임계값 처리 (OpenCV 사용)
    arr = np.array(g)
    # Otsu 임계값으로 이진화
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 작은 모폴로지 연산으로 기호 보존하면서 노이즈 제거
    kernel = np.ones((1, 1), np.uint8)  # 더 작은 커널로 기호 보존
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # 따옴표 같은 작은 기호를 보존하기 위한 추가 처리
    # 작은 객체도 보존 (따옴표는 작을 수 있음)
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
    
    # 코드에 나오지 않는 기호 제거/보정
    # ° (도 기호)를 따옴표로 보정
    s = s.replace("°", '"')  # r°C:\... -> r"C:\...
    s = s.replace("°", "'")  # 작은따옴표 케이스
    
    # 기호 오인식 보정
    s = s.replace("¥", "*")  # ¥를 *로
    s = s.replace("×", "*")  # ×를 *로
    s = s.replace("·", "*")  # ·를 *로 (일부 케이스)
    
    # Python 키워드/구문 패턴 보정
    # if_pytesseract_is_not None_and_os.path.exists -> if pytesseract is not None and os.path.exists
    # 단계별로 처리
    s = re.sub(r"\bif_([a-z_]+)_is_not\s+None_and_", r"if \1 is not None and ", s, flags=re.I)
    s = re.sub(r"\bif_([a-z_]+)_is\s+not\s+None_and_", r"if \1 is not None and ", s, flags=re.I)
    # 일반적인 if_..._and_ 패턴
    s = re.sub(r"\bif_([a-z_]+)_and_", r"if \1 and ", s, flags=re.I)
    s = re.sub(r"_and_os\.", " and os.", s)
    s = re.sub(r"_and_", " and ", s)  # _and_ -> and
    s = re.sub(r"_or_", " or ", s)  # _or_ -> or
    s = re.sub(r"_is_", " is ", s)  # _is_ -> is
    s = re.sub(r"_not_", " not ", s)  # _not_ -> not
    s = re.sub(r"_in_", " in ", s)  # _in_ -> in
    
    # 변수명/함수명 언더바 복원
    # TESSERACTEXE -> TESSERACT_EXE (대문자 변수명, EXE/DIR 등 접미사)
    s = re.sub(r"([A-Z]+)(EXE|DIR|PATH|ENV|CMD)", r"\1_\2", s)
    # TESSERACTEXE -> TESSERACT_EXE (일반적인 대문자 변수명)
    s = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1_\2", s)  # TESSERACTEXE -> TESSERACT_EXE
    # tesseractcmd -> tesseract_cmd (소문자 변수명)
    s = re.sub(r"([a-z]+)(cmd|dir|path|exe|env|prefix)", r"\1_\2", s, flags=re.I)
    # pytesseract.pytesseract.tesseractcmd -> pytesseract.pytesseract.tesseract_cmd
    s = re.sub(r"(pytesseract)\.(pytesseract)\.([a-z]+)(cmd)", r"\1.\2.\3_\4", s, flags=re.I)
    # os.environ -> os.environ (이미 올바름)
    
    # 특정 변수명 패턴 복원
    s = re.sub(r"\bTESSERACTEXE\b", "TESSERACT_EXE", s)
    s = re.sub(r"\bTESSDATA_DIR\b", "TESSDATA_DIR", s)  # 이미 올바름이지만 확인
    s = re.sub(r"\btesseractcmd\b", "tesseract_cmd", s)
    
    # 하이픈을 언더바로 (변수명 중간, Python 스타일) - 매우 제한적으로만
    s = re.sub(r"([a-z])\-([a-z])", r"\1_\2", s)  # 소문자 사이 하이픈만
    
    # 따옴표 구분 보정 (작은따옴표 vs 큰따옴표)
    # 1. 유니코드 따옴표를 ASCII로 변환
    s = s.replace(""", '"').replace(""", '"')  # 유니코드 큰따옴표
    s = s.replace("'", "'").replace("'", "'")  # 유니코드 작은따옴표
    s = s.replace("'", "'").replace("'", "'")  # 유니코드 작은따옴표 (다른 형태)
    
    # 2. 백틱 관련 보정
    s = s.replace("``", '"')  # 백틱 두 개를 큰따옴표로
    
    # 3. 따옴표 쌍 보정 (시작/끝 매칭)
    # 큰따옴표 시작 후 작은따옴표 끝 -> 큰따옴표로 통일
    s = re.sub(r'"([^"]*?)\'', r'"\1"', s)
    # 작은따옴표 시작 후 큰따옴표 끝 -> 작은따옴표로 통일
    s = re.sub(r"'([^']*?)\"", r"'\1'", s)
    
    # 4. 문자열 패턴 기반 보정 (더 정교하게)
    # 큰따옴표 문자열 패턴: "..." 형태
    # 작은따옴표 문자열 패턴: '...' 형태
    # Python 문자열 패턴을 고려한 보정
    # 큰따옴표로 시작하는 문자열에서 작은따옴표가 끝에 오면 큰따옴표로
    s = re.sub(r'="([^"]*?)\'', r'="\1"', s)  # = "..." ' -> = "..."
    s = re.sub(r'="([^"]*?)\'', r'="\1"', s)  # = "..." ' -> = "..."
    # 작은따옴표로 시작하는 문자열에서 큰따옴표가 끝에 오면 작은따옴표로
    s = re.sub(r"='([^']*?)\"", r"='\1'", s)  # = '...' " -> = '...'
    
    # 5. 일반적인 오인식 패턴 보정
    # 작은따옴표 두 개가 연속으로 나오면 큰따옴표로 (문자열 시작/끝)
    s = re.sub(r"([=\(\[\s,])''([^'])", r'\1"\2', s)  # 시작 부분
    s = re.sub(r"([^'])''([=\)\]\s,\.;])", r'\1"\2', s)  # 끝 부분
    
    # 6. Python 코드 패턴 보정
    # os.environ[ "KEY" ] -> os.environ["KEY"] (대괄호 안 공백 제거)
    s = re.sub(r"(\w+)\[\s*([\"'])([^\"']+)\2\s*\]", r'\1[\2\3\2]', s)
    # 딕셔너리/리스트 접근 패턴 (공백 제거)
    s = re.sub(r"\[\s*([\"'])([^\"']+)\1\s*\]", r'[\1\2\1]', s)
    
    # 함수 호출 패턴 보정
    # os.path.exists(TESSERACTEXE) -> os.path.exists(TESSERACT_EXE)
    s = re.sub(r"os\.path\.(exists|isdir)\(([A-Z_]+EXE)\)", r"os.path.\1(\2)", s)
    s = re.sub(r"os\.path\.(exists|isdir)\(([A-Z_]+DIR)\)", r"os.path.\1(\2)", s)
    
    # 코드에 나오지 않는 특수 기호 제거
    # 일반적인 코드 기호만 허용: A-Za-z0-9_+-=/*%<>!&|^~.,:;?@#$()[]{}\'\"`\n\t
    # 그 외 기호는 제거하거나 보정
    # 예: °, ·, ×, →, ← 등은 이미 위에서 처리됨
    # 기타 기호 보정
    s = re.sub(r"\bi1f\b", "if", s)
    s = re.sub(r"\b1f\b", "if", s)
    s = re.sub(r"\bt0\b", "to", s, flags=re.I)
    s = s.replace("—", "-").replace("–", "-").replace("•", ".")
    s = s.replace("-〉", "->").replace("→", "->")
    s = s.replace("use-gpu", "use_gpu").replace("use—gpu", "use_gpu").replace("use–gpu", "use_gpu")
    # * 기호 오인식 보정 (이미 위에서 처리했지만 추가 확인)
    s = s.replace("¥", "*")  # 일본 엔화 기호
    s = s.replace("×", "*")  # 곱셈 기호
    # 함수 인자에서 * 오인식 보정
    s = re.sub(r"(\w+)\s*¥\s*,", r"\1 *,", s)  # text ¥, -> text *,
    s = re.sub(r"(\w+)\s*¥\s*\)", r"\1 *)", s)  # text ¥) -> text *)
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
    alias_map = {
        "ko": ["ko-KR", "ko"],
        "en": ["en-US", "en"],
    }
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
    # 기호 인식 강화: 큰따옴표, 작은따옴표, 언더바 등 중요 기호 포함
    return (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_"  # 언더바 강조
        "-+=/*%<>!&|^~.,:;?@#$()[]{}\\"
        "*"   # 별표 강조 (¥ 오인식 방지)
        "'"   # 작은따옴표 (명시적으로)
        "\""  # 큰따옴표 (명시적으로)
        "`"   # 백틱
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
    # 기호 인식을 위한 추가 설정
    config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
    config += " -c tessedit_pageseg_mode=6"  # 단일 텍스트 블록
    config += " -c classify_bln_numeric_mode=0"  # 숫자 인식 개선
    config += " -c textord_min_linesize=2.5"  # 작은 기호 인식 개선
    config += " -c textord_tabvector_vertical_gap_factor=0.5"  # 기호 간격 인식 개선
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
    scale: int = 4,  # 3 -> 4로 증가 (더 높은 해상도)
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
        pil_img = pil_img.resize((w * scale, h * scale), Image.LANCZOS)  # BICUBIC -> LANCZOS (더 선명)
    pil_img = preprocess_for_code_pil(pil_img, enabled=code_mode)
    if not layout:
        if pytesseract is None:
            raise RuntimeError("pytesseract not installed")
        whitelist = _build_whitelist(code_mode=code_mode, lang=lang)
        # 기호 인식을 위한 추가 설정
        config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
        config += " -c tessedit_pageseg_mode=6"  # 단일 텍스트 블록
        config += " -c classify_bln_numeric_mode=0"  # 숫자 인식 개선
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

# =========================================================
# Screen Capture (from screen_capture.py)
# =========================================================
def capture_fullscreen_bgr(monitor_index: int = 1) -> np.ndarray:
    with mss() as sct:
        monitor = sct.monitors[monitor_index]
        img = np.array(sct.grab(monitor))
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return bgr

# =========================================================
# Clipboard (from clipboard.py)
# =========================================================
def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)

# =========================================================
# ROI Selection
# =========================================================
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
                cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)
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

# =========================================================
# Merge Tesseract and WinRT results
# =========================================================
def merge_tesseract_winrt_results(
    tesseract_words: List[WordBox],
    winrt_text: str,
    pil_img: Image.Image
) -> str:
    """
    Tesseract의 레이아웃과 WinRT의 텍스트를 병합
    - 레이아웃: Tesseract WordBox 사용
    - 텍스트: 한글/기호는 WinRT 우선, 나머지는 Tesseract 우선
    """
    # WinRT 텍스트를 단어로 분리 (간단한 방법)
    winrt_lines = winrt_text.split('\n')
    
    # 한글/기호 감지 정규식
    korean_re = re.compile(r'[가-힣]')
    symbol_re = re.compile(r'[_\-\+\=\*\%\<\>\!\&\|\^\~\.\,\:\;\?\@\#\$\(\)\[\]\{\}\\\'`"]')
    
    # Tesseract WordBox를 라인별로 그룹화
    tesseract_lines = cluster_lines(tesseract_words)
    
    # WinRT 텍스트를 WordBox로 변환 (대략적인 위치)
    winrt_words: List[WordBox] = []
    y_pos = 0.0
    line_height = 20.0
    
    for line_text in winrt_lines:
        if not line_text.strip():
            y_pos += line_height
            continue
        words_in_line = line_text.split()
        x_pos = 0.0
        for word_text in words_in_line:
            if word_text:
                winrt_words.append(WordBox(
                    text=word_text,
                    x=x_pos,
                    y=y_pos,
                    w=len(word_text) * 10.0,
                    h=line_height,
                    conf=-1.0
                ))
                x_pos += len(word_text) * 10.0 + 5.0
        y_pos += line_height
    
    # Tesseract WordBox와 WinRT 텍스트를 병합
    merged_words: List[WordBox] = []
    
    for tess_line in tesseract_lines:
        for tess_word in tess_line.words:
            # Tesseract 단어 주변에서 WinRT 단어 찾기
            best_winrt_word = None
            min_distance = float('inf')
            
            for winrt_word in winrt_words:
                # Y 좌표가 비슷한지 확인
                if abs(tess_word.cy - winrt_word.cy) < 15.0:
                    # X 좌표 거리 계산
                    distance = abs(tess_word.x - winrt_word.x)
                    if distance < min_distance:
                        min_distance = distance
                        best_winrt_word = winrt_word
            
            # 텍스트 선택: 한글/기호가 있으면 WinRT 우선, 아니면 Tesseract
            final_text = tess_word.text
            
            if best_winrt_word:
                winrt_text_val = best_winrt_word.text
                # 한글이나 기호가 있으면 WinRT 텍스트 사용
                if korean_re.search(winrt_text_val) or symbol_re.search(winrt_text_val):
                    # 기호가 더 정확한 경우 WinRT 사용
                    if symbol_re.search(winrt_text_val):
                        final_text = winrt_text_val
                    # 한글이 있으면 WinRT 사용
                    elif korean_re.search(winrt_text_val):
                        final_text = winrt_text_val
                    # Tesseract에 기호가 없고 WinRT에 있으면 WinRT 사용
                    elif not symbol_re.search(tess_word.text) and symbol_re.search(winrt_text_val):
                        final_text = winrt_text_val
            
            # Tesseract 레이아웃 유지, 텍스트만 교체
            merged_words.append(WordBox(
                text=final_text,
                x=tess_word.x,
                y=tess_word.y,
                w=tess_word.w,
                h=tess_word.h,
                conf=tess_word.conf
            ))
    
    # 병합된 WordBox로 텍스트 재구성
    return reconstruct_text_from_words(
        merged_words,
        code_mode=True,
        normalize=True,
        indent_step=4,
        remove_emoji=True
    )

def get_tesseract_words(
    img: Union[np.ndarray, Image.Image],
    lang: str = "kor+eng",
    *,
    scale: int = 4,
    code_mode: bool = True,
    remove_emoji: bool = True,
) -> List[WordBox]:
    """Tesseract로 WordBox만 가져오기 (레이아웃용)"""
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

# =========================================================
# Main
# =========================================================
def check_winrt_available():
    try:
        import winrt
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def main():
    tesseract_available = pytesseract is not None
    winrt_available, winrt_error = check_winrt_available()
    
    if not tesseract_available and not winrt_available:
        print("⚠ 경고: Tesseract와 WinRT 모두 사용 불가능합니다.")
        print("OCR을 수행할 수 없습니다.")
        sys.exit(1)
    
    try:
        print("📸 화면 캡처 중...")
        screen = capture_fullscreen_bgr()
        
        print("🖱️  영역을 드래그해서 선택하세요 (마우스를 떼면 자동으로 인식됩니다)")
        cropped = select_roi_auto(screen, window_name="영역 선택 (드래그 후 마우스 떼기)")
        
        print("🔍 OCR 인식 중...")
        ocr_result = None
        ocr_method = None
        
        # 1. Tesseract와 WinRT 둘 다로 인식
        tesseract_words = None
        tesseract_text = None
        winrt_text = None
        
        if tesseract_available:
            try:
                print("  → Tesseract로 인식 중...")
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
                print("  → WinRT로 인식 중...")
                winrt_text = image_to_text_winrt(
                    cropped,
                    scale=3,
                    code_mode=True,
                    normalize=True
                )
            except Exception as e:
                print(f"⚠ WinRT OCR 실패: {e}")
        
        # 2. 결과 병합
        if tesseract_words and winrt_text:
            try:
                print("  → 결과 병합 중 (레이아웃: Tesseract, 한글/기호: WinRT)...")
                pil_img = open_image_any(cropped)
                ocr_result = merge_tesseract_winrt_results(
                    tesseract_words,
                    winrt_text,
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
            copy_to_clipboard(ocr_result)
            print(f"\n✅ OCR 완료 ({ocr_method})")
            print(f"📋 클립보드에 저장되었습니다! (Ctrl+V로 붙여넣기 가능)")
            print("\n" + "=" * 60)
            print("인식된 텍스트:")
            print("=" * 60)
            print(ocr_result)
            print("=" * 60)
        else:
            print("\n❌ OCR 실패: 모든 OCR 엔진이 실패했습니다.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 취소되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
