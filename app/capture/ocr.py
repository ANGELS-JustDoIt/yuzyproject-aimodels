# app/capture/ocr.py
# OCR Core
# - Main: Tesseract(image_to_data) + layout reconstruction
# - Aux : WinRT OCR (ko/en 2-pass) + word bbox -> layout reconstruction
# - Optional: EasyOCR / PaddleOCR wrappers (for tools/ocr_bench import compatibility)

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# =========================================================
# Tesseract path (optional)
# =========================================================
TESSERACT_EXE = r"C:\Pyg\Program_Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR = r"C:\Pyg\Program_Files\Tesseract-OCR\tessdata"

if pytesseract is not None and os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
if os.path.isdir(TESSDATA_DIR):
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR


# =========================================================
# Sanitizer
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


# =========================================================
# Preprocess
# =========================================================
def preprocess_for_code_pil(img: Image.Image, enabled: bool) -> Image.Image:
    if not enabled:
        return img

    g = img.convert("L")
    g = ImageEnhance.Contrast(g).enhance(1.7)
    g = ImageEnhance.Sharpness(g).enhance(1.4)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    return g.convert("RGB")


def open_image_any(img: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        if cv2 is None:
            raise RuntimeError("opencv-python(cv2)가 필요합니다.")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).convert("RGB")
    raise TypeError("Unsupported image type (need PIL.Image or OpenCV BGR ndarray).")


# =========================================================
# OCR boxes & layout
# =========================================================
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


# =========================================================
# normalize (code)
# =========================================================
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
    import re
    if not s:
        return s

    # ------------------------------------------------------------
    # 0) 흔한 문자 오인식 / 유니코드 기호 정리 (항상 안전)
    # ------------------------------------------------------------
    # if 오인식
    s = re.sub(r"\bi1f\b", "if", s)
    s = re.sub(r"\b1f\b", "if", s)

    # 'to'를 't0'로 오인식 (0)
    s = re.sub(r"\bt0\b", "to", s, flags=re.I)

    # WinRT/OCR에서 섞이는 유니코드 기호 정리
    s = s.replace("—", "-").replace("–", "-").replace("•", ".")
    s = s.replace("-〉", "->").replace("→", "->")

    # 하이픈이 유니코드 long dash로 들어오는 케이스가 많아서 use_gpu 보정
    s = s.replace("use-gpu", "use_gpu").replace("use—gpu", "use_gpu").replace("use–gpu", "use_gpu")

    # ------------------------------------------------------------
    # 1) 공통 glue / 식별자 안전 복원
    #    (COMMON_GLUE, SAFE_ID_REPAIRS 는 파일 상단에 정의돼있다고 가정)
    # ------------------------------------------------------------
    for pat, rep in COMMON_GLUE:
        s = pat.sub(rep, s)
    for pat, rep in SAFE_ID_REPAIRS:
        s = pat.sub(rep, s)

    # ------------------------------------------------------------
    # 2) out_winrt 전용/빈발 패턴 교정 (보수적)
    # ------------------------------------------------------------
    s = re.sub(r"\bb001\b", "bool", s, flags=re.I)
    s = re.sub(r"\bboo1\b", "bool", s, flags=re.I)
    s = re.sub(r"\bTup1e\b", "Tuple", s)
    s = s.replace("IMREAD COLOR", "IMREAD_COLOR")

    # scale 오인식
    s = re.sub(r"\bscaLe\b|\bsca1e\b", "scale", s)

    # == 깨짐: " = =" / "=  =" 등
    s = re.sub(r"\s=\s=", " ==", s)
    s = re.sub(r"==\s=", "==", s)

    # import( -> import (
    s = re.sub(r"\bimport\(", "import (", s)
    
    # defload_x -> def load_x (토큰 경계만)
    s = re.sub(r"\bdef(?=[A-Za-z_])", "def ", s)

    # ------------------------------------------------------------
    # 3) 엔진/대입 등 구조 토큰 보정 (아주 제한적으로)
    # ------------------------------------------------------------
    # engine 'tesseract' -> engine == "tesseract"
    s = re.sub(r"\bengine\s+1['\"]tesseract['\"]", 'engine == "tesseract"', s)
    s = re.sub(r'\bengine\s+["\']tesseract["\']', 'engine == "tesseract"', s, flags=re.I)

    # if engine tesseract -> if engine == "tesseract"
    s = re.sub(r"\bif\s+engine\s+tesseract\b", 'if engine == "tesseract"', s, flags=re.I)

    # "engine == "tesseract"" 단독 라인 -> if engine == "tesseract":
    s = re.sub(r'^\s*engine\s*==\s*"tesseract"\s*$', 'if engine == "tesseract":', s, flags=re.I)

    # text run_xxx( -> text = run_xxx(
    s = re.sub(r"\btext\s+(run_[A-Za-z_]\w*\()", r"text = \1", s)

    # ------------------------------------------------------------
    # 4) underscore 누락 복원 (확실한 케이스만)
    # ------------------------------------------------------------
    # image_to_text easyocr / paddleocr 처럼 한 칸 들어간 케이스
    s = re.sub(r"\bimage_to_text\s+easyocr\b", "image_to_text_easyocr", s, flags=re.I)
    s = re.sub(r"\bimage_to_text\s+paddleocr\b", "image_to_text_paddleocr", s, flags=re.I)

    # 'image to text' 류에서 '_' 누락/공백/'t0'까지 복원
    s = re.sub(r"\bimage\s*(?:to|t0)\s*text\b", "image_to_text", s, flags=re.I)
    s = re.sub(r"\bimage\s*_?\s*(?:to|t0)\s*text\s*_?\s*easyocr\b", "image_to_text_easyocr", s, flags=re.I)
    s = re.sub(r"\bimage\s*_?\s*(?:to|t0)\s*text\s*_?\s*paddleocr\b", "image_to_text_paddleocr", s, flags=re.I)

    # 자주 쓰는 python 식별자에서 '_' 빠진 케이스(정확히 이 단어들만)
    s = re.sub(r"\bload\s*bgr\b", "load_bgr", s, flags=re.I)
    s = re.sub(r"\brun\s*tesseract\b", "run_tesseract", s, flags=re.I)
    s = re.sub(r"\brun\s*easyocr\b", "run_easyocr", s, flags=re.I)
    s = re.sub(r"\brun\s*paddleocr\b", "run_paddleocr", s, flags=re.I)
    s = re.sub(r"\bbench\s*one\b", "bench_one", s, flags=re.I)

    # args.1ang -> args.lang
    s = re.sub(r"\bargs\.1ang\b", "args.lang", s, flags=re.I)

    # te time. perf_counter( ) -> t0 = time.perf_counter()
    s = re.sub(r"\bte\s*time\.\s*perf_counter\s*\(\s*\)", "t0 = time.perf_counter()", s, flags=re.I)

    # ------------------------------------------------------------
    # 5) 추가: 자주 깨지는 토큰 몇 개
    # ------------------------------------------------------------
    # use—gpu 같은 케이스 -> use_gpu
    s = re.sub(r"\buse\s*[-–—]\s*gpu\b", "use_gpu", s, flags=re.I)

    # -----------------------------
    # screen_capture.py에서 자주 터지는 패턴들
    # -----------------------------

    # tmport -> import
    s = re.sub(r"^\s*tmport\b", "import", s, flags=re.I)

    # mSS / mss 오인식
    s = re.sub(r"\bfrom\s+mSS\s+import\s+mSS\b", "from mss import mss", s)
    s = re.sub(r"\bmSS\b", "mss", s)
    s = re.sub(r"\bmSS\(\)", "mss()", s)

    # OpenCV 철자 (0penCV)
    s = s.replace("0penCV", "OpenCV").replace("0pencv", "OpenCV")

    # cvtColor 오인식
    s = re.sub(r"\bcvtC0?1?0r\b", "cvtColor", s, flags=re.I)

    # destroyAllWindows 오인식
    s = re.sub(r"destroyA11b\W*indows", "destroyAllWindows", s, flags=re.I)

    # Fullscreen 오인식
    s = re.sub(r'Fu11screen', "Fullscreen", s)

    # waitKey(ø) -> waitKey(0)
    s = s.replace("waitKey(ø)", "waitKey(0)").replace("waitKey(Ø)", "waitKey(0)")

    # __name__ == "__main__" 오인식들
    s = re.sub(r"^\s*name\s+maln\b", 'if __name__ == "__main__":', s, flags=re.I)
    s = re.sub(r"__maln__", "__main__", s, flags=re.I)

    # screen capture_fullscreen_bgr() -> screen = capture_fullscreen_bgr()
    s = re.sub(r"\bscreen\s+capture_fullscreen_bgr\s*\(\s*\)", "screen = capture_fullscreen_bgr()", s, flags=re.I)

    # monitor sct. monitors [x] -> monitor = sct.monitors[x]
    s = re.sub(r"\bmonitor\s+sct\.\s*monitors\s*\[\s*(.+?)\s*\]", r"monitor = sct.monitors[\1]", s)


    # ------------------------------------------------------------
    # 6) 공백 정리 (leading indent 유지)
    # ------------------------------------------------------------
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
    lines = out.splitlines()
    
    def _is_plain_doc_line(t: str) -> bool:
        tt = t.strip()
        if not tt:
            return False
        # 이미 코드/주석/문자열이면 제외
        if tt.startswith(("#", '"""', "'''", '"', "'")):
            return False
        # 한글 포함 + 문장 형태면 doc 후보
        return bool(re.search(r"[가-힣]", tt))

    i = 0
    while i < len(lines):
        if re.match(r"^\s*def\s+\w+\s*\(.*\)\s*:", lines[i]):
            def_indent = len(lines[i]) - len(lines[i].lstrip(" "))
            body_indent = def_indent + indent_step

            j = i + 1
            doc_lines: List[str] = []
            while j < len(lines):
                ln = lines[j]
                if not ln.strip():
                    break

                cur_indent = len(ln) - len(ln.lstrip(" "))
                if cur_indent < body_indent:
                    break

                # 다음 코드 시작이면 중단
                if re.match(r"^\s*(with|if|elif|else:|for|while|try:|except\b|finally:|return\b|raise\b|class\b|def\b)\b", ln):
                    break

                if not _is_plain_doc_line(ln):
                    break

                doc_lines.append(ln.strip())
                j += 1

            if doc_lines:
                # 평문 doc 라인 제거
                del lines[i + 1 : i + 1 + len(doc_lines)]

                # docstring 삽입
                insert = [(" " * body_indent) + '"""']
                insert += [(" " * body_indent) + dl for dl in doc_lines]
                insert += [(" " * body_indent) + '"""']

                for k, v in enumerate(insert):
                    lines.insert(i + 1 + k, v)

                i = i + 1 + len(insert)
                continue

        i += 1


    for i, ln in enumerate(lines):
        if re.match(r"^\s*from\s+.+\s+import\s*\(\s*$", ln):
            # 이미 닫는 ')'가 있으면 패스
            if any(re.match(r"^\s*\)\s*$", x) for x in lines[i+1:]):
                break

            insert_at = None
            for j in range(i + 1, len(lines)):
                # 다음 def/class 만나기 전까지가 import 블록이라고 가정
                if re.match(r"^\s*(def|class)\s+", lines[j]):
                    insert_at = j
                    break

            if insert_at is None:
                # 파일 끝까지 def/class가 없으면 맨 끝에 삽입
                insert_at = len(lines)

            lines.insert(insert_at, ")")
            break

    out = "\n".join(lines).rstrip() + "\n"

    # 파일 헤더처럼 보이는데 '#'가 빠진 경우 보정
    first = out.splitlines()[0] if out.strip() else ""
    if first and first.lstrip().startswith("app/capture/") and not first.lstrip().startswith("#"):
        out = "# " + first.lstrip() + "\n" + "\n".join(out.splitlines()[1:]) + "\n"

    return sanitize_text(out, remove_emoji=remove_emoji, keep_newlines=True, collapse_spaces=False) + "\n"



# =========================================================
# WinRT
# =========================================================
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


# =========================================================
# Tesseract main (image_to_data -> layout)
# =========================================================
def _build_whitelist(code_mode: bool, lang: str) -> Optional[str]:
    l = (lang or "").lower()
    if not code_mode:
        return None
    if "kor" in l or "korean" in l or l.startswith("ko"):
        return None
    return (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_-+=/*%<>!&|^~.,:;?@#$()[]{}\\'`\""
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
    scale: int = 3,
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
        pil_img = pil_img.resize((w * scale, h * scale), Image.BICUBIC)

    pil_img = preprocess_for_code_pil(pil_img, enabled=code_mode)

    if not layout:
        if pytesseract is None:
            raise RuntimeError("pytesseract not installed")
        whitelist = _build_whitelist(code_mode=code_mode, lang=lang)
        config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
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
# Optional wrappers for tools/ocr_bench import compatibility
# =========================================================
def image_to_text_easyocr(
    img: Union[np.ndarray, Image.Image],
    *,
    langs: List[str],
    gpu: bool = False,
    remove_emoji: bool = True,
) -> str:
    try:
        import easyocr
    except Exception as e:
        raise RuntimeError(f"EasyOCR not installed/import failed: {e}")

    pil_img = open_image_any(img)
    rgb = np.array(pil_img.convert("RGB"))
    reader = easyocr.Reader(langs, gpu=gpu)
    # detail=0 -> list[str]
    res = reader.readtext(rgb, detail=0, paragraph=False)
    text = "\n".join(res)
    return sanitize_text(text, remove_emoji=remove_emoji, keep_newlines=True, collapse_spaces=True) + "\n"


def image_to_text_paddleocr(
    img: Union[np.ndarray, Image.Image],
    *,
    lang: str,
    use_gpu: bool = False,
    remove_emoji: bool = True,
) -> str:
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        raise RuntimeError(f"PaddleOCR not installed/import failed: {e}")

    pil_img = open_image_any(img)
    rgb = np.array(pil_img.convert("RGB"))
    ocr = PaddleOCR(lang=lang, use_angle_cls=False, use_gpu=use_gpu)
    out = ocr.ocr(rgb, cls=False)

    lines: List[str] = []
    for page in out:
        for item in page:
            try:
                lines.append(item[1][0])
            except Exception:
                pass

    text = "\n".join(lines)
    return sanitize_text(text, remove_emoji=remove_emoji, keep_newlines=True, collapse_spaces=True) + "\n"


def debug_loaded_info() -> str:
    return (
        f"ocr.py path={__file__}\n"
        f"pytesseract={'OK' if pytesseract is not None else 'NONE'}\n"
        f"cv2={'OK' if cv2 is not None else 'NONE'}\n"
        f"TESSDATA_PREFIX={os.environ.get('TESSDATA_PREFIX')}\n"
    )
