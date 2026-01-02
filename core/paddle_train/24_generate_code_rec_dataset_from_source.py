"""
PaddleOCR 학습용 REC 데이터셋 생성 스크립트 (단어 단위 분할 및 기호 학습 강화)

워크스페이스 내 모든 코드 파일에서 단어 및 짧은 구문 단위로 추출하여
30,000개의 고품질 Recognition 이미지를 생성합니다.

특징:
- 단어 단위 분할 (공백 기준)
- 기호 집중 학습 (30% 이상)
- 다양한 길이 조합 (1~3개 단어, 최대 15~20자)
- 글자 찌그러짐 방지 (폰트 크기 자동 조정)
- 영문 전용 (한글 제외)
"""

import sys
import os
import random
import re
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

# chardet 라이브러리 (인코딩 자동 감지)
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    print("[WARN] chardet이 설치되지 않았습니다. UTF-8로만 읽습니다.")

# Pygments 관련
try:
    from pygments.lexers import get_lexer_by_name
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    print("[WARN] Pygments가 설치되지 않았습니다. 문법 강조 없이 진행합니다.")

# UTF-8 출력 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent
CORE_TRAIN_DIR = Path(__file__).parent
TRAIN_DATA_DIR = CORE_TRAIN_DIR / "train_data"
REC_DIR = TRAIN_DATA_DIR / "rec"
REC_TRAIN_IMAGES_DIR = REC_DIR / "train"
REC_VAL_IMAGES_DIR = REC_DIR / "val"
REC_GT_TRAIN_FILE = REC_DIR / "rec_gt_train.txt"
REC_GT_VAL_FILE = REC_DIR / "rec_gt_test.txt"

# 목표 개수
TARGET_TRAIN_COUNT = 25000
TARGET_VAL_COUNT = 5000

# 제외할 디렉토리/파일
EXCLUDE_DIRS = {
    '__pycache__', 'node_modules', '.git', 'venv', 'venv_ocr',
    'outputs', 'output', 'train_data', 'code_syntax_dataset',
    '.next', 'dist', 'build', '.venv'
}

EXCLUDE_FILES = {
    'rec_gt.txt', 'rec_gt_train.txt', 'rec_gt_test.txt'
}

# 지원하는 파일 확장자
CODE_EXTENSIONS = {'.py', '.js', '.html', '.css', '.yml', '.yaml', '.ts', '.tsx', '.mjs'}

# 파일 확장자별 Lexer 매핑
LEXER_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.mjs': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.html': 'html',
    '.css': 'css',
    '.yml': 'yaml',
    '.yaml': 'yaml',
}

# 폰트 우선순위 (영문 코딩 전용 폰트)
FONT_PRIORITY = [
    'Fira Code',
    'JetBrains Mono',
    'Consolas',
    'Courier New',
    'Lucida Console',
    'Monaco',
    'Menlo',
    'DejaVu Sans Mono',
]

# 특수 기호 집중 학습용 기호
SYMBOLS = ['(', ')', '{', '}', '[', ']', '.', ':', ';', '=', '+', '-', '*', '/', '<', '>', '|', '_', '&', '$', '@', '!', '?', '%', '^', '~', '`', '#', '\\', '"', "'", ',', '\\n', '\\t']

# 테마 설정
DARK_THEMES = [
    {
        'bg': (30, 30, 30),
        'text': (212, 212, 212),
        'comment': (106, 153, 85),
        'keyword': (86, 156, 214),
        'string': (206, 145, 120),
        'number': (181, 206, 168),
        'name': (156, 220, 254),
    },
    {
        'bg': (25, 25, 35),
        'text': (200, 200, 220),
        'comment': (100, 150, 100),
        'keyword': (80, 150, 210),
        'string': (200, 140, 115),
        'number': (175, 200, 165),
        'name': (150, 215, 250),
    },
    {
        'bg': (28, 28, 28),
        'text': (248, 248, 242),
        'comment': (117, 113, 94),
        'keyword': (249, 38, 114),
        'string': (230, 219, 116),
        'number': (174, 129, 255),
        'name': (102, 217, 239),
    },
]

LIGHT_THEMES = [
    {
        'bg': (255, 255, 255),
        'text': (30, 30, 30),
        'comment': (0, 128, 0),
        'keyword': (0, 0, 255),
        'string': (163, 21, 21),
        'number': (0, 128, 128),
        'name': (0, 0, 128),
    },
]


def contains_hangul(text: str) -> bool:
    """한글 포함 여부 확인"""
    return any('\uac00' <= char <= '\ud7a3' for char in text)


def has_symbol(text: str) -> bool:
    """특수 기호 포함 여부 확인"""
    return any(sym in text for sym in SYMBOLS)


def find_code_files(root: Path) -> List[Path]:
    """워크스페이스 내 모든 코드 파일 찾기"""
    code_files = []
    for ext in CODE_EXTENSIONS:
        for file_path in root.rglob(f'*{ext}'):
            if any(excluded in file_path.parts for excluded in EXCLUDE_DIRS):
                continue
            if file_path.name in EXCLUDE_FILES:
                continue
            if file_path.is_file() and file_path.stat().st_size > 0:
                code_files.append(file_path)
    return code_files


def extract_code_lines(file_path: Path) -> List[str]:
    """파일에서 코드 라인 추출 (영문 전용)"""
    lines = []
    try:
        if CHARDET_AVAILABLE:
            rawdata = file_path.read_bytes()
            result = chardet.detect(rawdata)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            try:
                content = rawdata.decode('utf-8-sig', errors='replace')
            except:
                try:
                    content = rawdata.decode(encoding, errors='replace')
                except:
                    content = rawdata.decode('utf-8', errors='replace')
        else:
            try:
                content = file_path.read_text(encoding='utf-8-sig', errors='replace')
            except:
                content = file_path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return lines
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or len(line) < 2:
            continue
        if len(line) > 200:
            continue
        # 한글 제외
        if contains_hangul(line):
            continue
        lines.append(line)
    
    return lines


def split_into_words_and_phrases(line: str) -> List[str]:
    """
    라인을 단어 및 짧은 구문으로 분할
    - 단일 단어
    - 단어+기호 조합
    - 기호+단어 조합
    - 2~3개 단어 조합
    """
    segments = []
    
    # 공백 기준으로 분할
    words = line.split()
    
    if not words:
        return segments
    
    # 1. 단일 단어/기호
    for word in words:
        if len(word) <= 20:
            segments.append(word)
    
    # 2. 2개 단어 조합
    for i in range(len(words) - 1):
        combo = words[i] + ' ' + words[i+1]
        if len(combo) <= 20:
            segments.append(combo)
    
    # 3. 3개 단어 조합
    for i in range(len(words) - 2):
        combo = words[i] + ' ' + words[i+1] + ' ' + words[i+2]
        if len(combo) <= 20:
            segments.append(combo)
    
    # 4. 들여쓰기 포함 (원본 라인의 앞 공백 유지)
    if line and line[0] == ' ':
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces > 0 and leading_spaces <= 8:
            for word in words[:3]:  # 앞 3개 단어만
                indented = ' ' * leading_spaces + word
                if len(indented) <= 20:
                    segments.append(indented)
    
    return segments


def generate_symbol_focused_segments() -> List[str]:
    """기호 집중 학습용 세그먼트 생성"""
    segments = []
    
    # 기호만
    for sym in SYMBOLS[:20]:  # 주요 기호만
        segments.append(sym)
    
    # 기호 조합
    symbol_pairs = [
        '()', '{}', '[]', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/=',
        '->', '=>', '::', '...', '/*', '*/', '//', '/*', '*/'
    ]
    segments.extend(symbol_pairs)
    
    # 기호 + 단어
    common_words = ['if', 'for', 'while', 'def', 'class', 'import', 'from', 'return', 'const', 'let', 'var']
    for word in common_words:
        for sym in ['(', '{', '[', '.', ':', '=']:
            segments.append(sym + word)
            segments.append(word + sym)
            segments.append(sym + ' ' + word)
            segments.append(word + ' ' + sym)
    
    return segments


def load_font(font_size: int) -> Optional[ImageFont.FreeTypeFont]:
    """폰트 로드"""
    for font_name in FONT_PRIORITY:
        try:
            if sys.platform == 'win32':
                font_paths = [
                    f"C:/Windows/Fonts/{font_name}.ttf",
                    f"C:/Windows/Fonts/{font_name}.otf",
                    f"C:/Windows/Fonts/{font_name.replace(' ', '')}.ttf",
                    f"C:/Windows/Fonts/{font_name.replace(' ', '')}.otf",
                ]
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        try:
                            return ImageFont.truetype(font_path, font_size)
                        except:
                            continue
            try:
                return ImageFont.truetype(font_name, font_size)
            except:
                continue
        except:
            continue
    try:
        return ImageFont.load_default()
    except:
        return None


def tokenize_with_pygments(code: str, file_ext: str) -> List[Tuple[str, str]]:
    """Pygments로 토큰화"""
    if not PYGMENTS_AVAILABLE:
        return [('Text', code)]
    try:
        lexer_name = LEXER_MAP.get(file_ext, 'text')
        lexer = get_lexer_by_name(lexer_name)
        tokens = list(lexer.get_tokens(code))
        return [(str(token_type), text) for token_type, text in tokens]
    except:
        return [('Text', code)]


def get_token_color(token_type: str, theme: dict) -> tuple:
    """토큰 타입에 따른 색상 반환"""
    token_str = str(token_type)
    if 'Comment' in token_str:
        return theme.get('comment', theme['text'])
    elif 'Keyword' in token_str or 'Name.Builtin' in token_str:
        return theme.get('keyword', theme['text'])
    elif 'String' in token_str:
        return theme.get('string', theme['text'])
    elif 'Number' in token_str:
        return theme.get('number', theme['text'])
    elif 'Name' in token_str:
        return theme.get('name', theme['text'])
    else:
        return theme['text']


def render_text_with_syntax(
    text: str,
    file_ext: str,
    font: ImageFont.FreeTypeFont,
    theme: dict,
    target_height: int = 48,
    max_width: int = 640,
    padding: int = 10
) -> Image.Image:
    """
    문법 강조를 적용한 텍스트 이미지 생성
    글자가 찌그러지지 않도록 폰트 크기와 캔버스 크기를 정교하게 계산
    """
    # Pygments로 토큰화
    tokens = tokenize_with_pygments(text, file_ext)
    
    # 텍스트 크기 측정
    temp_img = Image.new("RGB", (2000, 200), theme['bg'])
    temp_draw = ImageDraw.Draw(temp_img)
    
    total_width = 0
    max_token_height = 0
    
    for token_type, token_text in tokens:
        bbox = temp_draw.textbbox((0, 0), token_text, font=font)
        token_width = bbox[2] - bbox[0]
        token_height = bbox[3] - bbox[1]
        total_width += token_width
        max_token_height = max(max_token_height, token_height)
    
    # 폰트 크기 조정 (텍스트가 이미지에 맞도록)
    available_width = max_width - (padding * 2)
    available_height = target_height - (padding * 2)
    
    # 스케일 계산 (더 여유 있게)
    if total_width > available_width or max_token_height > available_height:
        scale_w = available_width / total_width if total_width > 0 else 1.0
        scale_h = available_height / max_token_height if max_token_height > 0 else 1.0
        scale = min(scale_w, scale_h, 1.0) * 0.90  # 10% 여유
        
        try:
            current_size = font.size if hasattr(font, 'size') else 16
        except:
            current_size = 16
        new_size = max(10, int(current_size * scale))
        new_font = load_font(new_size)
        if new_font is not None:
            font = new_font
        
        # 다시 측정
        temp_draw = ImageDraw.Draw(temp_img)
        total_width = 0
        max_token_height = 0
        for token_type, token_text in tokens:
            bbox = temp_draw.textbbox((0, 0), token_text, font=font)
            token_width = bbox[2] - bbox[0]
            token_height = bbox[3] - bbox[1]
            total_width += token_width
            max_token_height = max(max_token_height, token_height)
    
    # 최종 이미지 크기 계산 (글자 찌그러짐 방지)
    final_width = min(max_width, int(total_width + padding * 2 + 4))
    final_height = target_height
    
    img = Image.new("RGB", (final_width, final_height), theme['bg'])
    draw = ImageDraw.Draw(img)
    
    # 텍스트 렌더링
    x_pos = padding
    y_pos = (final_height - max_token_height) // 2
    
    for token_type, token_text in tokens:
        color = get_token_color(token_type, theme)
        draw.text((x_pos, y_pos), token_text, fill=color, font=font)
        bbox = draw.textbbox((x_pos, y_pos), token_text, font=font)
        x_pos = bbox[2]
    
    return img


def enhance_image_quality(img: Image.Image) -> Image.Image:
    """이미지 품질 향상"""
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    return img


def is_valid_segment(segment: str) -> bool:
    """세그먼트 유효성 검증"""
    # 너무 짧거나 길면 제외
    if len(segment.strip()) < 1 or len(segment) > 20:
        return False
    
    # 의미 없는 숫자 나열 제외
    if re.match(r'^\d+$', segment.strip()):
        return False
    
    # 한글 제외
    if contains_hangul(segment):
        return False
    
    return True


def collect_all_segments() -> Tuple[List[str], List[str]]:
    """
    워크스페이스에서 모든 세그먼트 수집
    반환: (normal_segments, symbol_segments)
    """
    print("=" * 70)
    print("코드 파일 스캔 및 단어 단위 분할 중...")
    print("=" * 70)
    
    code_files = find_code_files(WORKSPACE_ROOT)
    print(f"발견된 코드 파일: {len(code_files):,}개")
    
    normal_segments = []
    symbol_segments = []
    
    for file_path in tqdm(code_files, desc="파일 읽기"):
        lines = extract_code_lines(file_path)
        for line in lines:
            segments = split_into_words_and_phrases(line)
            for segment in segments:
                if not is_valid_segment(segment):
                    continue
                if has_symbol(segment):
                    symbol_segments.append(segment)
                else:
                    normal_segments.append(segment)
    
    print(f"\n추출된 세그먼트:")
    print(f"  일반: {len(normal_segments):,}개")
    print(f"  기호 포함: {len(symbol_segments):,}개")
    print(f"  총: {len(normal_segments) + len(symbol_segments):,}개")
    
    # 기호 집중 학습용 세그먼트 추가
    symbol_focused = generate_symbol_focused_segments()
    symbol_segments.extend(symbol_focused)
    print(f"  기호 집중 학습용: {len(symbol_focused):,}개 추가")
    
    return normal_segments, symbol_segments


def generate_dataset():
    """데이터셋 생성 (Train + Validation)"""
    print("\n" + "=" * 70)
    print("PaddleOCR REC 데이터셋 생성 (단어 단위 분할)")
    print(f"목표: Train {TARGET_TRAIN_COUNT:,}개 + Val {TARGET_VAL_COUNT:,}개 = 총 {TARGET_TRAIN_COUNT + TARGET_VAL_COUNT:,}개")
    print("=" * 70)
    
    # 디렉토리 생성
    REC_TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    REC_VAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 세그먼트 수집
    normal_segments, symbol_segments = collect_all_segments()
    
    if len(normal_segments) == 0 and len(symbol_segments) == 0:
        print("[ERROR] 추출된 세그먼트가 없습니다!")
        return
    
    total_count = TARGET_TRAIN_COUNT + TARGET_VAL_COUNT
    
    # 기호 집중 학습 비율 30% 이상
    min_symbol_count = int(total_count * 0.3)
    target_symbol_count = min(min_symbol_count, len(symbol_segments))
    target_normal_count = total_count - target_symbol_count
    
    # 세그먼트 준비
    all_segments = []
    
    # 기호 세그먼트 먼저 배치
    if len(symbol_segments) >= target_symbol_count:
        all_segments.extend(random.sample(symbol_segments, target_symbol_count))
    else:
        all_segments.extend(symbol_segments)
        # 부족하면 반복
        while len(all_segments) < target_symbol_count:
            all_segments.extend(random.sample(symbol_segments, min(len(symbol_segments), target_symbol_count - len(all_segments))))
    
    # 일반 세그먼트로 채우기
    if len(normal_segments) >= target_normal_count:
        all_segments.extend(random.sample(normal_segments, target_normal_count))
    else:
        all_segments.extend(normal_segments)
        # 부족하면 반복
        while len(all_segments) < total_count:
            all_segments.extend(random.sample(normal_segments, min(len(normal_segments), total_count - len(all_segments))))
    
    # 랜덤 셔플
    random.shuffle(all_segments)
    all_segments = all_segments[:total_count]
    
    print(f"\n[INFO] 최종 세그먼트 구성:")
    symbol_count = sum(1 for s in all_segments if has_symbol(s))
    print(f"  기호 포함: {symbol_count:,}개 ({symbol_count/len(all_segments)*100:.1f}%)")
    print(f"  일반: {len(all_segments) - symbol_count:,}개")
    
    # 폰트 로드
    font = load_font(16)
    if font is None:
        print("[ERROR] 폰트를 로드할 수 없습니다.")
        return
    
    # Train 데이터셋 생성
    print(f"\n[1/2] Train 이미지 생성 중... ({TARGET_TRAIN_COUNT:,}개)")
    train_labels = []
    train_segments = all_segments[:TARGET_TRAIN_COUNT]
    
    random.seed(42)
    
    for i in tqdm(range(TARGET_TRAIN_COUNT), desc="Train 이미지 생성"):
        segment = train_segments[i]
        
        # 테마 선택
        if random.random() < 0.8:
            theme = random.choice(DARK_THEMES)
        else:
            theme = random.choice(LIGHT_THEMES)
        
        # 파일 확장자 랜덤 선택 (문법 강조용)
        file_ext = random.choice(['.py', '.js', '.ts', '.html', '.css'])
        
        # 이미지 생성
        img = render_text_with_syntax(
            segment,
            file_ext,
            font,
            theme,
            target_height=48,
            max_width=640,
            padding=10
        )
        
        # 품질 향상
        img = enhance_image_quality(img)
        
        # 저장
        img_name = f"word_{i + 1:06d}.jpg"
        img_path = REC_TRAIN_IMAGES_DIR / img_name
        img.save(img_path, "JPEG", quality=95, optimize=True)
        
        # 라벨 추가
        rel_path = f"train/{img_name}"
        clean_label = segment.replace('\t', ' ').replace('\n', ' ').strip()
        train_labels.append(f"{rel_path}\t{clean_label}\n")
    
    # Validation 데이터셋 생성
    print(f"\n[2/2] Validation 이미지 생성 중... ({TARGET_VAL_COUNT:,}개)")
    val_labels = []
    val_segments = all_segments[TARGET_TRAIN_COUNT:TARGET_TRAIN_COUNT + TARGET_VAL_COUNT]
    
    random.seed(123)
    
    for i in tqdm(range(TARGET_VAL_COUNT), desc="Val 이미지 생성"):
        if i >= len(val_segments):
            break
        segment = val_segments[i]
        
        # 테마 선택
        if random.random() < 0.8:
            theme = random.choice(DARK_THEMES)
        else:
            theme = random.choice(LIGHT_THEMES)
        
        # 파일 확장자 랜덤 선택
        file_ext = random.choice(['.py', '.js', '.ts', '.html', '.css'])
        
        # 이미지 생성
        img = render_text_with_syntax(
            segment,
            file_ext,
            font,
            theme,
            target_height=48,
            max_width=640,
            padding=10
        )
        
        # 품질 향상
        img = enhance_image_quality(img)
        
        # 저장
        img_name = f"word_{i + 1:06d}.jpg"
        img_path = REC_VAL_IMAGES_DIR / img_name
        img.save(img_path, "JPEG", quality=95, optimize=True)
        
        # 라벨 추가
        rel_path = f"val/{img_name}"
        clean_label = segment.replace('\t', ' ').replace('\n', ' ').strip()
        val_labels.append(f"{rel_path}\t{clean_label}\n")
    
    # 라벨 파일 저장
    print("\n라벨 파일 저장 중...")
    with open(REC_GT_TRAIN_FILE, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(train_labels)
    
    with open(REC_GT_VAL_FILE, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(val_labels)
    
    print("\n" + "=" * 70)
    print("[OK] 데이터셋 생성 완료!")
    print("=" * 70)
    print(f"\n📊 생성 결과:")
    print(f"  Train 이미지: {len(train_labels):,}개")
    print(f"  Val 이미지: {len(val_labels):,}개")
    print(f"  총 이미지: {len(train_labels) + len(val_labels):,}개")
    print(f"\n📁 저장 위치:")
    print(f"  Train 이미지: {REC_TRAIN_IMAGES_DIR}")
    print(f"  Val 이미지: {REC_VAL_IMAGES_DIR}")
    print(f"  Train 라벨: {REC_GT_TRAIN_FILE}")
    print(f"  Val 라벨: {REC_GT_VAL_FILE}")
    print(f"\n✨ 적용된 기능:")
    print(f"  - 단어 단위 분할 (공백 기준)")
    print(f"  - 기호 집중 학습 (30% 이상)")
    print(f"  - 다양한 길이 조합 (1~3개 단어, 최대 20자)")
    print(f"  - 글자 찌그러짐 방지 (폰트 크기 자동 조정)")
    print(f"  - 영문 전용 (한글 제외)")
    print(f"  - 고품질 렌더링 (대비 1.2x, 선명도 1.3x)")
    print(f"  - 어두운 테마 80% / 밝은 테마 20%")
    print(f"  - JPEG 품질: 95% (GPU 최적화)")


if __name__ == "__main__":
    generate_dataset()
