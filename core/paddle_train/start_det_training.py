"""Detection 모델 학습 시작 스크립트"""
import os
import sys
import subprocess
import yaml
from pathlib import Path
from datetime import datetime

# UTF-8 출력 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 스크립트 파일의 절대 경로를 기준으로 경로 계산
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VENV_OCR = PROJECT_ROOT / "venv_ocr"
VENV_OCR_PYTHON = VENV_OCR / "Scripts" / "python.exe"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 현재 디렉토리
BASE_DIR = SCRIPT_DIR
CONFIG_DIR = BASE_DIR / "configs"
DET_CONFIG = CONFIG_DIR / "det_ke_v5_train.yml"

# 출력 디렉토리 (Recognition과 동일한 타임스탬프 사용)
TRAINING_DIRS = sorted([d for d in OUTPUT_DIR.glob("ppocr_v5_training_*") if d.is_dir()], reverse=True)
if TRAINING_DIRS:
    OUTPUT_BASE = TRAINING_DIRS[0]  # 최신 학습 디렉토리 사용
else:
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_BASE = OUTPUT_DIR / f"ppocr_v5_training_{TIMESTAMP}"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

(OUTPUT_BASE / "det_model").mkdir(exist_ok=True)
(OUTPUT_BASE / "det_inference").mkdir(exist_ok=True)

print("=" * 70)
print("PP-OCRv5 Detection 모델 학습 시작 (RTX 4070Ti 최적화, venv_ocr)")
print("=" * 70)
print(f"\n[스크립트 위치] {SCRIPT_DIR}")
print(f"[프로젝트 루트] {PROJECT_ROOT}")
print(f"[출력 디렉토리] {OUTPUT_BASE}")

# venv_ocr 확인
if not VENV_OCR_PYTHON.exists():
    print(f"\n[ERROR] venv_ocr 가상환경을 찾을 수 없습니다: {VENV_OCR_PYTHON}")
    print("  setup_venv_ocr.bat을 실행하여 가상환경을 설정하세요.")
    sys.exit(1)

print(f"\n[가상환경 확인]")
print(f"  venv_ocr: {VENV_OCR}")
print(f"  Python: {VENV_OCR_PYTHON}")

# Config 파일 확인
if not DET_CONFIG.exists():
    print(f"\n[ERROR] Detection Config 파일을 찾을 수 없습니다: {DET_CONFIG}")
    sys.exit(1)

print(f"\n[Config 파일]")
print(f"  Detection: {DET_CONFIG}")

# Config 파일 경로 업데이트 (출력 디렉토리)
print(f"\n[Config 파일 경로 업데이트]")
print(f"  출력 디렉토리: {OUTPUT_BASE}")

# Detection Config 업데이트
with open(DET_CONFIG, 'r', encoding='utf-8') as f:
    det_config = yaml.safe_load(f)

det_config['Global']['save_model_dir'] = str(OUTPUT_BASE / "det_model")
det_config['Global']['save_inference_dir'] = str(OUTPUT_BASE / "det_inference")

DET_CONFIG_TEMP = BASE_DIR / "configs" / f"det_ke_v5_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yml"
with open(DET_CONFIG_TEMP, 'w', encoding='utf-8') as f:
    yaml.dump(det_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
print(f"  Detection Config 임시 파일: {DET_CONFIG_TEMP}")

# 업데이트된 Config 파일 사용
DET_CONFIG = DET_CONFIG_TEMP

# GPU 확인
print(f"\n[GPU 확인]")
try:
    gpu_check = subprocess.run(
        [str(VENV_OCR_PYTHON), "-c", 
         "import paddle; paddle.device.set_device('gpu'); print('GPU 사용 가능:', paddle.device.get_device())"],
        capture_output=True,
        text=True,
        timeout=10
    )
    if gpu_check.returncode == 0:
        print(f"  {gpu_check.stdout.strip()}")
    else:
        print(f"  [경고] GPU 확인 실패: {gpu_check.stderr}")
except Exception as e:
    print(f"  [경고] GPU 확인 중 오류: {e}")

# PaddleOCR 학습 스크립트 경로 확인
PADDLEOCR_ROOT = Path("C:/Pyg/Tools/PaddleOCR")
if PADDLEOCR_ROOT.exists() and (PADDLEOCR_ROOT / "tools" / "train.py").exists():
    TRAIN_SCRIPT = PADDLEOCR_ROOT / "tools" / "train.py"
    TRAIN_WORK_DIR = PADDLEOCR_ROOT
    print(f"\n[PaddleOCR 소스] {PADDLEOCR_ROOT}")
    print(f"  학습 스크립트: {TRAIN_SCRIPT}")
else:
    TRAIN_SCRIPT = None
    TRAIN_WORK_DIR = BASE_DIR
    print(f"\n[경고] PaddleOCR 소스를 찾을 수 없습니다. 기본 경로 사용: {BASE_DIR}")

# 학습 시작
print("\n" + "=" * 70)
print("Detection 모델 학습 시작")
print("=" * 70)
print(f"  Config: {DET_CONFIG}")
print(f"  Python: {VENV_OCR_PYTHON}")
print(f"  작업 디렉토리: {TRAIN_WORK_DIR}")
if TRAIN_SCRIPT:
    print(f"  학습 스크립트: {TRAIN_SCRIPT}")
    print(f"  명령어: python tools/train.py -c {DET_CONFIG}")
else:
    print(f"  명령어: python -m paddleocr.tools.train -c {DET_CONFIG}")
print("\n")

# 작업 디렉토리 설정
os.chdir(TRAIN_WORK_DIR)

# Detection 학습 실행
if TRAIN_SCRIPT:
    det_result = subprocess.run([
        str(VENV_OCR_PYTHON),
        str(TRAIN_SCRIPT),
        "-c",
        str(DET_CONFIG)
    ], cwd=str(TRAIN_WORK_DIR))
else:
    det_result = subprocess.run([
        str(VENV_OCR_PYTHON),
        "-m", "paddleocr.tools.train",
        "-c",
        str(DET_CONFIG)
    ], cwd=str(TRAIN_WORK_DIR))

if det_result.returncode != 0:
    print("\n[ERROR] Detection 학습 실패!")
    print(f"  종료 코드: {det_result.returncode}")
    sys.exit(det_result.returncode)

print("\n" + "=" * 70)
print("Detection 학습 완료!")
print("=" * 70)
print(f"\n[출력 디렉토리] {OUTPUT_BASE}")
print(f"  - Detection 모델: {OUTPUT_BASE / 'det_model'}")
print(f"  - Detection 추론: {OUTPUT_BASE / 'det_inference'}")

# 임시 Config 파일 정리
try:
    if DET_CONFIG_TEMP.exists():
        DET_CONFIG_TEMP.unlink()
        print(f"\n[정리] 임시 Config 파일 삭제 완료")
except Exception as e:
    print(f"\n[경고] 임시 Config 파일 삭제 실패: {e}")

