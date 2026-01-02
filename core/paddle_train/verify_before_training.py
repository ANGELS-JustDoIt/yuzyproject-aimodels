"""
학습 시작 전 파일 경로 및 설정 검증 스크립트
RTX 4070 Ti 최적화 설정 확인
"""

import os
import sys
from pathlib import Path

# UTF-8 출력 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트 (yuzyproject-aimodels)
PROJECT_ROOT = Path(__file__).parent.parent.parent

print("=" * 70)
print("PaddleOCR Fine-tuning 실행 전 검증")
print("=" * 70)

# 1. Pretrained 모델 경로 확인
pretrained_model = PROJECT_ROOT / "output" / "ppocr_v5_training_20260102_025236" / "rec_model" / "best_accuracy.pdparams"
print(f"\n[1] Pretrained 모델 확인:")
print(f"    경로: {pretrained_model}")
if pretrained_model.exists():
    size_mb = pretrained_model.stat().st_size / (1024 * 1024)
    print(f"    [OK] 존재함 (크기: {size_mb:.2f} MB)")
else:
    print(f"    [ERROR] 파일이 존재하지 않습니다!")

# 2. 데이터 디렉토리 확인
data_dir = PROJECT_ROOT / "core" / "paddle_train" / "train_data" / "rec"
print(f"\n[2] 데이터 디렉토리 확인:")
print(f"    경로: {data_dir}")
if data_dir.exists():
    print(f"    [OK] 디렉토리 존재")
    
    # train 폴더 확인
    train_dir = data_dir / "train"
    if train_dir.exists():
        train_count = len(list(train_dir.glob("*.jpg")))
        print(f"    [OK] train 폴더: {train_count:,}개 이미지")
    else:
        print(f"    [ERROR] train 폴더가 없습니다!")
    
    # val 폴더 확인
    val_dir = data_dir / "val"
    if val_dir.exists():
        val_count = len(list(val_dir.glob("*.jpg")))
        print(f"    [OK] val 폴더: {val_count:,}개 이미지")
    else:
        print(f"    [ERROR] val 폴더가 없습니다!")
else:
    print(f"    [ERROR] 디렉토리가 존재하지 않습니다!")

# 3. 라벨 파일 확인
train_label = data_dir / "rec_gt_train.txt"
val_label = data_dir / "rec_gt_test.txt"

print(f"\n[3] 라벨 파일 확인:")
print(f"    Train 라벨: {train_label}")
if train_label.exists():
    train_lines = len(train_label.read_text(encoding='utf-8').strip().split('\n'))
    print(f"    [OK] 존재함 ({train_lines:,}개 라인)")
    
    # 첫 번째 라인 확인
    first_line = train_label.read_text(encoding='utf-8').split('\n')[0]
    print(f"    샘플: {first_line[:80]}...")
else:
    print(f"    [ERROR] 파일이 존재하지 않습니다!")

print(f"    Val 라벨: {val_label}")
if val_label.exists():
    val_lines = len(val_label.read_text(encoding='utf-8').strip().split('\n'))
    print(f"    [OK] 존재함 ({val_lines:,}개 라인)")
else:
    print(f"    [ERROR] 파일이 존재하지 않습니다!")

# 4. 첫 번째 학습 이미지 확인
print(f"\n[4] 첫 번째 학습 이미지 확인:")
first_image = data_dir / "train" / "word_000001.jpg"
print(f"    경로: {first_image}")
if first_image.exists():
    size_kb = first_image.stat().st_size / 1024
    print(f"    [OK] 존재함 (크기: {size_kb:.2f} KB)")
else:
    print(f"    [ERROR] 파일이 존재하지 않습니다!")

# 5. 설정 파일 확인
config_file = PROJECT_ROOT / "core" / "paddle_train" / "configs" / "rec_ke_v5_finetune_v1.yml"
print(f"\n[5] 설정 파일 확인:")
print(f"    경로: {config_file}")
if config_file.exists():
    print(f"    [OK] 존재함")
    
    # 설정 내용 일부 확인
    config_content = config_file.read_text(encoding='utf-8')
    if "use_amp: true" in config_content:
        print(f"    [OK] AMP 활성화 확인")
    else:
        print(f"    [WARN] AMP 설정이 없습니다")
    
    if "batch_size_per_card: 64" in config_content:
        print(f"    [OK] Batch size 64 확인")
    else:
        print(f"    [WARN] Batch size가 64가 아닙니다")
    
    if "num_workers: 0" in config_content:
        print(f"    [OK] num_workers: 0 확인 (Windows 최적화)")
    else:
        print(f"    [WARN] num_workers가 0이 아닙니다")
else:
    print(f"    [ERROR] 파일이 존재하지 않습니다!")

# 6. venv_ocr 확인
venv_python = PROJECT_ROOT / "venv_ocr" / "Scripts" / "python.exe"
print(f"\n[6] venv_ocr 확인:")
print(f"    경로: {venv_python}")
if venv_python.exists():
    print(f"    [OK] 존재함")
else:
    print(f"    [ERROR] venv_ocr이 없습니다!")

# 7. PaddleOCR 경로 확인
paddleocr_train = Path("C:/Pyg/Tools/PaddleOCR/tools/train.py")
print(f"\n[7] PaddleOCR 학습 스크립트 확인:")
print(f"    경로: {paddleocr_train}")
if paddleocr_train.exists():
    print(f"    [OK] 존재함")
else:
    print(f"    [ERROR] 파일이 존재하지 않습니다!")

print("\n" + "=" * 70)
print("검증 완료!")
print("=" * 70)

