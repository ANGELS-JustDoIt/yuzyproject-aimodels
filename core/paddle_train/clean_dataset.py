"""
데이터셋 검수 및 세척 스크립트
불량 이미지 파일을 찾아서 라벨 파일에서 제거합니다.
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm

# 설정
CORE_TRAIN_DIR = Path(__file__).parent
DATA_DIR = CORE_TRAIN_DIR / "train_data" / "rec"
LABEL_FILES = ["rec_gt_train.txt", "rec_gt_test.txt"]

print("=" * 70)
print("데이터셋 검수 및 세척 시작")
print("=" * 70)

for label_name in LABEL_FILES:
    label_path = DATA_DIR / label_name
    
    if not label_path.exists():
        print(f"\n[SKIP] {label_name} 파일이 존재하지 않습니다.")
        continue
    
    new_labels = []
    error_count = 0
    error_indices = []
    
    print(f"\n[{label_name}] 검사 중...")
    
    # 기존 라벨 파일 읽기
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"총 {len(lines)}개 라인 검사 시작...")
    
    for idx, line in enumerate(tqdm(lines, desc=f"검사 중")):
        line = line.strip()
        if not line:
            error_count += 1
            error_indices.append(idx)
            continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            error_count += 1
            error_indices.append(idx)
            continue
        
        img_relative_path = parts[0].strip()
        
        # 경로 정규화 (슬래시를 OS 경로로 변환)
        img_relative_path = img_relative_path.replace('/', os.sep)
        
        # 전체 경로 생성
        full_path = DATA_DIR / img_relative_path
        
        # 1. 파일 존재 여부 확인
        if not full_path.exists():
            error_count += 1
            error_indices.append(idx)
            print(f"\n[ERROR] 파일 없음 (idx {idx}): {full_path}")
            continue
        
        # 2. 파일 크기 확인 (0바이트 체크)
        if full_path.stat().st_size == 0:
            error_count += 1
            error_indices.append(idx)
            print(f"\n[ERROR] 0바이트 파일 (idx {idx}): {full_path}")
            continue
        
        # 3. 실제 이미지 로드 시험 (PaddleOCR과 동일한 방식: cv2.imread)
        try:
            img = cv2.imread(str(full_path))
            if img is None:
                error_count += 1
                error_indices.append(idx)
                print(f"\n[ERROR] 이미지 로드 실패 (idx {idx}): {full_path}")
                continue
            
            # 이미지 shape 확인 (비정상적인 이미지 체크)
            if len(img.shape) != 3 or img.shape[2] != 3:
                error_count += 1
                error_indices.append(idx)
                print(f"\n[ERROR] 비정상 이미지 shape (idx {idx}): {full_path}, shape: {img.shape}")
                continue
                
        except Exception as e:
            error_count += 1
            error_indices.append(idx)
            print(f"\n[ERROR] 예외 발생 (idx {idx}): {full_path}, {e}")
            continue
        
        # 정상적인 라인은 유지
        new_labels.append(line + '\n')
    
    # 기존 라벨 파일 백업
    backup_path = label_path.with_suffix(label_path.suffix + '.bak')
    if backup_path.exists():
        backup_path.unlink()  # 기존 백업 삭제
    
    label_path.rename(backup_path)
    print(f"\n[BACKUP] 기존 파일을 {backup_path.name}로 백업했습니다.")
    
    # 깨끗한 라벨 파일로 교체
    with open(label_path, "w", encoding="utf-8", newline='\n') as f:
        f.writelines(new_labels)
    
    print(f"\n[RESULT] {label_name}:")
    print(f"  원본: {len(lines)}개")
    print(f"  불량: {error_count}개")
    print(f"  정상: {len(new_labels)}개")
    if error_indices:
        print(f"  불량 인덱스 (처음 10개): {error_indices[:10]}")

print("\n" + "=" * 70)
print("[OK] 데이터셋 검수 및 세척 완료!")
print("=" * 70)

