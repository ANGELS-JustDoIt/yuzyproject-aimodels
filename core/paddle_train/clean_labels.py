"""
라벨 파일의 텍스트 정제 스크립트
특수 문자, HTML 태그, 제어 문자 등을 제거하여 PaddleOCR이 처리할 수 있도록 정제합니다.
"""

import re
from pathlib import Path

CORE_TRAIN_DIR = Path(__file__).parent
DATA_DIR = CORE_TRAIN_DIR / "train_data" / "rec"
LABEL_FILES = ["rec_gt_train.txt", "rec_gt_test.txt"]

def clean_text(text):
    """텍스트 정제: 제어 문자, HTML 태그, 특수 문자 제거"""
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 제어 문자 제거 (탭, 줄바꿈 제외)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

print("=" * 70)
print("라벨 파일 텍스트 정제 시작")
print("=" * 70)

for label_name in LABEL_FILES:
    label_path = DATA_DIR / label_name
    
    if not label_path.exists():
        print(f"\n[SKIP] {label_name} 파일이 존재하지 않습니다.")
        continue
    
    print(f"\n[{label_name}] 정제 중...")
    
    cleaned_lines = []
    error_count = 0
    
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            error_count += 1
            continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            error_count += 1
            continue
        
        img_path = parts[0].strip()
        label_text = '\t'.join(parts[1:])  # 나머지 부분을 모두 라벨로
        
        # 텍스트 정제
        cleaned_text = clean_text(label_text)
        
        # 정제 후 빈 텍스트는 제외
        if not cleaned_text:
            error_count += 1
            continue
        
        # 최대 길이 제한 (PaddleOCR 설정에 맞춤)
        if len(cleaned_text) > 200:
            cleaned_text = cleaned_text[:200]
        
        cleaned_lines.append(f"{img_path}\t{cleaned_text}\n")
    
    # 백업
    backup_path = label_path.with_suffix(label_path.suffix + '.clean_bak')
    if backup_path.exists():
        backup_path.unlink()
    
    label_path.rename(backup_path)
    print(f"[BACKUP] 기존 파일을 {backup_path.name}로 백업했습니다.")
    
    # 정제된 라벨 저장
    with open(label_path, "w", encoding="utf-8", newline='\n') as f:
        f.writelines(cleaned_lines)
    
    print(f"[RESULT] {label_name}:")
    print(f"  원본: {len(lines)}개")
    print(f"  제외: {error_count}개")
    print(f"  정상: {len(cleaned_lines)}개")

print("\n" + "=" * 70)
print("[OK] 라벨 파일 텍스트 정제 완료!")
print("=" * 70)

