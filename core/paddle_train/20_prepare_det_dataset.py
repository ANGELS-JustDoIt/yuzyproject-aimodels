"""
C:\Pyg\OCR_Data에서 Detection 데이터셋 준비
- JSON과 이미지 매칭 확인
- 한글+영어 텍스트 포함
- 이미지 리사이즈 (깨지지 않게)
- train_data/det에 저장
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random

OCR_DATA_ROOT = Path("C:/Pyg/OCR_Data")
BASE_DIR = Path(__file__).parent.parent.parent
CORE_TRAIN_DIR = BASE_DIR / "core" / "paddle_train"
TRAIN_DATA_DIR = CORE_TRAIN_DIR / "train_data"
NEW_DET_DIR = TRAIN_DATA_DIR / "det"
NEW_DET_TRAIN_DIR = NEW_DET_DIR / "train"
NEW_DET_TEST_DIR = NEW_DET_DIR / "test"
NEW_DET_TRAIN_LABEL = NEW_DET_DIR / "train_label.txt"
NEW_DET_TEST_LABEL = NEW_DET_DIR / "test_label.txt"

# Detection 데이터셋 최대 크기 (메모리 절약)
MAX_DET_TRAIN = 25000  # Recognition과 비슷한 수준
MAX_DET_TEST = 5000

# 이미지 리사이즈 설정 (PaddleOCR Detection 권장: 최대 크기 제한)
MAX_DET_IMAGE_SIZE = 3200  # 최대 너비/높이

def resize_image_preserve_aspect(img_path, max_size=MAX_DET_IMAGE_SIZE):
    """이미지를 비율 유지하며 리사이즈"""
    try:
        img = Image.open(img_path)
        original_width, original_height = img.size
        
        # 이미 작으면 그대로
        if original_width <= max_size and original_height <= max_size:
            return img
        
        # 비율 계산
        ratio = min(max_size / original_width, max_size / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # 리사이즈 (고품질 리샘플링)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img_resized
    except Exception as e:
        print(f"[ERROR] 이미지 리사이즈 실패 {img_path}: {e}")
        return None

def convert_polygon_to_det_format(polygon_points, img_width, img_height, new_width, new_height):
    """polygon 좌표를 Detection 형식으로 변환 (절대 좌표)"""
    # 좌표 스케일 조정 (리사이즈된 이미지에 맞게)
    scale_x = new_width / img_width
    scale_y = new_height / img_height
    
    # polygon을 절대 좌표로 변환 (PaddleOCR Detection은 절대 좌표 사용)
    scaled_points = []
    for point in polygon_points:
        if len(point) >= 2:
            x = int(point[0] * scale_x)
            y = int(point[1] * scale_y)
            scaled_points.append(f"{x},{y}")
    
    # Detection 형식: "x1,y1 x2,y2 x3,y3 x4,y4"
    if len(scaled_points) >= 4:
        # 4개 점만 사용
        det_str = " ".join(scaled_points[:4])
        return det_str
    return None

def parse_json_annotation(json_path):
    """JSON 파일 파싱하여 이미지 경로와 Annotation 추출"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        images_info = data.get("Images", {})
        annotations = data.get("Annotation", [])
        
        # 이미지 파일명
        image_filename = images_info.get("file_name", "")
        original_width = images_info.get("width", 0)
        original_height = images_info.get("height", 0)
        
        return {
            "image_filename": image_filename,
            "original_width": original_width,
            "original_height": original_height,
            "annotations": annotations
        }
    except Exception as e:
        print(f"[ERROR] JSON 파싱 실패 {json_path}: {e}")
        return None

def find_image_file(json_path_obj, image_filename):
    """이미지 파일 찾기 (확장자 무관, 대소문자 무관)"""
    json_dir_path = Path(json_path_obj).parent if isinstance(json_path_obj, Path) else Path(json_path_obj).parent
    json_path_full = Path(json_path_obj) if isinstance(json_path_obj, (str, Path)) else json_path_obj
    
    # 파일명에서 확장자 제거 (stem)
    base_name = Path(image_filename).stem
    
    # TS_OCR_KE_PB 또는 VS_OCR_KE_PB 디렉토리에서 찾기
    if "TL_OCR_KE_PB" in str(json_path_full):
        ts_dir = json_path_full.parent.parent / "TS_OCR_KE_PB"
        # 원본 파일명 그대로 시도
        if (ts_dir / image_filename).exists():
            return ts_dir / image_filename
        # 대소문자 무관 확장자 시도
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            img_path = ts_dir / f"{base_name}{ext}"
            if img_path.exists():
                return img_path
    
    # VL_OCR_KE_PB의 경우 VS_OCR_KE_PB에서 찾기
    if "VL_OCR_KE_PB" in str(json_path_full):
        vs_dir = json_path_full.parent.parent / "VS_OCR_KE_PB"
        # 원본 파일명 그대로 시도
        if (vs_dir / image_filename).exists():
            return vs_dir / image_filename
        # 대소문자 무관 확장자 시도
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            img_path = vs_dir / f"{base_name}{ext}"
            if img_path.exists():
                return img_path
    
    # 같은 디렉토리에서도 시도
    if (json_dir_path / image_filename).exists():
        return json_dir_path / image_filename
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        img_path = json_dir_path / f"{base_name}{ext}"
        if img_path.exists():
            return img_path
    
    return None

def prepare_det_dataset():
    """Detection 데이터셋 준비"""
    print("=" * 70)
    print("C:\\Pyg\\OCR_Data -> train_data/det 변환")
    print("=" * 70)
    
    # 디렉토리 생성
    NEW_DET_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    NEW_DET_TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Training 및 Validation 디렉토리
    train_json_dir = OCR_DATA_ROOT / "Training" / "TL_OCR_KE_PB"
    val_json_dir = OCR_DATA_ROOT / "Validation" / "VL_OCR_KE_PB"
    
    train_data = []
    val_data = []
    
    # Training 데이터 수집
    if train_json_dir.exists():
        print(f"\n[1/4] Training JSON 파일 수집 중...")
        json_files = list(train_json_dir.glob("*.json"))
        print(f"  JSON 파일: {len(json_files):,}개")
        
        for json_path in tqdm(json_files[:MAX_DET_TRAIN * 2], desc="Training JSON 파싱"):
            json_data = parse_json_annotation(json_path)
            if json_data:
                img_path = find_image_file(json_path, json_data["image_filename"])
                if img_path and img_path.exists():
                    train_data.append({
                        "json_path": json_path,
                        "image_path": img_path,
                        "json_data": json_data
                    })
        
        print(f"  매칭된 데이터: {len(train_data):,}개")
        
        # 랜덤 샘플링 (MAX_DET_TRAIN 개만)
        if len(train_data) > MAX_DET_TRAIN:
            train_data = random.sample(train_data, MAX_DET_TRAIN)
            print(f"  샘플링 후: {len(train_data):,}개")
    
    # Validation 데이터 수집
    if val_json_dir.exists():
        print(f"\n[2/4] Validation JSON 파일 수집 중...")
        json_files = list(val_json_dir.glob("*.json"))
        print(f"  JSON 파일: {len(json_files):,}개")
        
        for json_path in tqdm(json_files[:MAX_DET_TEST * 2], desc="Validation JSON 파싱"):
            json_data = parse_json_annotation(json_path)
            if json_data:
                img_path = find_image_file(json_path, json_data["image_filename"])
                if img_path and img_path.exists():
                    val_data.append({
                        "json_path": json_path,
                        "image_path": img_path,
                        "json_data": json_data
                    })
        
        print(f"  매칭된 데이터: {len(val_data):,}개")
        
        # 랜덤 샘플링 (MAX_DET_TEST 개만)
        if len(val_data) > MAX_DET_TEST:
            val_data = random.sample(val_data, MAX_DET_TEST)
            print(f"  샘플링 후: {len(val_data):,}개")
    
    # Train 데이터셋 생성
    print(f"\n[3/4] Train 데이터셋 생성 중...")
    train_labels = []
    train_copied = 0
    train_skipped = 0
    
    for idx, item in enumerate(tqdm(train_data, desc="Train 이미지 처리")):
        json_data = item["json_data"]
        img_path = item["image_path"]
        original_width = json_data["original_width"]
        original_height = json_data["original_height"]
        
        # 이미지 리사이즈
        img_resized = resize_image_preserve_aspect(img_path, MAX_DET_IMAGE_SIZE)
        if img_resized is None:
            train_skipped += 1
            continue
        
        new_width, new_height = img_resized.size
        
        # 새 이미지 이름
        img_ext = img_path.suffix
        new_img_name = f"img_{idx + 1:06d}{img_ext}"
        new_img_path = NEW_DET_TRAIN_DIR / new_img_name
        
        # 이미지 저장
        try:
            img_resized.save(new_img_path, quality=95)
        except Exception as e:
            print(f"[ERROR] 이미지 저장 실패 {new_img_path}: {e}")
            train_skipped += 1
            continue
        
        # Annotation을 Detection 형식으로 변환 (JSON 형식)
        formatted_annotations = []
        for ann in json_data["annotations"]:
            polygon_points = ann.get("polygon_points", [])
            text = ann.get("text", "")
            
            if len(polygon_points) >= 4 and text:
                # 좌표 스케일 조정
                scale_x = new_width / original_width
                scale_y = new_height / original_height
                
                # 스케일된 points 생성
                scaled_points = []
                for point in polygon_points[:4]:  # 4개 점만 사용
                    if len(point) >= 2:
                        x = int(point[0] * scale_x)
                        y = int(point[1] * scale_y)
                        scaled_points.append([x, y])
                
                if len(scaled_points) == 4:
                    formatted_annotations.append({
                        "transcription": text,
                        "points": scaled_points,
                        "difficult": False
                    })
        
        if formatted_annotations:
            # JSON 형식으로 저장
            rel_img_path = f"train/{new_img_name}"
            label_json = json.dumps(formatted_annotations, ensure_ascii=False)
            label_line = f"{rel_img_path}\t{label_json}\n"
            train_labels.append(label_line)
            train_copied += 1
    
    # Test 데이터셋 생성
    print(f"\n[4/4] Test 데이터셋 생성 중...")
    test_labels = []
    test_copied = 0
    test_skipped = 0
    
    for idx, item in enumerate(tqdm(val_data, desc="Test 이미지 처리")):
        json_data = item["json_data"]
        img_path = item["image_path"]
        original_width = json_data["original_width"]
        original_height = json_data["original_height"]
        
        # 이미지 리사이즈
        img_resized = resize_image_preserve_aspect(img_path, MAX_DET_IMAGE_SIZE)
        if img_resized is None:
            test_skipped += 1
            continue
        
        new_width, new_height = img_resized.size
        
        # 새 이미지 이름
        img_ext = img_path.suffix
        new_img_name = f"img_{idx + 1:06d}{img_ext}"
        new_img_path = NEW_DET_TEST_DIR / new_img_name
        
        # 이미지 저장
        try:
            img_resized.save(new_img_path, quality=95)
        except Exception as e:
            print(f"[ERROR] 이미지 저장 실패 {new_img_path}: {e}")
            test_skipped += 1
            continue
        
        # Annotation을 Detection 형식으로 변환 (JSON 형식)
        formatted_annotations = []
        for ann in json_data["annotations"]:
            polygon_points = ann.get("polygon_points", [])
            text = ann.get("text", "")
            
            if len(polygon_points) >= 4 and text:
                # 좌표 스케일 조정
                scale_x = new_width / original_width
                scale_y = new_height / original_height
                
                # 스케일된 points 생성
                scaled_points = []
                for point in polygon_points[:4]:  # 4개 점만 사용
                    if len(point) >= 2:
                        x = int(point[0] * scale_x)
                        y = int(point[1] * scale_y)
                        scaled_points.append([x, y])
                
                if len(scaled_points) == 4:
                    formatted_annotations.append({
                        "transcription": text,
                        "points": scaled_points,
                        "difficult": False
                    })
        
        if formatted_annotations:
            # JSON 형식으로 저장
            rel_img_path = f"test/{new_img_name}"
            label_json = json.dumps(formatted_annotations, ensure_ascii=False)
            label_line = f"{rel_img_path}\t{label_json}\n"
            test_labels.append(label_line)
            test_copied += 1
    
    # 라벨 파일 저장
    print("\n라벨 파일 저장 중...")
    with open(NEW_DET_TRAIN_LABEL, 'w', encoding='utf-8') as f:
        f.writelines(train_labels)
    
    with open(NEW_DET_TEST_LABEL, 'w', encoding='utf-8') as f:
        f.writelines(test_labels)
    
    print(f"\n[OK] Detection 데이터셋 준비 완료!")
    print(f"  Train: {train_copied:,}개 이미지, {len(train_labels):,}개 라벨 (건너뜀: {train_skipped:,})")
    print(f"  Test: {test_copied:,}개 이미지, {len(test_labels):,}개 라벨 (건너뜀: {test_skipped:,})")
    print(f"\n저장 위치: {NEW_DET_DIR}")

if __name__ == "__main__":
    prepare_det_dataset()

