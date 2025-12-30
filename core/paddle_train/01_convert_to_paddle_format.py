"""
AIHub JSON 형식을 PaddleOCR 학습 포맷으로 변환
- Detection: ICP 형식 라벨 파일 생성
- Recognition: Crop 이미지 + 라벨 파일 생성
"""

import os
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import cv2
try:
    from shapely.geometry import Polygon as ShapelyPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


def safe_imread(image_path: Path) -> Optional[np.ndarray]:
    """Windows 한글 경로를 안전하게 이미지 읽기"""
    try:
        img_array = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None


def safe_imsave(image_path: Path, img: np.ndarray) -> bool:
    """Windows 한글 경로를 안전하게 이미지 저장"""
    try:
        ext = image_path.suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            success, encoded_img = cv2.imencode(ext, img, encode_param)
        elif ext == '.png':
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
            success, encoded_img = cv2.imencode(ext, img, encode_param)
        else:
            return False
        
        if success:
            with open(image_path, 'wb') as f:
                f.write(encoded_img.tobytes())
            return True
        return False
    except Exception as e:
        return False


def normalize_json_root(root):
    """
    JSON root가 list인 경우 dict로 변환
    """
    if isinstance(root, list):
        if len(root) == 0:
            return None
        return root[0]  # 첫 번째 항목 사용
    elif isinstance(root, dict):
        return root
    else:
        return None


def parse_polygon_points(points) -> Optional[List[List[int]]]:
    """
    polygon_points를 표준화하여 [[x,y],...] 형태로 변환
    입력: [[x,y],...] 또는 [{"x":..,"y":..},...]
    출력: [[x,y],...] (4개 또는 그 외)
    """
    if not points or len(points) < 3:
        return None
    
    try:
        # 첫 번째 원소로 형태 판단
        first = points[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            # [[x,y],...] 형태
            pts = [[int(p[0]), int(p[1])] for p in points if len(p) >= 2]
        elif isinstance(first, dict) and "x" in first and "y" in first:
            # [{"x":..,"y":..},...] 형태
            pts = [[int(p.get("x", 0)), int(p.get("y", 0))] for p in points]
        else:
            return None
        
        if len(pts) < 3:
            return None
        
        return pts
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return None


def normalize_polygon_points(
    points, 
    allow_fallback_bbox: bool = False,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
    min_area: float = 1.0,
    fix_self_intersection: bool = True
) -> Optional[List[List[int]]]:
    """
    4점 폴리곤을 정규화하고 검증
    
    Args:
        points: [[x,y],...] 또는 [{"x":..,"y":..},...]
        allow_fallback_bbox: True이면 4개가 아니어도 bbox로 변환
        img_width: 이미지 너비 (좌표 클램핑용)
        img_height: 이미지 높이 (좌표 클램핑용)
        min_area: 최소 면적 (이보다 작으면 None 반환)
        fix_self_intersection: True이면 self-intersecting 폴리곤 수정 시도
    
    Returns:
        정규화된 4점 폴리곤 [[x,y],...] 또는 None (invalid인 경우)
    """
    # 먼저 표준화
    pts = parse_polygon_points(points)
    if pts is None:
        return None
    
    # 중복 점 제거 (거리 기준)
    unique_pts = []
    for pt in pts:
        is_duplicate = False
        for existing in unique_pts:
            dist = np.sqrt((pt[0] - existing[0])**2 + (pt[1] - existing[1])**2)
            if dist < 1.0:  # 1픽셀 이내면 중복으로 간주
                is_duplicate = True
                break
        if not is_duplicate:
            unique_pts.append(pt)
    
    pts = unique_pts
    
    if len(pts) < 3:
        return None
    
    # 4개가 아니면 fallback 처리
    if len(pts) != 4:
        if allow_fallback_bbox and len(pts) >= 3:
            # bbox로 변환 (min/max)
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            pts = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        else:
            return None
    
    # 좌표 클램핑 (이미지 크기 정보가 있을 경우)
    if img_width is not None and img_height is not None:
        for i in range(len(pts)):
            pts[i][0] = max(0, min(int(pts[i][0]), img_width - 1))
            pts[i][1] = max(0, min(int(pts[i][1]), img_height - 1))
    
    # 좌상->우상->우하->좌하 순서로 정렬 (더 안정적)
    # 먼저 min(x+y)를 찾아서 좌상단으로
    sum_coords = [(p[0] + p[1], i) for i, p in enumerate(pts)]
    top_left_idx = min(sum_coords)[1]
    
    # 좌상단 점을 기준으로 각도 정렬
    top_left = pts[top_left_idx]
    other_pts = [pts[i] for i in range(len(pts)) if i != top_left_idx]
    
    def get_angle(p):
        dx = p[0] - top_left[0]
        dy = p[1] - top_left[1]
        if dx == 0 and dy == 0:
            return 0.0
        angle = np.arctan2(dy, dx)
        # -90도부터 시작 (위쪽이 0도)
        return (angle + np.pi/2 + 2*np.pi) % (2*np.pi)
    
    sorted_other = sorted(other_pts, key=get_angle)
    sorted_pts = [top_left] + sorted_other
    
    # 면적 검사
    try:
        if SHAPELY_AVAILABLE:
            poly = ShapelyPolygon(sorted_pts)
            
            # Self-intersection 수정 시도
            if fix_self_intersection and not poly.is_valid:
                # Buffer(0)로 self-intersection 수정
                fixed_poly = poly.buffer(0)
                if fixed_poly.is_valid and hasattr(fixed_poly, 'exterior'):
                    coords = list(fixed_poly.exterior.coords[:-1])  # 마지막 점 제거 (중복)
                    if len(coords) >= 4:
                        # 4점으로 다시 변환 (convex hull 사용)
                        convex = fixed_poly.convex_hull
                        if convex.is_valid and hasattr(convex, 'exterior'):
                            coords = list(convex.exterior.coords[:-1])
                            if len(coords) == 4:
                                sorted_pts = [[int(c[0]), int(c[1])] for c in coords]
                                poly = ShapelyPolygon(sorted_pts)
            
            # 면적 검사
            area = poly.area
            if area < min_area:
                return None
            
            # 최종 검증
            if not poly.is_valid:
                return None
                
        else:
            # Shapely 없을 때 간단한 면적 계산 (Shoelace formula)
            area = 0.0
            for i in range(4):
                j = (i + 1) % 4
                area += sorted_pts[i][0] * sorted_pts[j][1]
                area -= sorted_pts[j][0] * sorted_pts[i][1]
            area = abs(area) / 2.0
            if area < min_area:
                return None
            
    except Exception:
        return None
    
    return [[int(p[0]), int(p[1])] for p in sorted_pts]


def parse_aihub_json(obj, json_path: Optional[Path] = None) -> Tuple[Optional[str], Optional[int], Optional[int], List[Dict]]:
    """
    멀티 포맷 JSON 객체를 파싱하여 표준 형식으로 변환
    
    지원 포맷:
    - 포맷 A: AIHub 원본 구조 (Images.file_name + Annotation[*].text/polygon_points)
    - 포맷 B: root가 list이며 각 원소가 {"transcription","points"} 형태
    - 포맷 C: dict이지만 Annotation 원소 키가 {"transcription","points"}인 경우
    
    Args:
        obj: JSON 로드된 객체 (dict 또는 list)
        json_path: JSON 파일 경로 (포맷 B에서 image filename 추론용)
    
    Returns:
        (image_rel_path, width, height, anns)
        - image_rel_path: 이미지 파일명 (상대 경로 또는 basename)
        - width: 이미지 너비 (None이면 모름)
        - height: 이미지 높이 (None이면 모름)
        - anns: annotation 리스트, 각 항목은 {"text": str, "polygon": [[x,y],...]}
    """
    image_rel_path = None
    width = None
    height = None
    anns = []
    
    # 포맷 B: root가 list인 경우 (각 원소가 {"transcription","points"})
    if isinstance(obj, list):
        if len(obj) == 0:
            return None, None, None, []
        
        # 첫 번째 원소로 포맷 판단
        first = obj[0]
        if isinstance(first, dict) and "transcription" in first and "points" in first:
            # 포맷 B: [{"transcription":..., "points":...}, ...]
            for ann in obj:
                if not isinstance(ann, dict):
                    continue
                
                text = ann.get("transcription", "") or ""
                points_raw = ann.get("points", [])
                
                polygon = parse_polygon_points(points_raw)
                if polygon is None:
                    continue
                
                anns.append({
                    "text": text,
                    "polygon": polygon
                })
            
            # image filename은 json_path의 stem에서 추론
            if json_path is not None:
                image_rel_path = json_path.stem
            else:
                image_rel_path = None
            
            return image_rel_path, width, height, anns
    
    # 포맷 A 또는 C: dict인 경우
    if not isinstance(obj, dict):
        return None, None, None, []
    
    # 포맷 A: AIHub 원본 구조 확인
    if "Images" in obj and "Annotation" in obj:
        # 포맷 A: AIHub 원본 구조
        images_info = obj.get("Images", {})
        if isinstance(images_info, dict):
            image_rel_path = images_info.get("file_name", "")
            width = images_info.get("width")
            height = images_info.get("height")
        
        annotations_raw = obj.get("Annotation", [])
        if not isinstance(annotations_raw, list):
            annotations_raw = []
        
        for ann in annotations_raw:
            if not isinstance(ann, dict):
                continue
            
            text = ann.get("text", "") or ""
            polygon_raw = ann.get("polygon_points", [])
            
            polygon = parse_polygon_points(polygon_raw)
            if polygon is None:
                continue
            
            anns.append({
                "text": text,
                "polygon": polygon
            })
        
        if not image_rel_path:
            # Images.file_name이 없으면 json_path에서 추론
            if json_path is not None:
                image_rel_path = json_path.stem
        
        return image_rel_path, width, height, anns
    
    # 포맷 C: dict이지만 다른 키 구조 (직접 {"transcription","points"} 포함)
    # 또는 Annotation이 직접 list인 경우
    if "Annotation" in obj:
        annotations_raw = obj.get("Annotation", [])
        if isinstance(annotations_raw, list) and len(annotations_raw) > 0:
            first_ann = annotations_raw[0]
            if isinstance(first_ann, dict) and "transcription" in first_ann and "points" in first_ann:
                # 포맷 C 변형: Annotation이 [{"transcription","points"},...]
                for ann in annotations_raw:
                    if not isinstance(ann, dict):
                        continue
                    
                    text = ann.get("transcription", "") or ""
                    points_raw = ann.get("points", [])
                    
                    polygon = parse_polygon_points(points_raw)
                    if polygon is None:
                        continue
                    
                    anns.append({
                        "text": text,
                        "polygon": polygon
                    })
                
                # image filename 추론
                if "Images" in obj and isinstance(obj["Images"], dict):
                    image_rel_path = obj["Images"].get("file_name", "")
                if not image_rel_path and json_path is not None:
                    image_rel_path = json_path.stem
                
                return image_rel_path, width, height, anns
    
    # 그 외: obj 자체가 annotation list처럼 보이는 경우 (단일 dict)
    if "transcription" in obj and "points" in obj:
        # 단일 annotation dict
        text = obj.get("transcription", "") or ""
        points_raw = obj.get("points", [])
        
        polygon = parse_polygon_points(points_raw)
        if polygon is not None:
            anns.append({
                "text": text,
                "polygon": polygon
            })
        
        if json_path is not None:
            image_rel_path = json_path.stem
        
        return image_rel_path, width, height, anns
    
    return None, None, None, []


def clean_text(text: str, skip_noise: bool = True) -> Optional[str]:
    """텍스트 클린징"""
    if text is None:
        return None
    
    text = text.strip()
    
    # 빈 문자열 제거
    if not text:
        return None
    
    # "###" 포함 제거 (일반적으로 학습 방해 마커)
    if "###" in text:
        return None
    
    # 너무 짧은 잡음 제거 옵션
    if skip_noise:
        if len(text) == 1 and not text.isalnum():
            return None
    
    return text


def find_ke_folders(root: Path, only_categories: Optional[List[str]] = None) -> Tuple[List[Path], List[Path]]:
    """'_KE_'가 포함된 폴더 스캔 (Training: 01_data/02_data, Validation: 03_data/04_data 지원)"""
    image_folders = []
    label_folders = []
    
    if not root.exists():
        return image_folders, label_folders
    
    for item in root.rglob("*"):
        if not item.is_dir():
            continue
        
        path_str = str(item)
        if "_KE_" not in path_str:
            continue
        
        # 카테고리 필터링
        if only_categories:
            category_match = any(cat in path_str for cat in only_categories)
            if not category_match:
                continue
        
        # Training 폴더: 01_data(이미지), 02_data(라벨)
        # Validation 폴더: 03_data(이미지), 04_data(라벨)
        # 한글 폴더명도 지원: 01.원천데이터, 02.라벨링데이터
        if "01_data" in path_str or "01.원천데이터" in path_str or "03_data" in path_str:
            image_folders.append(item)
        elif "02_data" in path_str or "02.라벨링데이터" in path_str or "04_data" in path_str:
            label_folders.append(item)
    
    return sorted(set(image_folders)), sorted(set(label_folders))


def build_image_index(image_folders: List[Path], cache_file: Optional[Path] = None) -> Dict[str, Path]:
    """
    이미지 파일 인덱스 구축 (basename -> fullpath)
    캐시 파일이 있으면 로드, 없으면 스캔 후 저장
    """
    # 캐시 로드 시도
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                index = pickle.load(f)
            print(f"[OK] 이미지 인덱스 캐시 로드: {len(index)}개 파일")
            return index
        except Exception as e:
            print(f"[WARN] 캐시 로드 실패, 재구축: {e}")
    
    # 인덱스 구축
    print(f"[INFO] 이미지 인덱스 구축 중... (폴더 {len(image_folders)}개)")
    if len(image_folders) == 0:
        print("[WARN] 이미지 폴더가 없습니다!")
        return {}
    
    # 디버그: 폴더 목록 출력
    for i, folder in enumerate(image_folders[:5]):
        print(f"[DEBUG] 이미지 폴더 {i+1}: {folder}")
    
    index = {}
    image_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    file_count_per_folder = {}
    
    for img_folder in image_folders:
        folder_count = 0
        for img_path in img_folder.rglob("*"):
            if img_path.is_file() and img_path.suffix in image_exts:
                basename = img_path.name.lower()
                # 같은 basename이 여러 개 있을 수 있으므로 첫 번째 것만 사용
                if basename not in index:
                    index[basename] = img_path
                    folder_count += 1
        file_count_per_folder[str(img_folder)] = folder_count
    
    print(f"[OK] 이미지 인덱스 구축 완료: {len(index)}개 파일")
    if len(index) == 0:
        print("[WARN] 이미지 파일이 하나도 발견되지 않았습니다!")
        print("[DEBUG] 폴더별 파일 개수:")
        for folder, count in list(file_count_per_folder.items())[:5]:
            print(f"  {folder}: {count}개")
    
    # 캐시 저장
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(index, f)
            print(f"[OK] 인덱스 캐시 저장: {cache_file}")
        except Exception as e:
            print(f"[WARN] 캐시 저장 실패: {e}")
    
    return index


def find_image_file(image_filename: str, image_index: Dict[str, Path], debug: bool = False) -> Optional[Path]:
    """
    이미지 파일 찾기 (인덱스 사용)
    
    Args:
        image_filename: 찾을 이미지 파일명 (예: "OCR_KE_C2_000006.jpeg")
        image_index: 이미지 인덱스 (basename_lower -> Path)
        debug: 디버그 로그 출력 여부
    
    Returns:
        이미지 파일 Path 또는 None
    """
    if not image_filename:
        return None
    
    # 정확한 매칭 (소문자 변환)
    basename = image_filename.lower().strip()
    if basename in image_index:
        if debug:
            print(f"[DEBUG] find_image_file: 정확 매칭 성공 '{image_filename}' -> '{basename}'")
        return image_index[basename]
    
    # 확장자 변형 시도
    base_name = image_filename.rsplit('.', 1)[0] if '.' in image_filename else image_filename
    base_name = base_name.lower().strip()
    
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        candidate = base_name + ext.lower()
        if candidate in image_index:
            if debug:
                print(f"[DEBUG] find_image_file: 확장자 변형 매칭 성공 '{image_filename}' -> '{candidate}'")
            return image_index[candidate]
    
    # 디버그: 유사한 파일명 찾기
    if debug:
        similar = [k for k in image_index.keys() if base_name[:10] in k.lower()][:3]
        print(f"[DEBUG] find_image_file: '{image_filename}' (basename='{basename}') 찾기 실패. 유사 파일: {similar}")
    
    return None


def use_existing_det_label(existing_label_path: Path, output_dir: Path, split: str) -> bool:
    """
    기존 PaddleOCR det 라벨 파일을 그대로 사용 (복사)
    
    Args:
        existing_label_path: 기존 라벨 파일 경로
        output_dir: 출력 디렉토리
        split: "train" or "val"
    
    Returns:
        성공 여부
    """
    import shutil
    
    if not existing_label_path.exists():
        print(f"[ERROR] 기존 라벨 파일이 없습니다: {existing_label_path}")
        return False
    
    det_dir = output_dir / "det"
    det_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = det_dir / f"{split}_label.txt"
    
    try:
        shutil.copy2(existing_label_path, target_path)
        print(f"[OK] 기존 라벨 파일 복사 완료: {existing_label_path} -> {target_path}")
        return True
    except Exception as e:
        print(f"[ERROR] 라벨 파일 복사 실패: {e}")
        return False


def validate_det_label(label_file: Path, num_samples: int = 3) -> bool:
    """
    Detection 라벨 파일 검증
    
    Args:
        label_file: 검증할 라벨 파일 경로
        num_samples: 출력할 샘플 개수
    
    Returns:
        검증 성공 여부
    """
    if not label_file.exists():
        print(f"[ERROR] 라벨 파일이 없습니다: {label_file}")
        return False
    
    print(f"\n=== 라벨 파일 검증: {label_file} ===")
    
    valid_count = 0
    invalid_count = 0
    
    with open(label_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t', 1)
            if len(parts) != 2:
                print(f"[WARN] 라인 {line_idx + 1}: 형식 오류 (TAB 구분자 없음)")
                invalid_count += 1
                continue
            
            image_path = parts[0]
            json_str = parts[1]
            
            try:
                annotations = json.loads(json_str)
                if not isinstance(annotations, list):
                    print(f"[WARN] 라인 {line_idx + 1}: annotations가 list가 아님")
                    invalid_count += 1
                    continue
                
                if len(annotations) == 0:
                    print(f"[WARN] 라인 {line_idx + 1}: annotations가 비어있음")
                    invalid_count += 1
                    continue
                
                # 각 annotation 검증
                line_valid = True
                for ann_idx, ann in enumerate(annotations):
                    if not isinstance(ann, dict):
                        print(f"[WARN] 라인 {line_idx + 1}, annotation {ann_idx}: dict가 아님")
                        line_valid = False
                        break
                    
                    transcription = ann.get("transcription", "")
                    points = ann.get("points", [])
                    
                    if not transcription:
                        print(f"[WARN] 라인 {line_idx + 1}, annotation {ann_idx}: transcription이 비어있음")
                        line_valid = False
                        break
                    
                    if not isinstance(points, list) or len(points) != 4:
                        print(f"[WARN] 라인 {line_idx + 1}, annotation {ann_idx}: points가 4개가 아님 (got {len(points) if isinstance(points, list) else 'not list'})")
                        line_valid = False
                        break
                    
                    # points가 [[x,y],...] 형태인지 확인
                    for pt in points:
                        if not isinstance(pt, list) or len(pt) != 2:
                            print(f"[WARN] 라인 {line_idx + 1}, annotation {ann_idx}: point 형식 오류")
                            line_valid = False
                            break
                
                if line_valid:
                    valid_count += 1
                    # 처음 num_samples개 출력
                    if valid_count <= num_samples:
                        print(f"[OK] 라인 {line_idx + 1}:")
                        print(f"  이미지: {image_path}")
                        print(f"  Annotations: {len(annotations)}개")
                        for ann_idx, ann in enumerate(annotations[:3]):  # 최대 3개만 출력
                            print(f"    [{ann_idx}] transcription: '{ann.get('transcription', '')[:30]}...', points: {len(ann.get('points', []))}개")
                else:
                    invalid_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"[WARN] 라인 {line_idx + 1}: JSON 파싱 오류: {e}")
                invalid_count += 1
    
    print(f"\n=== 검증 결과 ===")
    print(f"유효한 라인: {valid_count}개")
    print(f"무효한 라인: {invalid_count}개")
    
    return valid_count > 0


def process_dataset(
    data_root: Path,
    output_dir: Path,
    split: str,  # "train" or "val"
    task: str = "both",  # "det", "rec", "both"
    skip_noise: bool = True,
    max_crops: int = 200000,
    allow_fallback_bbox: bool = False,
    limit_json: Optional[int] = None,
    log_every: int = 1000,
    skip_if_image_missing: bool = True,
    only_categories: Optional[List[str]] = None,
    use_existing_det_label_path: Optional[Path] = None,
    fail_on_zero_det: bool = False,
):
    """데이터셋 변환 (train 또는 val)"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    det_dir = output_dir / "det"
    rec_dir = output_dir / "rec"
    crops_dir = rec_dir / "crops"
    logs_dir = output_dir / "logs"
    
    det_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_dir / f"convert_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    error_file = logs_dir / f"errors_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log_print(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    def log_error(json_path: Path, reason: str):
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"{json_path}\t{reason}\n")
    
    log_print(f"=== {split.upper()} 데이터셋 변환 시작 ===")
    log_print(f"Task: {task}, Limit: {limit_json}, Log every: {log_every}")
    
    # 기존 라벨 파일 사용 옵션이 있으면 복사 후 종료
    if use_existing_det_label_path is not None and task in ["det", "both"]:
        log_print(f"[INFO] 기존 라벨 파일 사용 모드: {use_existing_det_label_path}")
        if use_existing_det_label(use_existing_det_label_path, output_dir, split):
            log_print(f"[OK] 기존 라벨 파일 사용 완료. 변환 스킵.")
            return
        else:
            log_print(f"[ERROR] 기존 라벨 파일 사용 실패. 정상 변환을 계속 진행합니다.")
    
    # 폴더 스캔
    image_folders, label_folders = find_ke_folders(data_root, only_categories=only_categories)
    
    # Train/Val 분리
    if split == "train":
        label_folders = [f for f in label_folders if "Training" in str(f)]
        image_folders = [f for f in image_folders if "Training" in str(f)]
    else:
        label_folders = [f for f in label_folders if "Validation" in str(f)]
        image_folders = [f for f in image_folders if "Validation" in str(f)]
    
    log_print(f"이미지 폴더: {len(image_folders)}개")
    log_print(f"라벨 폴더: {len(label_folders)}개")
    
    # 이미지 인덱스 구축
    cache_file = output_dir / f"image_index_{split}.pkl"
    image_index = build_image_index(image_folders, cache_file=cache_file)
    
    # JSON 파일 수집
    all_json_files = []
    for label_folder in label_folders:
        for json_file in label_folder.rglob("*.json"):
            all_json_files.append(json_file)
    
    # 제한 적용
    if limit_json:
        all_json_files = all_json_files[:limit_json]
        log_print(f"JSON 파일 제한: {len(all_json_files)}개만 처리")
    else:
        log_print(f"총 JSON 파일: {len(all_json_files)}개")
    
    # 파일 열기 (task에 따라)
    det_f = None
    rec_f = None
    
    if task in ["det", "both"]:
        det_label_file = det_dir / f"{split}_label.txt"
        det_f = open(det_label_file, 'w', encoding='utf-8')
        log_print(f"Detection 라벨 파일: {det_label_file}")
    
    if task in ["rec", "both"]:
        rec_label_file = rec_dir / f"{split}_label.txt"
        rec_f = open(rec_label_file, 'w', encoding='utf-8')
        log_print(f"Recognition 라벨 파일: {rec_label_file}")
    
    try:
        det_count = 0
        rec_count = 0
        crop_count = 0
        skip_no_annotations = 0
        skip_no_filename = 0
        skip_image_not_found = 0
        skip_image_load_failed = 0
        error_count = 0
        crop_limit_logged = False  # crop 제한 메시지 출력 플래그
        
        for idx, json_file in enumerate(all_json_files):
            try:
                # 진행 로그
                if (idx + 1) % log_every == 0:
                    total_skip = skip_no_annotations + skip_no_filename + skip_image_not_found + skip_image_load_failed
                    log_print(f"[{idx + 1}/{len(all_json_files)}] 진행 중... (det: {det_count}, rec: {rec_count}, skip: {total_skip} [no_ann:{skip_no_annotations}, no_fn:{skip_no_filename}, no_img:{skip_image_not_found}, img_err:{skip_image_load_failed}], error: {error_count})")
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 공통 파서 사용 (json_file 경로 전달)
                image_filename, json_width, json_height, anns = parse_aihub_json(data, json_path=json_file)
                
                # Skip 사유별 분리
                if not anns:
                    skip_no_annotations += 1
                    log_error(json_file, "No valid annotations")
                    continue
                
                if image_filename is None:
                    skip_no_filename += 1
                    log_error(json_file, "No image filename parsed")
                    continue
                
                # 이미지 파일 찾기 (처음 10개에 대해서만 디버그)
                debug_search = (idx < 10)
                image_path = find_image_file(image_filename, image_index, debug=debug_search)
                
                if image_path is None:
                    # 확장자가 없는 경우 (json_path.stem 기반) 다양한 확장자 시도
                    if '.' not in image_filename:
                        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                            candidate = image_filename + ext
                            image_path = find_image_file(candidate, image_index, debug=False)
                            if image_path:
                                break
                    
                    if image_path is None:
                        skip_image_not_found += 1
                        reason = f"Image not found: {image_filename} (index_size={len(image_index)})"
                        log_error(json_file, reason)
                        if debug_search:
                            # 디버그: 인덱스에 실제로 있는지 확인
                            test_basename = image_filename.lower().strip()
                            test_base = test_basename.rsplit('.', 1)[0] if '.' in test_basename else test_basename
                            similar_in_index = [k for k in image_index.keys() if test_base[:15] in k.lower()][:3]
                            log_error(json_file, f"DEBUG: similar_in_index={similar_in_index}")
                        
                        if not skip_if_image_missing:
                            continue
                        else:
                            continue
                
                # 이미지 로드 (det task일 때도 크기 정보 필요)
                img = None
                img_h, img_w = None, None
                
                # det task일 때도 이미지 크기를 읽어야 좌표 클램핑 가능
                img = safe_imread(image_path)
                if img is None:
                    skip_image_load_failed += 1
                    reason = f"Failed to read image: {image_path}"
                    log_error(json_file, reason)
                    continue
                img_h, img_w = img.shape[:2]
                
                # JSON에서 읽은 width/height가 있으면 검증
                if json_width is not None and json_height is not None:
                    if abs(json_width - img_w) > 10 or abs(json_height - img_h) > 10:
                        # 크기 불일치 경고 (하지만 계속 진행)
                        pass
                
                det_annotations = []
                
                # crop 제한 확인 (한 번만)
                crop_limit_reached = False
                if task in ["rec", "both"] and crop_count >= max_crops:
                    crop_limit_reached = True
                    if not crop_limit_logged:
                        log_print(f"[WARN] 최대 crop 개수 ({max_crops})에 도달했습니다. 이후 crop 생성을 중단합니다.")
                        crop_limit_logged = True
                
                for ann_idx, ann in enumerate(anns):
                    text = ann["text"]
                    polygon = ann["polygon"]
                    
                    # 폴리곤 정규화 (4점으로) - 이미지 크기 정보 전달
                    normalized_points = normalize_polygon_points(
                        polygon, 
                        allow_fallback_bbox=allow_fallback_bbox,
                        img_width=img_w,
                        img_height=img_h,
                        min_area=4.0,  # 최소 면적 4픽셀 (2x2)
                        fix_self_intersection=True
                    )
                    if normalized_points is None:
                        # Invalid polygon 스킵 (로그에 기록)
                        log_error(json_file, f"Invalid polygon in annotation {ann_idx}: {polygon}")
                        continue
                    
                    # 텍스트 클린징
                    cleaned_text = clean_text(text, skip_noise=skip_noise)
                    if cleaned_text is None:
                        continue
                    
                    # Detection 라벨 추가
                    if task in ["det", "both"]:
                        det_annotations.append({
                            "transcription": cleaned_text,
                            "points": normalized_points,
                            "difficult": 0
                        })
                    
                    # Recognition용 crop 생성 (제한 확인)
                    if task in ["rec", "both"] and img is not None and not crop_limit_reached:
                        # 폴리곤 좌표 추출
                        pts = np.array(normalized_points, dtype=np.int32)
                        x_min = max(0, int(pts[:, 0].min()))
                        x_max = min(img_w, int(pts[:, 0].max()))
                        y_min = max(0, int(pts[:, 1].min()))
                        y_max = min(img_h, int(pts[:, 1].max()))
                        
                        if x_max > x_min and y_max > y_min:
                            # 마스크로 crop
                            mask = np.zeros((img_h, img_w), dtype=np.uint8)
                            cv2.fillPoly(mask, [pts], 255)
                            
                            # ROI 추출
                            roi = img[y_min:y_max, x_min:x_max]
                            
                            if roi.size > 0:
                                # Crop 이미지 저장
                                crop_filename = f"{json_file.stem}_{ann_idx:04d}_{crop_count:08d}.jpg"
                                crop_path = crops_dir / crop_filename
                                
                                if safe_imsave(crop_path, roi):
                                    # Recognition 라벨 추가
                                    rel_crop_path = f"rec/crops/{crop_filename}"
                                    rec_f.write(f"{rel_crop_path}\t{cleaned_text}\n")
                                    rec_count += 1
                                    crop_count += 1
                                    
                                    # crop_count가 max_crops에 도달하면 더 이상 crop 생성 중단
                                    if crop_count >= max_crops:
                                        crop_limit_reached = True
                                        if not crop_limit_logged:
                                            log_print(f"[WARN] 최대 crop 개수 ({max_crops})에 도달했습니다. 이후 crop 생성을 중단합니다.")
                                            crop_limit_logged = True
                
                # Detection 라벨 저장
                if task in ["det", "both"] and det_annotations:
                    abs_image_path = str(image_path.resolve())
                    det_label_line = json.dumps(det_annotations, ensure_ascii=False)
                    det_f.write(f"{abs_image_path}\t{det_label_line}\n")
                    det_count += 1
                
            except KeyboardInterrupt:
                log_print("\n[WARN] 사용자 중단 (Ctrl+C)")
                raise
            except Exception as e:
                error_count += 1
                reason = f"{type(e).__name__}: {str(e)}"
                log_error(json_file, reason)
                log_print(f"[ERROR] Error processing {json_file.name}: {reason}")
                # 계속 진행 (중단 금지)
                continue
        
        total_skip = skip_no_annotations + skip_no_filename + skip_image_not_found + skip_image_load_failed
        
        log_print(f"\n=== 변환 완료 ===")
        if task in ["det", "both"]:
            log_print(f"Detection 이미지: {det_count}개")
        if task in ["rec", "both"]:
            log_print(f"Recognition crops: {crop_count}개 (라벨: {rec_count}개)")
        log_print(f"총 스킵: {total_skip}개")
        log_print(f"  - No valid annotations: {skip_no_annotations}개")
        log_print(f"  - No image filename: {skip_no_filename}개")
        log_print(f"  - Image not found: {skip_image_not_found}개")
        log_print(f"  - Image load failed: {skip_image_load_failed}개")
        log_print(f"에러: {error_count}개")
        
        # 디버깅: det_count가 0이면 경고
        if task in ["det", "both"] and det_count == 0:
            log_print(f"\n[WARN] ⚠️  WARNING: Detection 이미지가 0개입니다!")
            log_print(f"[WARN] 원인 분석 필요:")
            log_print(f"[WARN]   - No valid annotations: {skip_no_annotations}개")
            log_print(f"[WARN]   - No image filename: {skip_no_filename}개")
            log_print(f"[WARN]   - Image not found: {skip_image_not_found}개")
            log_print(f"[WARN]   - Image load failed: {skip_image_load_failed}개")
            
            if fail_on_zero_det:
                log_print(f"\n[ERROR] fail_on_zero_det 옵션이 설정되어 있어 종료합니다.")
                raise ValueError(f"Detection 이미지가 0개입니다. 원인 분석 후 재시도하세요.")
        
        # 검증: limit_json이 있고 det_count > 0이면 검증 실행
        if limit_json and limit_json <= 20 and task in ["det", "both"] and det_count > 0:
            log_print(f"\n=== 라벨 검증 실행 (limit_json={limit_json}, det_count={det_count}) ===")
            det_label_file = det_dir / f"{split}_label.txt"
            validate_det_label(det_label_file, num_samples=3)
        log_print(f"\n출력 파일:")
        if task in ["det", "both"]:
            log_print(f"  - Detection: {det_label_file}")
        if task in ["rec", "both"]:
            log_print(f"  - Recognition: {rec_label_file}")
        log_print(f"  - 로그: {log_file}")
        log_print(f"  - 에러 로그: {error_file}")
    
    finally:
        if det_f:
            det_f.close()
        if rec_f:
            rec_f.close()


def main():
    parser = argparse.ArgumentParser(description="AIHub JSON을 PaddleOCR 포맷으로 변환")
    parser.add_argument("--data_root", type=str, default=r"S:\OCR_Data",
                        help="AIHub 데이터 루트 경로")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="출력 디렉토리 (기본: 프로젝트 내 data/)")
    parser.add_argument("--split", type=str, choices=["train", "val", "both"], default="both",
                        help="변환할 데이터셋 (train, val, both)")
    parser.add_argument("--task", type=str, choices=["det", "rec", "both"], default="both",
                        help="변환할 작업 (det, rec, both)")
    parser.add_argument("--skip_noise", action="store_true", default=True,
                        help="잡음 텍스트 스킵")
    parser.add_argument("--max_crops", type=int, default=200000,
                        help="최대 crop 개수 (기본: 200000)")
    parser.add_argument("--allow_fallback_bbox", action="store_true",
                        help="4점이 아니어도 bbox로 변환 허용")
    parser.add_argument("--limit_json", type=int, default=None,
                        help="처리할 JSON 개수 제한 (테스트용)")
    parser.add_argument("--log_every", type=int, default=1000,
                        help="진행 로그 출력 주기 (기본: 1000)")
    parser.add_argument("--skip_if_image_missing", action="store_true", default=True,
                        help="이미지 없으면 스킵 (기본: True)")
    parser.add_argument("--only_categories", type=str, nargs="+", default=None,
                        help="특정 카테고리만 처리 (예: CT LF PB)")
    parser.add_argument("--use_existing_det_label", type=str, default=None,
                        help="기존 PaddleOCR det 라벨 파일 경로 (변환 스킵하고 복사)")
    parser.add_argument("--fail_on_zero_det", action="store_true",
                        help="det_count가 0이면 즉시 종료 (디버깅용)")
    parser.add_argument("--validate", action="store_true",
                        help="변환 후 라벨 파일 검증 실행")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "data"
    else:
        output_dir = Path(args.output_dir)
    
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print(f"[ERROR] 데이터 루트가 존재하지 않습니다: {data_root}")
        return
    
    # 기존 라벨 파일 경로 설정
    use_existing_det_label_path = None
    if args.use_existing_det_label:
        use_existing_det_label_path = Path(args.use_existing_det_label)
        if not use_existing_det_label_path.exists():
            print(f"[ERROR] 기존 라벨 파일이 없습니다: {use_existing_det_label_path}")
            return
    
    # 변환 실행
    if args.split in ["train", "both"]:
        print("\n" + "="*60)
        process_dataset(
            data_root=data_root,
            output_dir=output_dir,
            split="train",
            task=args.task,
            skip_noise=args.skip_noise,
            max_crops=args.max_crops,
            allow_fallback_bbox=args.allow_fallback_bbox,
            limit_json=args.limit_json,
            log_every=args.log_every,
            skip_if_image_missing=args.skip_if_image_missing,
            only_categories=args.only_categories,
            use_existing_det_label_path=use_existing_det_label_path,
            fail_on_zero_det=args.fail_on_zero_det,
        )
    
    if args.split in ["val", "both"]:
        print("\n" + "="*60)
        process_dataset(
            data_root=data_root,
            output_dir=output_dir,
            split="val",
            task=args.task,
            skip_noise=args.skip_noise,
            max_crops=args.max_crops,
            allow_fallback_bbox=args.allow_fallback_bbox,
            limit_json=args.limit_json,
            log_every=args.log_every,
            skip_if_image_missing=args.skip_if_image_missing,
            only_categories=args.only_categories,
            use_existing_det_label_path=use_existing_det_label_path,
            fail_on_zero_det=args.fail_on_zero_det,
        )
    
    # 검증 실행
    if args.validate:
        print("\n" + "="*60)
        print("=== 라벨 파일 검증 ===")
        if args.split in ["train", "both"]:
            train_label = output_dir / "det" / "train_label.txt"
            if train_label.exists():
                validate_det_label(train_label, num_samples=3)
        if args.split in ["val", "both"]:
            val_label = output_dir / "det" / "val_label.txt"
            if val_label.exists():
                validate_det_label(val_label, num_samples=3)
    
    print("\n[OK] 모든 변환이 완료되었습니다!")


if __name__ == "__main__":
    main()
