"""
PaddleOCR Detection 모델 추론 데모 스크립트

학습된 Detection 모델을 사용하여 이미지에서 텍스트 영역을 검출하고 시각화합니다.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import json

import numpy as np
import cv2


def safe_imread(image_path: Path) -> Optional[np.ndarray]:
    """Windows 한글 경로를 안전하게 이미지 읽기"""
    try:
        img_array = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Failed to read {image_path}: {e}")
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
        print(f"Failed to save {image_path}: {e}")
        return False


def draw_detection_result(
    image: np.ndarray,
    boxes: List[List[List[int]]],
    texts: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
) -> np.ndarray:
    """Detection 결과를 이미지에 그리기"""
    result_img = image.copy()
    
    for idx, box in enumerate(boxes):
        if len(box) != 4:
            continue
        
        # 폴리곤 그리기
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(result_img, [pts], True, (0, 255, 0), 2)
        
        # 텍스트와 점수 표시 (있을 경우)
        if texts is not None and idx < len(texts):
            text = texts[idx]
            if scores is not None and idx < len(scores):
                label = f"{text} ({scores[idx]:.2f})"
            else:
                label = text
            
            # 텍스트 배경
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            x, y = int(pts[0][0]), int(pts[0][1])
            cv2.rectangle(
                result_img,
                (x, y - text_height - baseline - 5),
                (x + text_width, y),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                result_img,
                label,
                (x, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
    
    return result_img


def paddle_ocr_det_only(
    image_bgr: np.ndarray,
    det_model_dir: str,
    use_gpu: bool = True,
) -> Tuple[List[List[List[int]]], List[float]]:
    """
    PaddleOCR Detection만 수행 (Recognition 없이)
    
    Args:
        image_bgr: BGR 형식 이미지 (numpy array)
        det_model_dir: Detection 모델 디렉토리 경로
        use_gpu: GPU 사용 여부
    
    Returns:
        boxes: 검출된 박스 좌표 리스트 [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
        scores: 각 박스의 신뢰도 점수 리스트
    """
    try:
        from paddleocr import PaddleOCR
        
        # Detection만 사용하는 OCR 객체 생성
        ocr = PaddleOCR(
            det_model_dir=det_model_dir,
            rec=False,  # Recognition 비활성화
            use_angle_cls=False,
            use_gpu=use_gpu,
            lang='korean',
            show_log=False
        )
        
        # 추론 수행
        result = ocr.ocr(image_bgr, cls=False, det=True, rec=False)
        
        if result is None or len(result) == 0 or result[0] is None:
            return [], []
        
        boxes = []
        scores = []
        
        for line in result[0]:
            if line is None:
                continue
            box = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            score = line[1] if len(line) > 1 else 1.0
            
            boxes.append(box)
            scores.append(score)
        
        return boxes, scores
        
    except ImportError:
        print("ERROR: paddleocr가 설치되지 않았습니다.")
        print("설치 방법: pip install paddleocr")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: 추론 중 오류 발생: {e}")
        return [], []


def paddle_ocr_full(
    image_bgr: np.ndarray,
    det_model_dir: Optional[str] = None,
    rec_model_dir: Optional[str] = None,
    use_gpu: bool = True,
) -> Tuple[List[List[List[int]]], List[str], List[float]]:
    """
    PaddleOCR Detection + Recognition 수행
    
    Args:
        image_bgr: BGR 형식 이미지 (numpy array)
        det_model_dir: Detection 모델 디렉토리 경로 (None이면 기본 모델 사용)
        rec_model_dir: Recognition 모델 디렉토리 경로 (None이면 기본 모델 사용)
        use_gpu: GPU 사용 여부
    
    Returns:
        boxes: 검출된 박스 좌표 리스트
        texts: 인식된 텍스트 리스트
        scores: 각 박스의 신뢰도 점수 리스트
    """
    try:
        from paddleocr import PaddleOCR
        
        kwargs = {
            'use_angle_cls': False,
            'use_gpu': use_gpu,
            'lang': 'korean',
            'show_log': False
        }
        
        if det_model_dir:
            kwargs['det_model_dir'] = det_model_dir
        if rec_model_dir:
            kwargs['rec_model_dir'] = rec_model_dir
        
        ocr = PaddleOCR(**kwargs)
        
        # 추론 수행
        result = ocr.ocr(image_bgr, cls=False)
        
        if result is None or len(result) == 0 or result[0] is None:
            return [], [], []
        
        boxes = []
        texts = []
        scores = []
        
        for line in result[0]:
            if line is None:
                continue
            box = line[0]
            text_info = line[1]
            
            if isinstance(text_info, tuple):
                text, score = text_info
            else:
                text = str(text_info)
                score = 1.0
            
            boxes.append(box)
            texts.append(text)
            scores.append(score)
        
        return boxes, texts, scores
        
    except ImportError:
        print("ERROR: paddleocr가 설치되지 않았습니다.")
        print("설치 방법: pip install paddleocr")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: 추론 중 오류 발생: {e}")
        return [], [], []


def infer_images(
    image_paths: List[Path],
    det_model_dir: str,
    rec_model_dir: Optional[str] = None,
    output_dir: Path = None,
    use_gpu: bool = True,
    det_only: bool = False,
):
    """여러 이미지에 대해 추론 수행 및 결과 저장"""
    
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "output" / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"추론 결과 저장 위치: {output_dir}")
    print("="*60)
    
    results_summary = []
    
    for idx, image_path in enumerate(image_paths):
        print(f"\n[{idx+1}/{len(image_paths)}] 처리 중: {image_path.name}")
        
        # 이미지 로드
        image = safe_imread(image_path)
        if image is None:
            print(f"  ❌ 이미지 로드 실패")
            continue
        
        # 추론 수행
        if det_only:
            boxes, scores = paddle_ocr_det_only(image, det_model_dir, use_gpu)
            texts = None
        else:
            boxes, texts, scores = paddle_ocr_full(
                image, det_model_dir, rec_model_dir, use_gpu
            )
        
        print(f"  ✅ 검출된 박스: {len(boxes)}개")
        
        if texts is not None:
            for i, (box, text, score) in enumerate(zip(boxes, texts, scores)):
                print(f"    [{i+1}] {text} (score: {score:.3f})")
        
        # 결과 시각화
        result_img = draw_detection_result(image, boxes, texts, scores)
        
        # 저장
        output_path = output_dir / f"result_{image_path.stem}.jpg"
        if safe_imsave(output_path, result_img):
            print(f"  💾 저장 완료: {output_path}")
        else:
            print(f"  ❌ 저장 실패: {output_path}")
        
        # 요약 정보 저장
        results_summary.append({
            'image': str(image_path),
            'num_boxes': len(boxes),
            'boxes': boxes,
            'texts': texts if texts else [],
            'scores': scores if scores else []
        })
    
    # 요약 JSON 저장
    summary_file = output_dir / "results_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 요약 정보 저장: {summary_file}")
    
    print("\n" + "="*60)
    print("추론 완료!")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR Detection 추론 데모")
    parser.add_argument("--det_model_dir", type=str, required=True,
                        help="Detection 모델 디렉토리 경로")
    parser.add_argument("--rec_model_dir", type=str, default=None,
                        help="Recognition 모델 디렉토리 경로 (선택)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="테스트할 이미지 디렉토리")
    parser.add_argument("--image_files", type=str, nargs="+", default=None,
                        help="테스트할 이미지 파일 경로들")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="결과 저장 디렉토리")
    parser.add_argument("--det_only", action="store_true",
                        help="Detection만 수행 (Recognition 없이)")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="GPU 사용 (기본: True)")
    parser.add_argument("--use_cpu", action="store_true",
                        help="CPU 사용 (--use_gpu와 함께 사용하면 CPU 우선)")
    
    args = parser.parse_args()
    
    # 이미지 파일 수집
    image_paths = []
    
    if args.image_files:
        image_paths = [Path(f) for f in args.image_files]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths = sorted(image_paths)[:5]  # 최대 5개
    else:
        # 기본: 프로젝트 내 샘플 이미지 또는 사용자 입력 요청
        print("❌ 이미지 파일 또는 디렉토리를 지정해주세요.")
        print("  --image_dir <경로> 또는 --image_files <파일1> <파일2> ...")
        return
    
    if not image_paths:
        print("❌ 처리할 이미지가 없습니다.")
        return
    
    use_gpu = args.use_gpu and not args.use_cpu
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # 추론 실행
    infer_images(
        image_paths=image_paths,
        det_model_dir=args.det_model_dir,
        rec_model_dir=args.rec_model_dir,
        output_dir=output_dir,
        use_gpu=use_gpu,
        det_only=args.det_only,
    )


if __name__ == "__main__":
    main()

