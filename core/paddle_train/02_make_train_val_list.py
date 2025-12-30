"""
변환된 라벨 파일을 기반으로 PaddleOCR 학습용 리스트 파일 생성
Detection과 Recognition용 리스트를 정리합니다.
"""

import argparse
from pathlib import Path
from datetime import datetime


def make_train_val_lists(data_dir: Path, output_dir: Path = None):
    """
    data_dir 아래의 변환된 라벨 파일들을 확인하고
    PaddleOCR 학습에 필요한 리스트 파일을 생성합니다.
    """
    if output_dir is None:
        output_dir = data_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    det_dir = data_dir / "det"
    rec_dir = data_dir / "rec"
    
    # Detection 리스트
    det_train_file = det_dir / "train_label.txt"
    det_val_file = det_dir / "val_label.txt"
    
    # Recognition 리스트
    rec_train_file = rec_dir / "train_label.txt"
    rec_val_file = rec_dir / "val_label.txt"
    
    # 존재 여부 확인
    files_info = {
        "Detection Train": det_train_file,
        "Detection Val": det_val_file,
        "Recognition Train": rec_train_file,
        "Recognition Val": rec_val_file,
    }
    
    print("="*60)
    print("변환된 라벨 파일 확인")
    print("="*60)
    
    for name, file_path in files_info.items():
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"✅ {name}: {line_count}개 라인 ({file_path})")
        else:
            print(f"❌ {name}: 파일 없음 ({file_path})")
    
    # 요약 파일 생성
    summary_file = output_dir / "dataset_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"데이터셋 요약 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("="*60 + "\n\n")
        
        for name, file_path in files_info.items():
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as label_f:
                    line_count = sum(1 for _ in label_f)
                f.write(f"{name}: {line_count}개\n")
                f.write(f"  경로: {file_path.resolve()}\n\n")
            else:
                f.write(f"{name}: 파일 없음\n\n")
    
    print(f"\n✅ 요약 파일 생성: {summary_file}")
    
    # PaddleOCR 설정 파일에서 사용할 경로 정보 출력
    print("\n" + "="*60)
    print("PaddleOCR 설정 파일에 사용할 경로 정보")
    print("="*60)
    
    base_path = data_dir.resolve()
    
    print("\n[Detection]")
    if det_train_file.exists():
        print(f"train_file: {det_train_file.resolve()}")
    if det_val_file.exists():
        print(f"val_file: {det_val_file.resolve()}")
    
    print("\n[Recognition]")
    if rec_train_file.exists():
        print(f"train_file: {rec_train_file.resolve()}")
    if rec_val_file.exists():
        print(f"val_file: {rec_val_file.resolve()}")
    
    print("\n⚠️  참고:")
    print("  - 위 절대 경로를 PaddleOCR config yaml 파일의 train_file, val_file에 설정하세요.")
    print("  - 또는 상대 경로를 사용하려면 PaddleOCR 실행 디렉토리를 기준으로 경로를 수정하세요.")


def main():
    parser = argparse.ArgumentParser(description="변환된 라벨 파일 확인 및 요약")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="데이터 디렉토리 (기본: 프로젝트 내 data/)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="출력 디렉토리 (기본: data_dir와 동일)")
    
    args = parser.parse_args()
    
    if args.data_dir is None:
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"
    else:
        data_dir = Path(args.data_dir)
    
    if args.output_dir is None:
        output_dir = None
    else:
        output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"❌ 데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        return
    
    make_train_val_lists(data_dir, output_dir)


if __name__ == "__main__":
    main()

