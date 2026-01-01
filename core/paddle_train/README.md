# PaddleOCR 커스텀 학습 파이프라인

AIHub 다중언어 OCR 데이터셋(한국어+영어)을 사용하여 PaddleOCR 모델을 fine-tuning하는 파이프라인입니다.

## 📁 폴더 구조

```
core/paddle_train/
├── _setup_folders.py             # 폴더 구조 자동 생성 스크립트
├── 00_verify_dataset.ipynb       # 데이터셋 검증 및 시각화
├── 01_convert_to_paddle_format.py # JSON → PaddleOCR 포맷 변환
├── 02_make_train_val_list.py     # 학습 리스트 생성
├── 03_train_det.md               # Detection 학습 가이드
├── 04_train_rec.md               # Recognition 학습 가이드
├── 05_infer_demo.py              # 추론 데모 스크립트
├── 06_integration_guide.md       # 프로젝트 통합 가이드
├── 07_performance_analysis.md    # 학습 성능 분석 결과
├── 08_export_inference_model.bat # Inference 모델 변환 스크립트
├── README.md                     # 본 문서
├── configs/                      # 학습 설정 파일
│   └── det_ke_finetune.yml       # Detection fine-tuning 설정
├── data/                         # 변환된 데이터
│   ├── det/                      # Detection 라벨
│   │   ├── train_label.txt       # 학습용 라벨 파일
│   │   └── val_label.txt         # 검증용 라벨 파일
│   ├── image_index_train.pkl     # 학습 이미지 인덱스 캐시
│   ├── image_index_val.pkl       # 검증 이미지 인덱스 캐시
│   └── logs/                     # 실행 로그
│       ├── convert_train_*.log   # 데이터 변환 로그
│       ├── convert_val_*.log     # 검증 데이터 변환 로그
│       └── errors_*.log          # 에러 로그
└── (output은 상위 디렉토리에 생성됨)
    output/det_ke_model/          # Detection 모델 체크포인트
        ├── config.yml            # 실제 사용된 설정 파일
        ├── best_accuracy.*       # 최고 성능 모델
        ├── best_model/           # 최고 모델 백업
        ├── train.log             # 학습 로그
        └── train_console.log     # 콘솔 출력 로그
```

## 🚀 사용 순서

### 1. 데이터셋 검증 (00_verify_dataset.ipynb)

Jupyter Notebook을 실행하여 데이터셋을 검증합니다:

1. `S:\OCR_Data`에서 "_KE_" 폴더를 스캔
2. 이미지-라벨 매칭 검증
3. 텍스트 품질 통계
4. 샘플 시각화

**실행 방법:**
```bash
jupyter notebook 00_verify_dataset.ipynb
```

### 2. 데이터 변환 (01_convert_to_paddle_format.py)

AIHub JSON 형식을 PaddleOCR 학습 포맷으로 변환합니다.

**실행 방법:**
```bash
python 01_convert_to_paddle_format.py --split both --max_crops 200000
```

**옵션:**
- `--data_root`: AIHub 데이터 루트 경로 (기본: `S:\OCR_Data`)
- `--output_dir`: 출력 디렉토리 (기본: `data/`)
- `--split`: 변환할 데이터셋 (`train`, `val`, `both`)
- `--max_crops`: 최대 crop 개수 (기본: 200000)

**출력:**
- `data/det/train_label.txt`, `data/det/val_label.txt`
- `data/rec/crops/*.jpg` (crop 이미지들)
- `data/rec/train_label.txt`, `data/rec/val_label.txt`

### 3. 학습 리스트 확인 (02_make_train_val_list.py)

변환된 라벨 파일을 확인하고 요약 정보를 생성합니다.

**실행 방법:**
```bash
python 02_make_train_val_list.py
```

**출력:**
- 각 라벨 파일의 라인 수 확인
- `data/dataset_summary.txt` 생성
- PaddleOCR 설정에 사용할 경로 정보 출력

### 4. Detection 모델 학습 (03_train_det.md 참조)

PP-OCRv4 Detection 모델을 fine-tuning합니다.

**사전 요구사항:**
- PaddleOCR 학습 도구 설치
- GPU 환경 (권장)

**실행 방법:**
```bash
# PaddleOCR 디렉토리로 이동
cd path/to/PaddleOCR

# 학습 실행
python tools/train.py \
  -c C:\Pyg\Projects\semi\yuzyproject-aimodels\core\paddle_train\configs\det_ke_finetune.yml \
  -o Train.dataset.label_file_list=[절대경로/train_label.txt] \
     Eval.dataset.label_file_list=[절대경로/val_label.txt]
```

**학습 완료 후:**
```bash
# 추론용 모델로 변환
python tools/export_model.py \
  -c configs/det_ke_finetune.yml \
  -o Global.checkpoints=output/det_ke_model/best_accuracy \
     Global.save_inference_dir=output/det_ke_inference
```

### 5. 추론 테스트 (05_infer_demo.py)

학습된 모델로 추론을 테스트합니다.

**실행 방법:**
```bash
# Detection만 테스트
python 05_infer_demo.py \
  --det_model_dir output/det_ke_inference \
  --image_dir path/to/test/images \
  --det_only

# Detection + Recognition 테스트
python 05_infer_demo.py \
  --det_model_dir output/det_ke_inference \
  --rec_model_dir output/rec_ke_inference \
  --image_dir path/to/test/images
```

**출력:**
- `output/inference_results/result_*.jpg`: 결과 이미지
- `output/inference_results/results_summary.json`: 결과 요약

### 6. capture.py 통합 (06_integration_guide.md 참조)

학습된 모델을 `core/capture.py`에 통합합니다.

## 📝 주요 설정 파일

### configs/det_ke_finetune.yml

Detection 모델 학습 설정:
- Pretrained 모델: PP-OCRv4 Detection (ch_PP-OCRv4_det_train/best_accuracy)
- 배치 크기: 16 (GPU 메모리에 따라 조정, OOM 발생 시 8로 낮춤)
- 학습 에폭: 20 (1주일 내 완료 목표)
- 이미지 크기: 640x640 (EastRandomCropData)
- 데이터 증강: 좌우 반전, 회전, 리사이즈
- Warmup: 1 epoch
- Eval 주기: 4000 step

**Windows 환경 고려사항:**
- `num_workers`: 4 (Train/Eval 모두)
- `batch_size_per_card`: 16 (OOM 없으면 16, OOM이면 8)
- `save_epoch_step`: 1 (매 epoch마다 체크포인트 저장)

## 🔧 문제 해결

### 1. CUDA out of memory

- `batch_size_per_card`를 줄이세요 (8 → 4 → 2)
- `num_workers`를 줄이세요 (2 → 1 → 0)

### 2. 경로 오류

- 모든 경로를 절대 경로로 변경
- Windows 경로는 백슬래시를 슬래시로 변경하거나 이스케이프 처리

### 3. 이미지 로딩 실패 (한글 경로)

- `safe_imread`, `safe_imsave` 함수 사용
- `np.fromfile` + `cv2.imdecode` 방식 사용

### 4. 데이터 변환 시간 오래 걸림

- `max_crops` 옵션으로 crop 개수 제한
- 필요시 병렬 처리 고려

## 📊 데이터셋 정보

- **원본 데이터**: `S:\OCR_Data`
- **사용 언어**: 한국어 + 영어 (KE)
- **포맷**: AIHub JSON (Images + Annotation)
- **학습/검증 분리**: Training / Validation

## 🎯 학습 목표

1. **Detection**: 코드 스크린샷에서 텍스트 박스를 정확하게 검출
2. **Recognition**: (선택) 검출된 텍스트를 정확하게 인식

## 📚 참고 자료

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv4 모델](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/PP-OCRv4_introduction.md)
- [AIHub 다중언어 OCR 데이터](https://aihub.or.kr/)

## ⚠️ 주의사항

1. **원본 데이터 보호**: `S:\OCR_Data`는 읽기 전용으로 사용. 절대 수정하지 마세요.
2. **GPU 메모리**: Windows 환경에서 GPU 메모리 부족 시 배치 크기 조정 필수 (16 → 12 → 8)
3. **학습 시간**: 
   - 현재 설정 기준: **epoch당 약 6시간, 전체 20 epoch 완료까지 약 5일** 소요 예상
   - 실제 소요 시간은 데이터셋 크기와 GPU 성능에 따라 달라질 수 있습니다
4. **학습 재개**: 학습 중단 시 체크포인트에서 재개 가능 (매 epoch마다 저장됨)
5. **설정 파일**: `output/det_ke_model/config.yml`은 실제 사용된 설정 기록 (참고용)

