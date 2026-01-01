# PaddleOCR 코드 문법 인식 모델 학습 - 빠른 시작 가이드

이 문서는 **RTX 4070Ti GPU 환경**에서 PaddleOCR v5 코드 문법 인식 모델 학습을 위한 빠른 시작 가이드입니다.

## 빠른 시작 (3단계)

### 1단계: 환경 설정

```bash
# 1. 저장소 클론 및 브랜치 전환
git clone <repository-url>
cd semi
git checkout gonida

# 2. 가상환경 생성 및 활성화
cd yuzyproject-aimodels
python -m venv venv_ocr
venv_ocr\Scripts\activate  # Windows
# source venv_ocr/bin/activate  # Linux

# 3. PaddlePaddle GPU 설치 (CUDA 11.8 기준)
pip install paddlepaddle-gpu==3.2.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 4. 필수 패키지 설치
cd ..
cd PaddleOCR
pip install -r requirements.txt
cd ..
cd yuzyproject-aimodels
pip install -r requirements.txt
```

### 2단계: 데이터셋 생성

```bash
# yuzyproject-aimodels 디렉토리에서 실행
python prepare_code_syntax_dataset.py
```

**예상 소요 시간:** 10-30분 (데이터셋 크기에 따라)

### 3단계: 학습 시작

```bash
cd ../PaddleOCR
python tools/train.py -c configs/rec/rec_code_syntax_finetune.yml
```

## 파일 구조

```
semi/
├── PaddleOCR/
│   ├── configs/rec/rec_code_syntax_finetune.yml  # 학습 설정 (RTX 4070Ti 최적화)
│   ├── tools/train.py                             # 학습 스크립트
│   └── output/rec/code_syntax/                    # 학습 결과 (체크포인트, 로그)
│
└── yuzyproject-aimodels/
    ├── prepare_code_syntax_dataset.py             # 데이터셋 생성 스크립트
    └── code_syntax_dataset/                       # 생성된 데이터셋
        ├── train_images/                          # 학습 이미지
        ├── val_images/                            # 검증 이미지
        ├── train_list.txt                         # 학습 라벨
        └── val_list.txt                           # 검증 라벨
```

## 주요 설정 (RTX 4070Ti 최적화)

현재 설정 파일 (`PaddleOCR/configs/rec/rec_code_syntax_finetune.yml`)에는 다음 최적화가 적용되어 있습니다:

- ✅ **Batch Size**: 32 (RTX 4070Ti 메모리 활용)
- ✅ **Num Workers**: 6 (데이터 로딩 속도 향상)
- ✅ **Pin Memory**: true (GPU 전송 속도 향상)
- ✅ **Gradient Clipping**: clip_norm 5.0 (NaN 방지)
- ✅ **Mixed Precision**: use_amp true (메모리 효율)
- ✅ **NRTR 전용**: CTCLoss 제외 (Windows 호환성)

## 모듈 버전

### 핵심 패키지

- `paddlepaddle==3.2.0` (GPU 버전)
- `paddleocr>=2.9.0`
- `Pillow>=12.0.0`
- `numpy>=1.26.4`
- `opencv-python`, `opencv-contrib-python`
- `PyYAML>=6.0.2`

전체 패키지 목록은 다음 파일을 참조하세요:
- `yuzyproject-aimodels/requirements.txt`
- `PaddleOCR/requirements.txt`

## 학습 모니터링

### 로그 확인

```bash
# 실시간 로그 확인 (Windows)
type PaddleOCR\output\rec\code_syntax\train.log

# 실시간 로그 확인 (Linux)
tail -f PaddleOCR/output/rec/code_syntax/train.log
```

### 주요 메트릭

- **NRTRLoss**: 손실 값 (NaN 발생 시 학습 중단 필요)
- **norm_edit_dis**: 정규화된 편집 거리 (낮을수록 좋음)
- **acc**: 정확도

### 체크포인트 관리

**저장 위치:**
- `output/rec/code_syntax/best_accuracy.pdparams`: 최고 성능 모델
- `output/rec/code_syntax/latest.pdparams`: 최신 모델

**학습 재개:**
```bash
python tools/train.py -c configs/rec/rec_code_syntax_finetune.yml \
    -o Global.checkpoints=./output/rec/code_syntax/latest
```

## 문제 해결

### GPU 인식 확인

```bash
python -c "import paddle; paddle.utils.run_check()"
```

### NaN Loss 발생 시

1. 학습 중단 (Ctrl+C)
2. NaN 발생 이전 체크포인트 확인
3. 해당 체크포인트부터 재개:
   ```bash
   python tools/train.py -c configs/rec/rec_code_syntax_finetune.yml \
       -o Global.checkpoints=./output/rec/code_syntax/best_accuracy
   ```

### 메모리 부족 시

설정 파일에서 다음 값 조정:
- `batch_size_per_card: 32 → 16`
- `num_workers: 6 → 4`

## 상세 가이드

더 자세한 내용은 `PaddleOCR/TRAINING_SETUP.md`를 참조하세요.

## 참고사항

- **CTC 관련**: 현재 설정은 NRTR 전용입니다. CTCLoss는 사용하지 않습니다.
- **Windows 호환성**: warp-ctc 호환성 문제로 CTCLoss를 제외했습니다.
- **데이터셋**: `code_syntax_dataset/` 디렉토리는 `.gitignore`에 포함되어 있어 Git에 업로드되지 않습니다.

