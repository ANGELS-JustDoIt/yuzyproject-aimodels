# 데이터셋 생성 및 학습 설정 가이드

이 문서는 **PaddleOCR 코드 문법 인식 모델**을 위한 데이터셋 생성부터 학습 시작까지의 전체 과정을 상세히 설명합니다.

## 목차

1. [환경 설정](#1-환경-설정)
2. [데이터셋 생성](#2-데이터셋-생성)
3. [학습 시작](#3-학습-시작)
4. [문제 해결](#4-문제-해결)

---

## 1. 환경 설정

### 1.1 필수 요구사항

- **OS**: Windows 10/11
- **Python**: 3.12.10
- **GPU**: NVIDIA GPU (CUDA 지원)
- **CUDA**: 12.x (RTX 50 시리즈의 경우)

### 1.2 저장소 클론 및 설정

```bash
# 저장소 클론
git clone <repository-url>
cd semi
git checkout gonida
```

### 1.3 가상환경 생성 및 활성화

```bash
cd yuzyproject-aimodels

# 가상환경 생성
python -m venv venv_ocr

# 가상환경 활성화 (Windows)
venv_ocr\Scripts\activate
```

### 1.4 PaddlePaddle GPU 설치

#### 일반 GPU (CUDA 11.8/12.x)

```bash
# PaddlePaddle GPU 설치
pip install paddlepaddle-gpu==3.2.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

#### RTX 50 시리즈 GPU (특수 빌드)

RTX 50 시리즈 GPU의 경우 별도의 빌드된 패키지가 필요합니다.

```bash
# 설치 스크립트 실행 (권장)
install_rtx50_paddle.bat

# 또는 수동 설치
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp312-cp312-win_amd64.whl
```

### 1.5 필수 패키지 설치

```bash
# PaddleOCR 필수 패키지
cd ../PaddleOCR
pip install -r requirements.txt

# 프로젝트 필수 패키지
cd ../yuzyproject-aimodels
pip install -r requirements.txt
```

### 1.6 GPU 인식 확인

```bash
python -c "import paddle; paddle.utils.run_check()"
```

출력 예시:
```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 GPU.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

---

## 2. 데이터셋 생성

### 2.1 데이터셋 생성 스크립트

데이터셋 생성은 `prepare_code_syntax_dataset.py` 스크립트를 사용합니다.

이 스크립트는 다음 작업을 수행합니다:
- 코드 문법 이미지 생성 (25,000개 학습용, 5,000개 검증용)
- 다양한 폰트, 테마, 크기로 이미지 변형
- 레이블 파일 생성 (train_list.txt, val_list.txt)

### 2.2 데이터셋 생성 코드 상세

#### 주요 구성 요소

**1. 코드 샘플 (`CODE_SAMPLES`)**
- Python 문법 패턴 150개 이상 포함
- 함수 정의, 클래스, 반복문, 조건문, 예외 처리 등
- 한글 포함 코드 샘플

**2. 이미지 생성 함수 (`generate_code_image`)**
```python
def generate_code_image(code_text: str, font_name: str = "Consolas", 
                       font_size: int = 14, theme: str = "dark",
                       width: int = 2000, padding: int = 50):
    """
    코드 텍스트를 이미지로 변환
    - 텍스트 잘림 방지
    - 자동 크롭
    - 최종 크기: 48x640px (PaddleOCR 요구사항)
    """
```

**3. 데이터셋 생성 함수 (`create_dataset`)**
```python
def create_dataset(num_train: int = 25000, num_val: int = 5000, 
                  verify_with_ocr: bool = False):
    """
    데이터셋 생성
    - 학습 데이터: 25,000개
    - 검증 데이터: 5,000개
    - OCR 검증: 기본적으로 비활성화 (너무 느림)
    """
```

### 2.3 데이터셋 생성 실행

#### 방법 1: Python 스크립트 직접 실행

```bash
# yuzyproject-aimodels 디렉토리에서 실행
cd yuzyproject-aimodels
python prepare_code_syntax_dataset.py
```

#### 방법 2: 배치 파일 사용 (기존 데이터셋 백업 포함)

```bash
regenerate_dataset_fixed.bat
```

### 2.4 데이터셋 생성 시간

- **예상 소요 시간**: 10-30분 (시스템 성능에 따라 다름)
- **생성되는 파일**:
  - `code_syntax_dataset/train_images/` (25,000개 JPG 파일)
  - `code_syntax_dataset/val_images/` (5,000개 JPG 파일)
  - `code_syntax_dataset/train_list.txt` (레이블 파일)
  - `code_syntax_dataset/val_list.txt` (레이블 파일)

### 2.5 데이터셋 구조

```
code_syntax_dataset/
├── train_images/
│   ├── train_00000.jpg
│   ├── train_00001.jpg
│   └── ... (25,000개)
├── val_images/
│   ├── val_00000.jpg
│   ├── val_00001.jpg
│   └── ... (5,000개)
├── train_list.txt
└── val_list.txt
```

#### 레이블 파일 형식 (`train_list.txt`, `val_list.txt`)

```
train_images/train_00000.jpg	def get_paddleocr_words(
train_images/train_00001.jpg	img: Union[np.ndarray, Image.Image],
train_images/train_00002.jpg	scale: int = 3,
...
```

각 줄은 `이미지_경로\t레이블_텍스트` 형식입니다.

### 2.6 데이터셋 생성 스크립트 전체 코드

전체 코드는 `prepare_code_syntax_dataset.py` 파일을 참조하세요. 주요 기능:

- **폰트 지원**: Consolas, Courier New, Lucida Console, MS Gothic
- **테마 지원**: Dark, Light
- **이미지 크기**: 최종 48x640px (PaddleOCR 요구사항)
- **텍스트 처리**: 
  - 텍스트 잘림 방지
  - 자동 크롭
  - 비율 유지 리사이징

---

## 3. 학습 시작

### 3.1 학습 설정 파일 확인

학습 설정은 `PaddleOCR/configs/rec/rec_code_syntax_finetune.yml` 파일에 정의되어 있습니다.

#### 주요 설정 항목

```yaml
Global:
  use_gpu: true
  epoch_num: 100
  use_amp: true  # Mixed Precision Training
  pretrained_model: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv5_mobile_rec_pretrained.pdparams

Optimizer:
  name: Adam
  lr:
    learning_rate: 0.0005
    name: Cosine
    warmup_epoch: 5

Train:
  loader:
    batch_size_per_card: 32
    num_workers: 6
    pin_memory: true

Eval:
  loader:
    batch_size_per_card: 32
    num_workers: 6
```

### 3.2 학습 실행

#### 방법 1: 배치 파일 사용 (권장)

```bash
# yuzyproject-aimodels 디렉토리에서 실행
start_training_clean.bat
```

이 배치 파일은 다음 작업을 수행합니다:
1. PaddleOCR 디렉토리로 이동
2. 가상환경 활성화
3. 학습 시작

#### 방법 2: 명령줄 직접 실행

```bash
cd ../PaddleOCR
python tools/train.py -c configs/rec/rec_code_syntax_finetune.yml \
  -o Global.use_gpu=true \
  Global.use_amp=true \
  Global.amp_level=O1 \
  Global.epoch_num=100 \
  Global.print_batch_step=50 \
  Train.loader.batch_size_per_card=32 \
  Train.loader.num_workers=6 \
  Train.loader.pin_memory=true \
  Eval.loader.batch_size_per_card=32 \
  Eval.loader.num_workers=6 \
  Optimizer.lr.learning_rate=0.0005
```

### 3.3 학습 모니터링

#### 로그 파일 확인

```bash
# 실시간 로그 확인 (Windows PowerShell)
Get-Content PaddleOCR\output\rec\code_syntax\train.log -Wait -Tail 50

# 또는 배치 파일 사용
run_monitoring.bat
```

#### 주요 메트릭

학습 중 다음 메트릭을 확인하세요:

- **NRTRLoss**: 손실 값 (NaN 발생 시 학습 중단 필요)
- **norm_edit_dis**: 정규화된 편집 거리 (낮을수록 좋음)
- **acc**: 정확도

#### 출력 예시

```
[2025-01-XX XX:XX:XX] train step: 50, epoch: 1, batch: 50, loss: 2.345, norm_edit_dis: 0.123, acc: 0.789
[2025-01-XX XX:XX:XX] eval step: 200, epoch: 1, batch: 200, loss: 2.123, norm_edit_dis: 0.098, acc: 0.856
```

### 3.4 체크포인트 관리

#### 저장 위치

- **최고 성능 모델**: `PaddleOCR/output/rec/code_syntax/best_accuracy.pdparams`
- **최신 모델**: `PaddleOCR/output/rec/code_syntax/latest.pdparams`

#### 학습 재개

이전 체크포인트에서 학습을 재개하려면:

```bash
python tools/train.py -c configs/rec/rec_code_syntax_finetune.yml \
  -o Global.checkpoints=./output/rec/code_syntax/latest
```

### 3.5 학습 완료 후

학습이 완료되면 다음 파일들이 생성됩니다:

```
PaddleOCR/output/rec/code_syntax/
├── best_accuracy.pdparams      # 최고 성능 모델
├── latest.pdparams              # 최신 모델
├── train.log                    # 학습 로그
└── inference/                   # 추론 모델 (export 시)
```

---

## 4. 문제 해결

### 4.1 GPU 인식 문제

**증상**: GPU를 인식하지 못함

**해결 방법**:
```bash
# GPU 확인
python -c "import paddle; print(paddle.device.get_device())"

# CUDA 확인
python -c "import paddle; print(paddle.is_compiled_with_cuda())"

# GPU 개수 확인
python -c "import paddle; print(paddle.device.cuda.device_count())"
```

### 4.2 메모리 부족 (OOM) 문제

**증상**: `Out of Memory` 에러 발생

**해결 방법**:

1. 배치 사이즈 줄이기:
   ```yaml
   Train:
     loader:
       batch_size_per_card: 16  # 32 → 16
   ```

2. num_workers 줄이기:
   ```yaml
   Train:
     loader:
       num_workers: 4  # 6 → 4
   ```

3. Mixed Precision 활성화 (이미 활성화되어 있음):
   ```yaml
   Global:
     use_amp: true
     amp_level: O1
   ```

### 4.3 NaN Loss 발생

**증상**: 학습 중 loss가 NaN이 됨

**해결 방법**:

1. 학습 중단 (Ctrl+C)
2. NaN 발생 이전 체크포인트 확인
3. Learning rate 조정:
   ```yaml
   Optimizer:
     lr:
       learning_rate: 0.0002  # 0.0005 → 0.0002
   ```
4. 해당 체크포인트부터 재개:
   ```bash
   python tools/train.py -c configs/rec/rec_code_syntax_finetune.yml \
     -o Global.checkpoints=./output/rec/code_syntax/best_accuracy \
     Optimizer.lr.learning_rate=0.0002
   ```

### 4.4 데이터셋 경로 문제

**증상**: 데이터셋을 찾을 수 없음

**해결 방법**:

1. 데이터셋 경로 확인:
   ```yaml
   Train:
     dataset:
       data_dir: ../yuzyproject-aimodels/code_syntax_dataset/
       label_file_list: ["../yuzyproject-aimodels/code_syntax_dataset/train_list.txt"]
   ```

2. 상대 경로가 맞는지 확인 (PaddleOCR 디렉토리에서 실행해야 함)

### 4.5 데이터셋 생성 실패

**증상**: `prepare_code_syntax_dataset.py` 실행 시 오류

**해결 방법**:

1. 필수 패키지 설치 확인:
   ```bash
   pip install Pillow numpy
   ```

2. 폰트 경로 확인 (Windows):
   - `C:/Windows/Fonts/Consolas.ttf`
   - `C:/Windows/Fonts/Courier New.ttf`

3. 디스크 공간 확인 (데이터셋은 약 1-2GB 필요)

### 4.6 PaddleOCR 설치 문제

**증상**: `ModuleNotFoundError: No module named 'paddleocr'`

**해결 방법**:

```bash
pip install paddleocr>=2.9.0
```

### 4.7 Windows 특화 문제

#### 문제: Shared Memory 오류

**해결 방법**:
```yaml
Train:
  loader:
    use_shared_memory: False  # Windows에서는 False 권장
```

#### 문제: 경로 구분자 문제

**해결 방법**: 설정 파일에서 경로는 슬래시(`/`) 사용 권장

---

## 5. 참고 자료

### 5.1 관련 파일

- **데이터셋 생성**: `yuzyproject-aimodels/prepare_code_syntax_dataset.py`
- **학습 설정**: `PaddleOCR/configs/rec/rec_code_syntax_finetune.yml`
- **학습 스크립트**: `PaddleOCR/tools/train.py`
- **학습 실행 배치**: `yuzyproject-aimodels/start_training_clean.bat`

### 5.2 주요 디렉토리

```
semi/
├── PaddleOCR/
│   ├── configs/rec/rec_code_syntax_finetune.yml
│   ├── tools/train.py
│   └── output/rec/code_syntax/  # 학습 결과
│
└── yuzyproject-aimodels/
    ├── prepare_code_syntax_dataset.py
    ├── code_syntax_dataset/  # 생성된 데이터셋 (Git 제외)
    └── venv_ocr/  # 가상환경 (Git 제외)
```

### 5.3 Git 무시 항목

다음 항목들은 `.gitignore`에 포함되어 Git에 업로드되지 않습니다:

- `venv_ocr/` (가상환경)
- `code_syntax_dataset/` (데이터셋)
- `output_root/` (모델 파일)
- `outputs/` (임시 출력)
- `*.pdparams` (모델 체크포인트)
- `PaddleOCR/output/` (학습 결과)

---

## 6. 요약: 빠른 시작 체크리스트

- [ ] 저장소 클론 및 브랜치 전환
- [ ] 가상환경 생성 및 활성화
- [ ] PaddlePaddle GPU 설치
- [ ] 필수 패키지 설치
- [ ] GPU 인식 확인
- [ ] 데이터셋 생성 (25,000 + 5,000개)
- [ ] 학습 설정 파일 확인
- [ ] 학습 시작
- [ ] 학습 모니터링
- [ ] 학습 완료 후 모델 확인

---

## 문의 및 지원

문제가 발생하면 다음을 확인하세요:

1. 이 가이드의 [문제 해결](#4-문제-해결) 섹션
2. `PaddleOCR` 공식 문서
3. 학습 로그 파일 (`train.log`)

