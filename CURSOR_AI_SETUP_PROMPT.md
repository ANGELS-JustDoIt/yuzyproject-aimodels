# RTX 4070Ti 데스크탑 PaddleOCR 학습 환경 설정 프롬프트

이 프롬프트를 Cursor AI에게 전달하여 RTX 4070Ti 데스크탑에서 PaddleOCR 코드 문법 인식 모델 학습 환경을 구성하세요.

---

## 프로젝트 구조 파악 및 환경 설정

나는 RTX 4070Ti GPU를 사용하는 Windows 데스크탑에서 PaddleOCR v5를 사용한 코드 문법 인식 모델 학습 환경을 구성하려고 합니다.

현재 상황:
- `gonida` 브랜치에서 `git pull`을 받았습니다.
- `yuzyproject-aimodels/` 폴더에 모든 AI 관련 파일이 있습니다.
- `PaddleOCR/`은 서브모듈로 포함되어 있지 않습니다 (새로 클론해야 합니다).
- Python 3.8 이상이 설치되어 있습니다.
- CUDA 11.8 이상이 설치되어 있습니다.

다음 작업을 수행해주세요:

### 1단계: 프로젝트 구조 확인

먼저 현재 디렉토리 구조를 확인하고, 다음 파일들이 있는지 확인해주세요:
- `yuzyproject-aimodels/prepare_code_syntax_dataset.py` (데이터셋 생성 스크립트)
- `yuzyproject-aimodels/requirements.txt` (Python 패키지 목록)
- `yuzyproject-aimodels/README_OCR_TRAINING.md` (학습 가이드)
- `.gitignore` (루트에 있음)

### 2단계: PaddleOCR 클론

PaddleOCR이 없으므로 공식 저장소에서 클론해주세요:
```bash
cd yuzyproject-aimodels
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd ..
```

### 3단계: 가상환경 생성 및 활성화

`yuzyproject-aimodels` 디렉토리에 가상환경을 생성하고 활성화해주세요:
```bash
cd yuzyproject-aimodels
python -m venv venv_ocr
venv_ocr\Scripts\activate
```

### 4단계: PaddlePaddle GPU 설치

CUDA 11.8 기준으로 PaddlePaddle GPU 버전을 설치해주세요:
```bash
pip install paddlepaddle-gpu==3.2.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

**참고:** Windows의 경우 공식 사이트 확인: https://www.paddlepaddle.org.cn/install/quick

### 5단계: 필수 패키지 설치

다음 순서로 패키지를 설치해주세요:
```bash
# PaddleOCR 의존성
cd ../PaddleOCR
pip install -r requirements.txt

# yuzyproject-aimodels 의존성
cd ../yuzyproject-aimodels
pip install -r requirements.txt
```

### 6단계: GPU 환경 확인

GPU가 정상적으로 인식되는지 확인해주세요:
```bash
python -c "import paddle; paddle.utils.run_check()"
```

### 7단계: 학습 설정 파일 확인

`PaddleOCR/configs/rec/rec_code_syntax_finetune.yml` 파일이 있는지 확인하고, 없다면 생성해주세요. 이 파일은 RTX 4070Ti에 최적화된 설정이 포함되어야 합니다:

**주요 설정:**
- `batch_size_per_card: 32` (Train/Eval 모두)
- `num_workers: 6`
- `pin_memory: true`
- `clip_norm: 5.0` (Gradient Clipping, NaN 방지)
- `learning_rate: 0.0005`
- `lr.name: Cosine`
- `warmup_epoch: 5`
- NRTR 전용 학습 (CTCLoss 사용하지 않음)

### 8단계: 데이터셋 생성

데이터셋이 없다면 생성해주세요:
```bash
cd yuzyproject-aimodels
python prepare_code_syntax_dataset.py
```

**예상 소요 시간:** 10-30분

### 9단계: 학습 시작

모든 준비가 완료되면 학습을 시작해주세요:
```bash
cd ../PaddleOCR
python tools/train.py -c configs/rec/rec_code_syntax_finetune.yml
```

---

## 문제 해결

### GPU 인식 안 될 때
- CUDA 버전 확인: `nvidia-smi`
- PaddlePaddle 재설치 시도

### NaN Loss 발생 시
- 학습 중단 후 이전 체크포인트부터 재개
- `clip_norm: 5.0` 설정 확인

### 메모리 부족 시
- `batch_size_per_card`를 32에서 16으로 줄이기

---

## 참고 파일

- 학습 가이드: `yuzyproject-aimodels/README_OCR_TRAINING.md`
- 상세 설정: `PaddleOCR/TRAINING_SETUP.md` (있을 경우)

위 단계들을 순서대로 수행하고, 각 단계마다 결과를 확인해주세요. 문제가 발생하면 알려주세요.

