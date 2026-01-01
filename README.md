# AI Models 프로젝트

FastAPI + PyTorch + Transformers 기반 코드 분석/시각화 서버 및 PaddleOCR 코드 문법 인식 모델 학습 프로젝트입니다.

## 빠른 시작

### 기본 서버 실행

자세한 내용은 [README.md](./README.md)의 환경 설정 섹션을 참조하세요.

### OCR 모델 학습

**데이터셋 생성부터 학습 시작까지의 전체 가이드**: [SETUP_GUIDE.md](./SETUP_GUIDE.md)를 참조하세요.

빠른 요약:

1. **환경 설정**
   ```bash
   python -m venv venv_ocr
   venv_ocr\Scripts\activate
   pip install paddlepaddle-gpu==3.2.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
   cd ../PaddleOCR && pip install -r requirements.txt
   cd ../yuzyproject-aimodels && pip install -r requirements.txt
   ```

2. **데이터셋 생성**
   ```bash
   python prepare_code_syntax_dataset.py
   ```

3. **학습 시작**
   ```bash
   start_training_clean.bat
   ```

## 프로젝트 구조

```
yuzyproject-aimodels/
├── core/                      # 핵심 모듈
│   ├── analyzer.py            # 코드 분석 (LLM 기반)
│   └── capture.py             # OCR 캡처 기능
├── prepare_code_syntax_dataset.py  # 데이터셋 생성 스크립트
├── SETUP_GUIDE.md            # 상세 설정 가이드 (데이터셋 생성 + 학습)
├── README_OCR_TRAINING.md    # OCR 학습 빠른 시작 가이드
└── requirements.txt          # Python 패키지 목록
```

## 주요 기능

### 1. 코드 분석 (LLM 기반)

- FastAPI 서버를 통한 코드 분석 API
- Qwen2.5-Coder 모델 사용
- 프로젝트 전체 코드 구조 분석 및 시각화

### 2. OCR 코드 문법 인식

- PaddleOCR 기반 코드 이미지 인식
- 코드 문법 특화 데이터셋으로 학습
- 한글 + 영어 + 코드 기호 인식

## 환경 설정

이 프로젝트는 **Windows 환경**에서 개발되었습니다.

- **Python**: 3.12.10
- **가상환경**: `venv_ocr` (OCR 학습용)

자세한 환경 설정 방법은 [README.md](./README.md)를 참조하세요.

## 데이터셋 및 학습

- **데이터셋 생성**: `prepare_code_syntax_dataset.py`
- **학습 설정**: `PaddleOCR/configs/rec/rec_code_syntax_finetune.yml`
- **학습 실행**: `start_training_clean.bat`

**상세 가이드**: [SETUP_GUIDE.md](./SETUP_GUIDE.md) 참조

## Git 무시 항목

다음 항목들은 Git에 업로드되지 않습니다 (`.gitignore` 참조):

- `venv_ocr/` - 가상환경
- `code_syntax_dataset/` - 데이터셋 (대용량)
- `output_root/` - 학습된 모델 파일
- `outputs/` - 임시 출력 파일
- `*.pdparams` - PaddleOCR 체크포인트

## 참고 문서

- [SETUP_GUIDE.md](./SETUP_GUIDE.md) - 데이터셋 생성 및 학습 전체 가이드
- [README_OCR_TRAINING.md](./README_OCR_TRAINING.md) - OCR 학습 빠른 시작

## 라이센스

이 프로젝트의 라이센스 정보는 저장소 루트의 LICENSE 파일을 참조하세요.
