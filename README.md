# AI Models Service

FastAPI 기반의 코드 분석 및 시각화 AI 서비스입니다. PyTorch와 Transformers를 활용하여 코드를 분석하고, OCR 기능을 통해 화면 캡처 및 텍스트 추출을 지원합니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [환경 요구사항](#환경-요구사항)
- [설치 및 설정](#설치-및-설정)
- [실행 방법](#실행-방법)
- [API 엔드포인트](#api-엔드포인트)
- [프로젝트 구조](#프로젝트-구조)
- [주요 의존성](#주요-의존성)

## 🚀 주요 기능

- **코드 분석**: 프로젝트 전체 코드를 분석하여 API 엔드포인트 및 플로우 추출
- **코드 시각화**: 분석 결과를 JSON 형식으로 반환하여 프론트엔드에서 시각화 가능
- **OCR 기능**: Windows 화면 캡처 및 이미지에서 텍스트 추출
  - Tesseract OCR 지원
  - Windows WinRT OCR 지원
  - 코드 모드 지원 (들여쓰기, 포맷팅 자동 처리)

## 🛠 기술 스택

- **Framework**: FastAPI
- **AI/ML**: PyTorch, Transformers, Hugging Face
- **Model**: Qwen2.5-Coder-1.5B-Instruct
- **OCR**: Tesseract, Windows WinRT
- **Server**: Uvicorn

## 📦 환경 요구사항

- **Python**: 3.12.10
- **OS**: Windows (OCR 기능 사용 시)
- **GPU**: CUDA 지원 GPU (선택사항, 더 빠른 추론을 위해 권장)

## 🔧 설치 및 설정

### 1. 가상환경 생성 및 활성화

```bash
cd yuzyproject-aimodels
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. Windows OCR 기능 사용 시 (선택사항)

화면 캡처 후 드래그 영역을 OCR 하는 기능을 사용하려면 추가 패키지 설치:

```bash
pip install -U winrt-runtime
pip install -U winrt-Windows.Foundation winrt-Windows.Foundation.Collections
pip install -U winrt-Windows.Media.Ocr winrt-Windows.Globalization winrt-Windows.Graphics.Imaging winrt-Windows.Storage.Streams
```

### 4. 환경 변수 설정 (선택사항)

`.env` 파일을 생성하여 다음 변수를 설정할 수 있습니다:

```env
FRONTEND_URL=http://localhost:3000,http://127.0.0.1:3000
NODE_ENV=development
```

## ▶️ 실행 방법

### 개발 모드 실행

```bash
# 가상환경 활성화 후
uvicorn main:app --reload

# 또는
python -m uvicorn main:app --reload
```

서버가 실행되면 기본적으로 `http://localhost:8000`에서 접근 가능합니다.

### 서버 상태 확인

브라우저에서 `http://localhost:8000/health` 접속 → `{ "ok": true }` 응답이 오면 정상입니다.

### API 문서 확인

FastAPI 자동 생성 문서:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📡 API 엔드포인트

### Health Check

```
GET /health
```

서버 상태 확인

**Response:**

```json
{
  "ok": true
}
```

### 코드 분석

```
POST /analyze
```

프로젝트 코드를 분석하여 API 엔드포인트 및 플로우를 추출합니다.

**Request:**

- `context_file` (file, optional): 코드가 포함된 텍스트 파일 업로드
- `context_text` (form-data, optional): 코드 텍스트 직접 전달

**Response:**

```json
{
  "requestId": "uuid",
  "analysis": {
    "api": [...]
  },
  "explain": {...}
}
```

### 코드 시각화

```
POST /visualize
```

프론트엔드에서 사용하는 엔드포인트로, 코드를 분석하여 시각화용 JSON을 반환합니다.

**Request Body (JSON):**

```json
{
  "code": "<프로젝트 전체 코드 텍스트>"
}
```

**Response:**

```json
{
  "api": [
    {
      "method": "POST",
      "path": "/api/endpoint",
      "description": "...",
      "code": "..."
    }
  ]
}
```

### 화면 캡처 및 OCR

```
POST /capture
```

이미지 파일을 업로드하거나 Windows 화면 캡처를 통해 OCR을 수행합니다.

**Request (multipart/form-data):**

- `file` (file, optional): 이미지 파일 업로드 (없으면 화면 캡처 모드)
- `lang` (string, default: "kor+eng"): OCR 언어
- `scale` (int, default: 3): 이미지 스케일
- `code_mode` (bool, default: true): 코드 모드 활성화
- `layout` (bool, default: true): 레이아웃 모드 활성화
- `normalize` (bool, default: true): 정규화 활성화

**Response:**

```json
{
  "success": true,
  "text": "추출된 텍스트",
  "method": "Tesseract + WinRT 병합",
  "error": null
}
```

### 분석 결과 조회

```
GET /result/{request_id}/code
GET /result/{request_id}/analysis
GET /result/{request_id}/explain
```

이전 분석 요청의 결과를 조회합니다.

## 📁 프로젝트 구조

```
yuzyproject-aimodels/
├── main.py                 # FastAPI 애플리케이션 진입점
├── requirements.txt        # Python 의존성 목록
├── core/                   # 핵심 모듈
│   ├── analyzer.py        # 코드 분석 로직
│   └── capture.py         # OCR 및 캡처 기능
├── aimodels/              # 분석 결과 저장 디렉토리
├── outputs/               # 요청별 출력 파일 저장 디렉토리
│   └── {request_id}/      # 각 요청별 결과 파일
│       ├── project_full_context.txt
│       ├── project_flows.json
│       └── project_explain.json
└── venv/                  # 가상환경 (gitignore)
```

## 📚 주요 의존성

### AI/ML 관련

- `torch`: PyTorch 딥러닝 프레임워크
- `torchvision`, `torchaudio`: PyTorch 확장 패키지
- `transformers`: Hugging Face Transformers 라이브러리
- `accelerate`: 모델 가속화
- `safetensors`: 안전한 텐서 저장 형식

### 웹 서버

- `fastapi`: 고성능 웹 프레임워크
- `uvicorn[standard]`: ASGI 서버
- `python-multipart`: 파일 업로드 지원

### 클라우드/LLM 연동

- `openai`: OpenAI API 클라이언트
- `google-generativeai`: Google Gemini API 클라이언트
- `google-api-python-client`: Google API 클라이언트

### 기타

- `pillow`: 이미지 처리
- `numpy`: 수치 연산

## 🔍 주요 기능 상세

### 코드 분석 프로세스

1. **입력 처리**: 코드 텍스트 또는 파일을 받아 전처리
2. **모델 로딩**: Qwen2.5-Coder 모델을 서버 시작 시 한 번 로딩
3. **분석 수행**:
   - Pass A: 코드에서 API 엔드포인트 추출
   - Pass B: 추출된 정보를 바탕으로 설명 생성
4. **결과 저장**: `outputs/{request_id}/` 디렉토리에 결과 파일 저장
5. **JSON 반환**: 프론트엔드에서 사용할 수 있는 형식으로 반환

### OCR 기능

- **Tesseract OCR**: 오픈소스 OCR 엔진
- **Windows WinRT OCR**: Windows 네이티브 OCR API
- **병합 처리**: 두 OCR 결과를 병합하여 정확도 향상
- **코드 모드**: 코드 텍스트의 들여쓰기와 포맷팅을 자동으로 처리

## ⚠️ 주의사항

- 모델 파일은 첫 요청 시 자동으로 다운로드되므로 초기 실행 시 시간이 걸릴 수 있습니다.
- GPU가 없는 환경에서도 실행 가능하지만, CPU 모드로 실행되므로 속도가 느릴 수 있습니다.
- Windows OCR 기능은 Windows 환경에서만 사용 가능합니다.
- 대용량 코드 분석 시 메모리 사용량이 증가할 수 있습니다.

## 🐛 문제 해결

### 모델 다운로드 실패

- 인터넷 연결 확인
- Hugging Face 토큰 설정 (필요 시)

### OCR 기능 오류

- Tesseract 설치 확인
- Windows 환경 확인 (WinRT 사용 시)

### 메모리 부족

- `max_total_lines`, `max_total_chars` 설정 조정
- 배치 크기 감소

## 📝 라이선스

ISC
