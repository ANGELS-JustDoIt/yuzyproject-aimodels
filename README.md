## 환경설정

이 프로젝트는 **FastAPI + PyTorch + Transformers 기반 코드 분석/시각화 서버**입니다.  
Windows 환경에서 아래 순서대로 준비해 주세요.

- **Python 버전**

  - Python 3.12.10

- **가상환경 생성 및 활성화 (예시)**

```bash
cd yuzyproject-aimodels
python -m venv venv
venv\Scripts
activate
```

- **필수 파이썬 패키지 설치 (requirements 전체 설치)**  
  `requirements.txt`에 이 프로젝트에 필요한 의존성이 모두 정의되어 있습니다.

```bash
pip install -r requirements.txt
```

- **주요 의존성 (개별 설치가 필요할 때 참고용)**
  - **모델 관련**: `torch`, `torchvision`, `torchaudio`, `transformers`, `accelerate`, `safetensors`
  - **웹 서버**: `fastapi`, `uvicorn[standard]`, `python-multipart`
  - **클라우드/LLM 연동**: `openai`, `google-generativeai`, `google-api-python-client`

개별 설치가 필요하다면 예를 들어 아래처럼 설치할 수 있습니다.

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate safetensors
pip install fastapi "uvicorn[standard]" python-multipart
pip install openai google-generativeai google-api-python-client
```

- **Windows OCR(캡처 기능) 관련 패키지**  
  화면 캡처 후 드래그 영역을 OCR 하는 기능을 사용하려면 아래 winrt 패키지들도 추가로 설치해야 합니다.

```bash
pip install -U winrt-runtime
pip install -U winrt-Windows.Foundation winrt-Windows.Foundation.Collections
pip install -U winrt-Windows.Media.Ocr winrt-Windows.Globalization winrt-Windows.Graphics.Imaging winrt-Windows.Storage.Streams
```

## 실행 방법

FastAPI 서버를 띄워서 프론트엔드(Next.js)에서 호출하거나, 직접 API를 호출해서 사용할 수 있습니다.

1. **가상환경 활성화**

```bash
cd yuzyproject-aimodels
venv\Scripts
activate
```

2. **서버 실행 (개발용)**

```bash
uvicorn main:app --reload

# 에러나는 경우는 이렇게 실행하세요
python -m uvicorn main:app --reload
```

3. **기본 동작 확인**

   - 브라우저에서 `http://localhost:8000/health` 접속 → `{ "ok": true }` 응답이 오면 정상입니다.

4. **주요 엔드포인트**
   - **`POST /analyze`**
     - 코드 분석
   - **`POST /visualize`**
     - JSON Body: `{ "code": "<프로젝트 전체 코드 텍스트>" }`
     - 분석 결과 JSON 반환 (프론트엔드 시각화용)
   - **`POST /capture`**
     - Windows 화면에서 드래그로 선택한 영역을 캡처 후 OCR 수행, 결과 텍스트를 JSON으로 반환
