# project_root/main.py
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from core.analyzer import (
    AnalyzerConfig,
    load_model_once,
    analyze_from_text,
    explain_from_analysis,
    INPUT_FILE,
    OUTPUT_JSON,
)
from core.capture import capture_and_ocr, check_paddleocr_available

app = FastAPI(title="TaskFlow Analyzer", version="1.0.0")

# CORS 설정: 프론트엔드(Next.js)에서 호출할 수 있도록 허용
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
def startup():
    # LLM 모델은 서버 시작 시 1회 로딩
    load_model_once()
    
    # PaddleOCR 초기화 (서버 시작 시 미리 로딩 - v5 학습된 모델 우선 사용)
    try:
        paddleocr_available, error = check_paddleocr_available()
        if paddleocr_available:
            # v5 학습된 모델이 사용되는지 확인
            from core.capture import _PADDLEOCR_TRAINED_INSTANCE, _check_trained_model_available
            if _check_trained_model_available() and _PADDLEOCR_TRAINED_INSTANCE is not None:
                print(f"[OK] PaddleOCR 준비 완료 (v5 학습된 모델 사용, 코드 문법 인식 최적화, 88.96% 정확도)")
            else:
                print(f"[OK] PaddleOCR 준비 완료 (기본 v5 모델 사용, PP-OCRv5, 한글+영어 최적화)")
        else:
            print(f"[WARN] PaddleOCR 사용 불가: {error}")
    except Exception as e:
        print(f"[WARN] PaddleOCR 초기화 경고: {e}")


@app.get("/health")
def health():
    return {"ok": True}


def _read_text_file_safe(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


class VisualizeRequest(BaseModel):
    # 프론트엔드에서 보내는 통합 코드 텍스트
    code: str


@app.post("/analyze")
async def analyze(
    # 1) 통합 텍스처 파일 업로드 (권장)
    context_file: Optional[UploadFile] = File(default=None),
    # 2) 텍스트 직접 전달 (테스트용)
    context_text: Optional[str] = Form(default=None),
):
    if context_file is None and (context_text is None or not context_text.strip()):
        raise HTTPException(
            status_code=400,
            detail="context_file 또는 context_text 둘 중 하나는 필요합니다."
        )

    request_id = str(uuid.uuid4())
    out_dir = OUTPUTS_DIR / request_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 입력 텍스트 확보 ----
    if context_file is not None:
        raw = await context_file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="ignore")
    else:
        text = context_text

    cfg = AnalyzerConfig(
        max_total_lines=2000,
        max_total_chars=50_000,
        max_new_tokens=16384,  # 모든 API 엔드포인트를 포함하기 위해 충분히 큰 값
        repetition_penalty=1.1,
    )

    # ---- Pass A / Pass B ----
    analysis = analyze_from_text(text, out_dir=str(out_dir), cfg=cfg)
    explain = explain_from_analysis(analysis, out_dir=str(out_dir))

    return JSONResponse(content={
        "requestId": request_id,
        "analysis": analysis,
        "explain": explain,
    })


@app.post("/visualize")
async def visualize(req: VisualizeRequest):
    """
    Next.js 프론트엔드에서 사용하는 엔드포인트.
    - URL: POST /visualize
    - Body(JSON): { "code": "<프로젝트 전체 코드 텍스트>" }

    반환 형식은 프론트에서 기대하는 것처럼 최상단에 `api` 배열이 오는 JSON입니다.
    """
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="code 필드는 비어 있을 수 없습니다.")

    # outputs 디렉터리 하위에 요청별 결과 저장 (analyze와 동일한 구조)
    request_id = str(uuid.uuid4())
    out_dir = OUTPUTS_DIR / request_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AnalyzerConfig(
        max_total_lines=2000,
        max_total_chars=50_000,
        max_new_tokens=16384,  # 모든 API 엔드포인트를 포함하기 위해 충분히 큰 값
        repetition_penalty=1.1,
    )

    analysis = analyze_from_text(req.code, out_dir=str(out_dir), cfg=cfg)

    # 프론트는 analysisResult.api 를 기대하므로,
    # LLM이 만든 JSON을 그대로 반환 (이미 "api" 필드를 포함하고 있어야 함)
    return JSONResponse(content=analysis)


@app.get("/result/{request_id}/code", response_class=PlainTextResponse)
def get_code(request_id: str):
    p = OUTPUTS_DIR / request_id / INPUT_FILE
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{p} not found")
    return _read_text_file_safe(p)


@app.get("/result/{request_id}/analysis")
def get_analysis(request_id: str):
    p = OUTPUTS_DIR / request_id / OUTPUT_JSON
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{p} not found")
    return JSONResponse(content=json_load_safe(_read_text_file_safe(p)))


@app.get("/result/{request_id}/explain")
def get_explain(request_id: str):
    p = OUTPUTS_DIR / request_id / "project_explain.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{p} not found")
    return JSONResponse(content=json_load_safe(_read_text_file_safe(p)))


@app.post("/capture")
def capture():
    """
    화면 캡처 -> 드래그로 영역 선택 -> OCR 인식 -> 클립보드 저장
    
    Returns:
        JSONResponse: {
            "success": bool,
            "text": str (OCR 결과),
            "method": str (사용된 OCR 방법),
            "error": str (에러 메시지, 실패 시)
        }
    """
    try:
        result = capture_and_ocr()
        
        # 방어적 코드: None이나 예상치 못한 구조 처리
        if result is None:
            result = {
                "success": False,
                "error": "OCR 처리 중 예상치 못한 오류가 발생했습니다.",
                "text": "",
                "method": ""
            }
        
        # 필수 필드 확인 및 기본값 설정
        if not isinstance(result, dict):
            result = {
                "success": False,
                "error": "OCR 결과 형식이 올바르지 않습니다.",
                "text": "",
                "method": ""
            }
        else:
            # 필수 필드가 없으면 기본값 설정
            if "success" not in result:
                result["success"] = False
            if "text" not in result:
                result["text"] = ""
            if "method" not in result:
                result["method"] = ""
            if "error" not in result:
                result["error"] = None
            
            # text가 None이면 빈 문자열로 변환
            if result.get("text") is None:
                result["text"] = ""
            else:
                result["text"] = str(result["text"])
        
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        error_msg = f"서버 오류: {str(e)}"
        print(f"❌ /capture 엔드포인트 오류: {error_msg}")
        print(f"   상세 오류: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": error_msg,
                "text": "",
                "method": ""
            }
        )


def json_load_safe(text: str):
    import json
    return json.loads(text)
