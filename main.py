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
from core.capture import capture_and_ocr

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
    # 모델은 서버 시작 시 1회 로딩
    load_model_once()


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
    result = capture_and_ocr()
    return JSONResponse(content=result)


def json_load_safe(text: str):
    import json
    return json.loads(text)
