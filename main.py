# project_root/main.py
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import PlainTextResponse, JSONResponse

from core.analyzer import (
    AnalyzerConfig,
    load_model_once,
    analyze_from_text,
    explain_from_analysis,
    INPUT_FILE,
    OUTPUT_JSON,
)

app = FastAPI(title="TaskFlow Analyzer", version="1.0.0")

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
        max_time_seconds=200,
        max_new_tokens=4096,
        repetition_penalty=1.1
    )

    # ---- Pass A / Pass B ----
    analysis = analyze_from_text(text, out_dir=str(out_dir), cfg=cfg)
    explain = explain_from_analysis(analysis, out_dir=str(out_dir))

    return JSONResponse(content={
        "requestId": request_id,
        "analysis": analysis,
        "explain": explain,
    })


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

def json_load_safe(text: str):
    import json
    return json.loads(text)
