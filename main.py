# project_root/main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from core.analyzer import (
    INPUT_FILE,
    OUTPUT_JSON,
    AnalyzerConfig,
    load_model_once,
    analyze_to_json,
)

app = FastAPI(title="TaskFlow Analyzer", version="1.0.0")


@app.on_event("startup")
def startup():
    load_model_once()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/code", response_class=PlainTextResponse)
def get_code():
    if not os.path.exists(INPUT_FILE):
        raise HTTPException(
            status_code=404,
            detail=f"{INPUT_FILE} not found. Call /visualize first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


@app.get("/visualize")
def visualize():
    cfg = AnalyzerConfig(
        max_total_lines=2000,
        max_total_chars=50_000,
        max_time_seconds=200,
        max_new_tokens=4096,
        repetition_penalty=1.1
    )

    try:
        data = analyze_to_json(cfg)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualize/file")
def visualize_file():
    if not os.path.exists(OUTPUT_JSON):
        raise HTTPException(status_code=404, detail=f"{OUTPUT_JSON} not found. Call /visualize first.")
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        return JSONResponse(content=json_load_safe(f.read()))


def json_load_safe(text: str):
    import json
    return json.loads(text)
