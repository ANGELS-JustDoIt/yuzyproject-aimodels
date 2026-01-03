# project_root/main.py
import os
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
# 환경 변수에서 허용할 origin 목록을 가져오거나 기본값 사용
allowed_origins = os.getenv("FRONTEND_URL", "http://localhost:3000,http://127.0.0.1:3000").split(",")
# IP 주소 패턴도 허용 (개발 환경)
if os.getenv("NODE_ENV") != "production":
    # 모든 origin 허용 (개발 환경)
    origins = ["*"]
else:
    origins = allowed_origins

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
async def capture(
    file: UploadFile = File(default=None),
    lang: str = "kor+eng",
    scale: int = 3,
    code_mode: bool = True,
    layout: bool = True,
    normalize: bool = True,
):
    """
    이미지 파일 업로드 -> OCR 인식
    
    Parameters:
        file: 업로드할 이미지 파일 (선택적, 없으면 화면 캡처 모드)
        lang: OCR 언어 (기본값: "kor+eng")
        scale: 이미지 스케일 (기본값: 3)
        code_mode: 코드 모드 활성화 (기본값: True)
        layout: 레이아웃 모드 활성화 (기본값: True)
        normalize: 정규화 활성화 (기본값: True)
    
    Returns:
        JSONResponse: {
            "success": bool,
            "text": str (OCR 결과),
            "method": str (사용된 OCR 방법),
            "error": str (에러 메시지, 실패 시)
        }
    """
    try:
        # 파일이 업로드된 경우
        if file is not None:
            from PIL import Image
            import io
            import numpy as np
            import cv2
            
            # 파일 읽기
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다.")
            
            # OCR 수행
            from core.capture import image_to_text, get_tesseract_words, get_winrt_words, merge_tesseract_winrt_results, open_image_any
            
            try:
                # Tesseract와 WinRT 모두 시도
                tesseract_words = None
                winrt_words = None
                ocr_result = None
                ocr_method = None
                
                try:
                    tesseract_words = get_tesseract_words(
                        img,
                        lang=lang,
                        scale=scale,
                        code_mode=code_mode,
                        remove_emoji=True
                    )
                except Exception as e:
                    print(f"⚠ Tesseract OCR 실패: {e}")
                
                try:
                    from core.capture import check_winrt_available
                    winrt_available, _ = check_winrt_available()
                    if winrt_available:
                        winrt_words = get_winrt_words(
                            img,
                            scale=scale,
                            code_mode=code_mode,
                            remove_emoji=True
                        )
                except Exception as e:
                    print(f"⚠ WinRT OCR 실패: {e}")
                
                # 결과 병합 또는 단일 결과 사용
                if tesseract_words and winrt_words:
                    try:
                        pil_img = open_image_any(img)
                        ocr_result = merge_tesseract_winrt_results(
                            tesseract_words,
                            winrt_words,
                            pil_img
                        )
                        ocr_method = "Tesseract + WinRT 병합"
                    except Exception as e:
                        print(f"⚠ 병합 실패: {e}, Tesseract 결과 사용")
                        from core.capture import reconstruct_text_from_words
                        ocr_result = reconstruct_text_from_words(
                            tesseract_words,
                            code_mode=code_mode,
                            normalize=normalize,
                            indent_step=4,
                            remove_emoji=True,
                        )
                        ocr_method = "Tesseract (병합 실패)"
                elif tesseract_words:
                    from core.capture import reconstruct_text_from_words
                    ocr_result = reconstruct_text_from_words(
                        tesseract_words,
                        code_mode=code_mode,
                        normalize=normalize,
                        indent_step=4,
                        remove_emoji=True,
                    )
                    ocr_method = "Tesseract"
                elif winrt_words:
                    from core.capture import image_to_text_winrt
                    ocr_result = image_to_text_winrt(
                        img,
                        scale=scale,
                        code_mode=code_mode,
                        normalize=normalize,
                        indent_step=4,
                        remove_emoji=True,
                    )
                    ocr_method = "WinRT"
                else:
                    # 기본 OCR 시도
                    ocr_result = image_to_text(
                        img,
                        lang=lang,
                        scale=scale,
                        code_mode=code_mode,
                        layout=layout,
                        normalize=normalize,
                        indent_step=4,
                        remove_emoji=True,
                    )
                    ocr_method = "Tesseract (기본)"
                
                if ocr_result:
                    return JSONResponse(content={
                        "success": True,
                        "text": ocr_result,
                        "method": ocr_method,
                        "error": None
                    })
                else:
                    return JSONResponse(content={
                        "success": False,
                        "error": "OCR 결과가 비어있습니다.",
                        "text": "",
                        "method": ""
                    })
            except Exception as e:
                import traceback
                error_msg = f"OCR 처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
                return JSONResponse(content={
                    "success": False,
                    "error": error_msg,
                    "text": "",
                    "method": ""
                })
        else:
            # 파일이 없으면 기존 화면 캡처 모드
            result = capture_and_ocr()
            return JSONResponse(content=result)
    except Exception as e:
        import traceback
        error_msg = f"처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        return JSONResponse(content={
            "success": False,
            "error": error_msg,
            "text": "",
            "method": ""
        })


def json_load_safe(text: str):
    import json
    return json.loads(text)
