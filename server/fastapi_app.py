# server/fastapi_app.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import ctypes
import cv2
from pathlib import Path
from datetime import datetime

from app.capture.screen_capture import capture_fullscreen_bgr
from app.capture.ocr import image_to_text  # OCR은 2순위지만, 현재는 결과 확인용으로만 사용

app = FastAPI(title="OCR Crop Server", version="1.0.0")

# -----------------------------
# 저장 폴더 + 정적 서빙
# -----------------------------
CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(exist_ok=True)
app.mount("/captures", StaticFiles(directory=str(CAPTURE_DIR)), name="captures")


# -----------------------------
# Windows DPI scale factor
# -----------------------------
def get_windows_scale_factor() -> float:
    """Windows DPI scale factor (예: 1.0, 1.25, 1.5, 2.0)"""
    try:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor aware
        except Exception:
            pass
        ctypes.windll.user32.SetProcessDPIAware()
        dpi = ctypes.windll.user32.GetDpiForSystem()  # 96 = 100%
        return float(dpi) / 96.0
    except Exception:
        return 1.0


# -----------------------------
# Request Model
# -----------------------------
class CropRequest(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    w: int = Field(..., gt=0)
    h: int = Field(..., gt=0)

    # 2순위(정확도) 파라미터 — 지금은 테스트 편의용
    lang: str = Field("kor+eng")
    scale: int = Field(3, ge=1, le=6)


# -----------------------------
# Helpers
# -----------------------------
def _crop_from_screen(screen, x: int, y: int, w: int, h: int):
    H, W = screen.shape[:2]
    if x >= W or y >= H:
        raise HTTPException(status_code=400, detail=f"좌표가 화면 밖입니다. screen=({W}x{H})")

    x2 = min(x + w, W)
    y2 = min(y + h, H)
    if x2 <= x or y2 <= y:
        raise HTTPException(status_code=400, detail="crop 영역이 유효하지 않습니다.")

    cropped = screen[y:y2, x:x2]
    return cropped, (x, y, x2 - x, y2 - y), (W, H)


def _save_png(bgr, prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{prefix}_{ts}.png"
    path = CAPTURE_DIR / name
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="파일 저장 실패")
    return name


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/screen-info")
def screen_info():
    screen = capture_fullscreen_bgr()
    h, w = screen.shape[:2]
    return {"screen_size": {"w": w, "h": h}, "dpi_scale": get_windows_scale_factor()}


@app.get("/screenshot")
def screenshot():
    screen = capture_fullscreen_bgr()
    ok, buf = cv2.imencode(".png", screen)
    if not ok:
        raise HTTPException(status_code=500, detail="png encode 실패")
    return Response(content=buf.tobytes(), media_type="image/png")


@app.post("/save-crop")
def save_crop(req: CropRequest):
    """
    ✅ 핵심:
    - 요청 받은 좌표(x,y,w,h)는 "서버 스크린샷 픽셀" 기준이어야 함
    - 호출 순간 실제 화면을 캡쳐해서 crop 저장
    """
    screen = capture_fullscreen_bgr()
    cropped, crop_info, screen_wh = _crop_from_screen(screen, req.x, req.y, req.w, req.h)

    # 1) crop 이미지 저장
    crop_name = _save_png(cropped, "crop")
    crop_url = f"/captures/{crop_name}"

    # (선택) 2) OCR도 같이 — 지금은 결과 확인용
    try:
        text = image_to_text(cropped, lang=req.lang, scale=req.scale)
    except Exception as e:
        text = f"[OCR 오류: {e}]"

    x, y, w, h = crop_info
    W, H = screen_wh
    return {
        "saved": True,
        "screen_size": {"w": W, "h": H},
        "crop": {"x": x, "y": y, "w": w, "h": h},
        "file": crop_name,
        "url": crop_url,
        "text": text,
    }


# -----------------------------
# Pick UI (drag => auto save)
# -----------------------------
@app.get("/pick", response_class=HTMLResponse)
def pick_ui():
    info = screen_info()
    W = info["screen_size"]["w"]
    H = info["screen_size"]["h"]
    dpi_scale = info.get("dpi_scale", 1.0)

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>OCR Pick</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; background:#111; color:#eee; }}
    .row {{ display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap; }}
    .panel {{ width: 420px; padding: 12px; background:#1b1b1b; border-radius:12px; }}
    .btn {{ padding:8px 10px; border:0; border-radius:10px; cursor:pointer; }}
    .btn.gray {{ background:#333; color:#eee; }}
    .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; }}
    .small {{ font-size: 12px; color:#bbb; }}

    #wrap {{ position: relative; display:inline-block; border:1px solid #333; border-radius:12px; overflow:hidden; }}
    #img {{ display:block; max-width: 1200px; width:100%; height:auto; cursor:crosshair; user-select:none; }}
    #box {{
      position:absolute; border:2px solid #22c55e; background:rgba(34,197,94,0.15);
      display:none; pointer-events:none;
    }}

    pre {{ white-space: pre-wrap; word-break: break-word; background:#0b0b0b; padding:10px; border-radius:10px; }}
    .ok {{ color:#22c55e; }}
    .bad {{ color:#ef4444; }}
    .info {{ color:#60a5fa; }}
    img.preview {{ max-width:900px; border:1px solid #333; border-radius:12px; margin-top:8px; }}
  </style>
</head>
<body>
  <h2>드래그 → 즉시 저장(서버 캡처 기준)</h2>
  <div class="small">screen_size: {W}x{H} / dpi_scale: {dpi_scale}</div>

  <div class="row">
    <div class="panel">
      <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;">
        <button class="btn gray" id="btnRefresh">스크린샷 새로고침</button>
      </div>

      <div class="mono">
        x: <span id="x">-</span><br/>
        y: <span id="y">-</span><br/>
        w: <span id="w">-</span><br/>
        h: <span id="h">-</span><br/>
      </div>

      <hr style="border:0;border-top:1px solid #333; margin:12px 0;"/>

      <div class="small">상태</div>
      <pre id="status" class="mono info">이미지 위에서 드래그하고 마우스를 놓으면 자동으로 저장됩니다.</pre>

      <div class="small">저장된 crop 미리보기</div>
      <div id="preview"></div>

      <div class="small">OCR 결과(2순위)</div>
      <pre id="ocr" class="mono" style="max-height:260px; overflow:auto;">(여기에 표시)</pre>
    </div>

    <div>
      <div id="wrap">
        <img id="img" src="/screenshot?ts={datetime.now().timestamp()}" draggable="false"/>
        <div id="box"></div>
      </div>
      <div class="small" style="margin-top:8px;">
        ✅ 좌표는 항상 서버 스크린샷 픽셀({W}x{H}) 기준으로 계산됨 (축소 표시여도 OK)
      </div>
    </div>
  </div>

<script>
const img = document.getElementById('img');
const box = document.getElementById('box');
const elX = document.getElementById('x');
const elY = document.getElementById('y');
const elW = document.getElementById('w');
const elH = document.getElementById('h');
const statusEl = document.getElementById('status');
const previewEl = document.getElementById('preview');
const ocrEl = document.getElementById('ocr');

let dragging = false;
let saving = false;
let start = null;
let cur = null;

function clamp(v, min, max) {{
  return Math.max(min, Math.min(max, v));
}}

function clientToImageXY(clientX, clientY) {{
  const r = img.getBoundingClientRect();
  const px = (clientX - r.left);
  const py = (clientY - r.top);

  const sx = img.naturalWidth / r.width;
  const sy = img.naturalHeight / r.height;

  const ix = clamp(Math.round(px * sx), 0, img.naturalWidth - 1);
  const iy = clamp(Math.round(py * sy), 0, img.naturalHeight - 1);
  return {{ x: ix, y: iy }};
}}

function drawBox(a, b) {{
  const r = img.getBoundingClientRect();
  const sx = r.width / img.naturalWidth;
  const sy = r.height / img.naturalHeight;

  const x1 = Math.min(a.x, b.x);
  const y1 = Math.min(a.y, b.y);
  const x2 = Math.max(a.x, b.x);
  const y2 = Math.max(a.y, b.y);

  const left = x1 * sx;
  const top  = y1 * sy;
  const w    = Math.max(1, (x2 - x1)) * sx;
  const h    = Math.max(1, (y2 - y1)) * sy;

  box.style.display = 'block';
  box.style.left = left + 'px';
  box.style.top = top + 'px';
  box.style.width = w + 'px';
  box.style.height = h + 'px';
}}

function updateNums(x,y,w,h) {{
  elX.textContent = x;
  elY.textContent = y;
  elW.textContent = w;
  elH.textContent = h;
}}

async function saveCrop(x,y,w,h) {{
  saving = true;
  statusEl.textContent = "⏳ 저장 중...";
  statusEl.className = "mono info";
  previewEl.innerHTML = "";
  ocrEl.textContent = "(처리 중)";

  const payload = {{ x, y, w, h, lang: "kor+eng", scale: 3 }};
  const res = await fetch("/save-crop", {{
    method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify(payload)
  }});
  const data = await res.json();

  if (!res.ok) {{
    statusEl.textContent = "❌ 저장 실패: " + JSON.stringify(data);
    statusEl.className = "mono bad";
    saving = false;
    return;
  }}

  statusEl.textContent = "✅ 저장 완료: " + data.url;
  statusEl.className = "mono ok";

  previewEl.innerHTML = '<img class="preview" src="' + data.url + '?ts=' + Date.now() + '"/>';
  ocrEl.textContent = data.text || "";

  saving = false;
}}

img.addEventListener("pointerdown", (e) => {{
  if (saving) return;
  if (!img.naturalWidth) return;

  dragging = true;
  img.setPointerCapture(e.pointerId);

  start = clientToImageXY(e.clientX, e.clientY);
  cur = start;
  drawBox(start, cur);

  updateNums(start.x, start.y, 1, 1);
  statusEl.textContent = "드래그 중... (놓으면 저장)";
  statusEl.className = "mono info";
}});

img.addEventListener("pointermove", (e) => {{
  if (!dragging || saving) return;
  cur = clientToImageXY(e.clientX, e.clientY);
  drawBox(start, cur);

  const x = Math.min(start.x, cur.x);
  const y = Math.min(start.y, cur.y);
  const w = Math.max(1, Math.abs(cur.x - start.x));
  const h = Math.max(1, Math.abs(cur.y - start.y));
  updateNums(x,y,w,h);
}});

img.addEventListener("pointerup", async (e) => {{
  if (!dragging || saving) return;
  dragging = false;

  cur = clientToImageXY(e.clientX, e.clientY);
  drawBox(start, cur);

  const x = Math.min(start.x, cur.x);
  const y = Math.min(start.y, cur.y);
  const w = Math.max(1, Math.abs(cur.x - start.x));
  const h = Math.max(1, Math.abs(cur.y - start.y));

  updateNums(x,y,w,h);
  box.style.display = "none";

  await saveCrop(x,y,w,h);
}});

document.getElementById("btnRefresh").onclick = () => {{
  img.src = "/screenshot?ts=" + Date.now();
  box.style.display = "none";
  statusEl.textContent = "이미지 위에서 드래그하고 놓으면 자동 저장됩니다.";
  statusEl.className = "mono info";
}};
</script>
</body>
</html>
"""
    return HTMLResponse(html)
