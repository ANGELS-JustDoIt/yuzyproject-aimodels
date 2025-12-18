# project_root/core/analyzer.py
import os
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# (유지) 기존 설정/파일명
# =========================

TARGET_FOLDER_PATH = r"C:\Pyg\Projects\semi\yuzyproject-aimodels\server"  # (옵션) 로컬 폴더 스캔용
OUTPUT_FILENAME = "project_full_context.txt"

TARGET_EXTENSIONS = [
    '.mjs', '.js', '.ts', '.py', '.java', '.go', '.json', '.yaml', '.yml',
    '.sh', '.rb', '.php', '.html', '.css', '.scss', '.md', '.jsx', '.tsx'
]

IGNORE_DIRS = {
    'node_modules', 'venv', '.git', '__pycache__',
    'dist', 'build', '.idea', '.vscode', 'coverage',
    'frontend', 'front', 'client', 'web',
}

IGNORE_FILES = {
    'package-lock.json',
    'yarn.lock',
    '.DS_Store'
}

INPUT_FILE = "project_full_context.txt"
OUTPUT_JSON = "project_flows.json"

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

SYSTEM_PROMPT = """
You are a 'Senior Backend Architect'.
Your task is to generate a **Deep & Precise Call Graph** JSON.

### 🚨 CRITICAL RULES (DO NOT BE LAZY)
1. **NEVER Leave Children Empty**: You MUST trace down to the **Service/Repository** and **Database (Mongoose)** layers.
   - Bad: `"children": []`
   - Good: `"children": [{"function": "repository...", "children": [{"function": "User.find..."}]}]`
2. **Trace Middleware**: If a route has `isAuth`, add it as the FIRST child node.
3. **Analyze Logic**:
   - `signup`: findByUserid -> bcrypt -> User.save.
   - `post`: isAuth -> controller -> repository -> Post.find/save.
4. **No Recursion**: A function (`login`) CANNOT call itself (`login`).

### ONE-SHOT EXAMPLE (Follow this depth strictly!)
Input Code:
`router.post('/post', isAuth, createPost)`
`function createPost() { ... postRepository.create(...) }`
`function create() { ... new Post(...).save() }`

Output JSON:
{
  "category": "post",
  "endpoints": [
    {
      "method": "POST",
      "url": "/post",
      "function": "createPost",
      "children": [
        { "function": "isAuth", "file": "middleware/auth.mjs", "description": "Auth Check", "children": [] },
        { "function": "create", "file": "data/post.mjs", "description": "Repository Logic", "children": [
            { "function": "Post.save()", "file": "mongoose", "description": "DB Insert", "children": [] }
          ]
        }
      ]
    }
  ]
}

### JSON OUTPUT FORMAT
Return ONLY valid JSON. Structure:
{
  "api": [
    { "category": "auth", "categoryName": "Auth Feature", "endpoints": [...] },
    { "category": "post", "categoryName": "Post Feature", "endpoints": [...] }
  ]
}
"""

# =========================
# 추가: 초보자 설명 프롬프트(2-pass B)
# =========================
EXPLAIN_PROMPT = """
You are a backend tutor for beginners.
Given the call graph JSON, explain each endpoint in Korean with very concrete steps.

Return ONLY valid JSON. No markdown. No commentary.

Rules:
- Each endpoint must include:
  1) oneLineSummary (1줄)
  2) stepByStep (최소 6개 단계, 쉬운 말)
  3) dataFlow (변수 흐름: from -> to -> meaning)
  4) pitfalls (초보자가 헷갈리는 포인트 3개)
  5) glossary (용어 5개: middleware, controller, service, repository, model)

Schema:
{
  "api": [
    {
      "category": "...",
      "categoryName": "...",
      "endpoints": [
        {
          "method": "...",
          "url": "...",
          "function": "...",
          "oneLineSummary": "...",
          "stepByStep": ["..."],
          "dataFlow": [{"var":"...", "from":"...", "to":"...", "meaning":"..."}],
          "pitfalls": ["...","...","..."],
          "glossary": [{"term":"...", "meaning":"..."}]
        }
      ]
    }
  ]
}
"""

# =========================
# 최적화 설정(유지)
# =========================
@dataclass
class AnalyzerConfig:
    max_total_lines: int = 2000
    max_total_chars: int = 50_000
    max_time_seconds: int = 200
    max_new_tokens: int = 4096
    repetition_penalty: float = 1.1

_TOKENIZER = None
_MODEL = None


def _init_torch_perf():
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass


def load_model_once():
    global _TOKENIZER, _MODEL

    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    _init_torch_perf()

    print(f"🔄 모델 로딩 중... ({MODEL_ID})")
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token
    _MODEL.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ 모델 로딩 완료! (device={device})")
    return _TOKENIZER, _MODEL


def _extract_json(text: str) -> str:
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        return "{}"
    return text[start:end + 1]


def _generate_once(tokenizer, model, messages: List[Dict[str, str]], cfg: AnalyzerConfig) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **model_inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=cfg.repetition_penalty,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_time=cfg.max_time_seconds,
        )

    gen_ids = out[0][model_inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# =========================
# NEW: 업로드 텍스쳐(text) 기반 분석
# =========================
def analyze_from_text(
    context_text: str,
    out_dir: str,
    cfg: Optional[AnalyzerConfig] = None
) -> Dict[str, Any]:
    """
    업로드된 텍스쳐(text)를 받아:
    1) out_dir/project_full_context.txt 저장 (파일명 유지)
    2) LLM로 콜그래프 JSON 생성 -> out_dir/project_flows.json 저장 (파일명 유지)
    3) 결과 dict 반환
    """
    if cfg is None:
        cfg = AnalyzerConfig()

    os.makedirs(out_dir, exist_ok=True)

    # 1) 텍스쳐 저장 (파일명 유지)
    context_path = os.path.join(out_dir, INPUT_FILE)
    with open(context_path, "w", encoding="utf-8") as f:
        f.write(context_text)

    # 2) 모델 로딩(캐시)
    tokenizer, model = load_model_once()

    # 3) 분석
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze the source code to visualize the logic flow. Generate the Nested JSON Structure:\n\n{context_text[:cfg.max_total_chars]}"},
    ]
    raw = _generate_once(tokenizer, model, messages, cfg)
    json_str = _extract_json(raw)

    # 4) 파싱/저장
    try:
        data = json.loads(json_str)
    except Exception:
        # 복구 패스(기존 로직 유지)
        repair_messages = [
            {"role": "system", "content": "Return ONLY valid JSON. Do not add any commentary."},
            {"role": "user", "content": f"Fix this into valid JSON only:\n\n{json_str[:8000]}"},
        ]
        repair_cfg = AnalyzerConfig(
            max_total_lines=cfg.max_total_lines,
            max_total_chars=cfg.max_total_chars,
            max_time_seconds=min(20, cfg.max_time_seconds),
            max_new_tokens=2048,
            repetition_penalty=cfg.repetition_penalty,
        )
        repaired = _generate_once(tokenizer, model, repair_messages, repair_cfg)
        repaired_json_str = _extract_json(repaired)
        data = json.loads(repaired_json_str)

    out_json_path = os.path.join(out_dir, OUTPUT_JSON)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def explain_from_analysis(analysis_json: Dict[str, Any], out_dir: str, cfg: Optional[AnalyzerConfig] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = AnalyzerConfig(max_new_tokens=2048, max_time_seconds=60)

    tokenizer, model = load_model_once()

    messages = [
        {"role": "system", "content": EXPLAIN_PROMPT},
        {"role": "user", "content": json.dumps(analysis_json, ensure_ascii=False)[:80000]},
    ]

    raw = _generate_once(tokenizer, model, messages, cfg)
    json_str = _extract_json(raw)

    try:
        data = json.loads(json_str)
    except Exception as e:
        # 🔥 실패 시 raw 저장하고 "에러 정보"를 반환 (서버 500 방지)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "raw_explain_output.txt"), "w", encoding="utf-8") as f:
            f.write(raw)
        with open(os.path.join(out_dir, "raw_explain_json_attempt.txt"), "w", encoding="utf-8") as f:
            f.write(json_str)

        return {
            "_error": f"explain_json_parse_failed: {str(e)}",
            "_raw_saved": True
        }

    out_path = os.path.join(out_dir, "project_explain.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data



# =========================
# 기존 로컬 폴더 스캔 파이프라인(옵션으로 유지)
# =========================
def _should_ignore_dir(dirname: str) -> bool:
    return dirname in IGNORE_DIRS


def _should_collect_file(filename: str) -> bool:
    if filename in IGNORE_FILES:
        return False
    _, ext = os.path.splitext(filename)
    return ext.lower() in TARGET_EXTENSIONS


def _read_file_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def build_project_full_context(target_folder: str, output_file: str, cfg: AnalyzerConfig) -> Dict[str, Any]:
    if not os.path.exists(target_folder):
        raise FileNotFoundError(f"❌ 대상 폴더가 없습니다: {target_folder}")

    chunks: List[str] = []
    included_files = 0
    skipped_files = 0
    total_lines = 0
    total_chars = 0

    for root, dirs, files in os.walk(target_folder):
        dirs[:] = [d for d in dirs if not _should_ignore_dir(d)]
        for file in files:
            if not _should_collect_file(file):
                skipped_files += 1
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, target_folder)
            try:
                lines = _read_file_lines(full_path)
                if total_lines >= cfg.max_total_lines or total_chars >= cfg.max_total_chars:
                    break

                remain_lines = cfg.max_total_lines - total_lines
                take_lines = lines[:max(0, remain_lines)]
                content = "".join(take_lines)

                remain_chars = cfg.max_total_chars - total_chars
                if len(content) > remain_chars:
                    content = content[:max(0, remain_chars)]

                block = "\n".join([
                    "===== FILE START =====",
                    f"PATH: {rel_path}",
                    "----- CODE -----",
                    content.rstrip("\n"),
                    "===== FILE END =====",
                    ""
                ])

                added_lines = content.count("\n") + 1 if content else 0
                added_chars = len(content)

                if added_lines == 0:
                    skipped_files += 1
                    continue

                chunks.append(block)
                included_files += 1
                total_lines += added_lines
                total_chars += added_chars

            except Exception:
                skipped_files += 1
                continue

        if total_lines >= cfg.max_total_lines or total_chars >= cfg.max_total_chars:
            break

    merged = "\n".join(chunks)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(merged)

    return {
        "output_file": output_file,
        "included_files": included_files,
        "skipped_files": skipped_files,
        "total_lines": total_lines,
        "total_chars": total_chars,
    }


def analyze_to_json(cfg: Optional[AnalyzerConfig] = None) -> Dict[str, Any]:
    """
    (옵션) 기존 로컬 폴더 스캔 → 분석
    """
    if cfg is None:
        cfg = AnalyzerConfig()

    build_project_full_context(TARGET_FOLDER_PATH, OUTPUT_FILENAME, cfg)

    tokenizer, model = load_model_once()

    with open(OUTPUT_FILENAME, "r", encoding="utf-8", errors="ignore") as f:
        code_context = f.read()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze the source code to visualize the logic flow. Generate the Nested JSON Structure:\n\n{code_context}"}
    ]

    response = _generate_once(tokenizer, model, messages, cfg)
    json_str = _extract_json(response)
    data = json.loads(json_str)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data
