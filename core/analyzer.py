# project_root/core/analyzer.py
import os
import json
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 대상 폴더 경로 설정(실제 백엔드 경로로 변경 필요)
TARGET_FOLDER_PATH = r"C:\Pyg\Projects\semi\yuzyproject-aimodels\server"

# 결과물을 저장할 파일명
OUTPUT_FILENAME = "project_full_context.txt"

# 수집할 파일 확장자
TARGET_EXTENSIONS = [
    '.mjs', '.js', '.ts', '.py', '.java', '.go', '.json', '.yaml', '.yml',
    '.sh', '.rb', '.php', '.html', '.css', '.scss', '.md', '.jsx', '.tsx'
]

IGNORE_DIRS = {
    'node_modules', 'venv', '.git', '__pycache__',
    'dist', 'build', '.idea', '.vscode', 'coverage',
    'frontend', 'front', 'client', 'web',
}


# 제외할 파일 (파일 이름만 정확히)
IGNORE_FILES = {
    'package-lock.json',
    'yarn.lock',
    '.DS_Store'
}

# 설정값 (유지)
INPUT_FILE = "project_full_context.txt"
OUTPUT_JSON = "project_flows.json"

# 모델 ID (유지)
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


# =========================
# (유지) 네 프롬프트 그대로
# =========================
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
# 최적화 설정
# =========================
@dataclass
class AnalyzerConfig:
    # “원래 500~1700줄 분석했다” 요구 반영 (기본 1700줄)
    max_total_lines: int = 1700
    # 추가 안전장치: 문자 수 제한(너가 쓰던 50,000자 감각 유지)
    max_total_chars: int = 50_000

    # 생성 제어: 1분 내 목표 (기본 55초 제한)
    max_time_seconds: int = 200
    max_new_tokens: int = 4096  # 2048보다 정확도 더 나오는데, 시간 제한이 있으니 안전

    # 반복 방지 (유지)
    repetition_penalty: float = 1.1


# =========================
# 모델 캐시(서버에서 재사용하려고 전역 싱글턴)
# =========================
_TOKENIZER = None
_MODEL = None


def _init_torch_perf():
    # GPU면 TF32 허용(속도↑, 품질 큰 차이 없음)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 실행 환경: {device.upper()}")
    print(f"🚀 실행 환경: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print("✅ Qwen 1.5B 최종 프롬프트(Post 깊이 강화) 설정 완료")

    print(f"🔄 모델 로딩 중... ({MODEL_ID})")

    # torch_dtype 경고 없애려고 dtype 사용
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
    print("✅ 모델 로딩 완료!")
    return _TOKENIZER, _MODEL


# =========================
# 파일 수집(속도/토큰 최적화 핵심)
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


def build_project_full_context(
    target_folder: str,
    output_file: str,
    cfg: AnalyzerConfig
) -> Dict[str, Any]:
    """
    - 네 요구: 폴더 -> 확장자 기반 수집 -> 하나의 텍스쳐 파일
    - 최적화: max_total_lines / max_total_chars 내에서만 누적(예전 500~1700줄 감각 재현)
    """
    if not os.path.exists(target_folder):
        raise FileNotFoundError(f"❌ 대상 폴더가 없습니다: {target_folder}")

    print("✅ 설정이 완료되었습니다.")
    print(f"   - 대상 폴더: {target_folder}")
    print(f"   - 수집 확장자: {TARGET_EXTENSIONS}")

    print(f"📦 폴더 스캔 시작: {target_folder}")

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

                # 남은 예산 계산
                if total_lines >= cfg.max_total_lines or total_chars >= cfg.max_total_chars:
                    break

                remain_lines = cfg.max_total_lines - total_lines
                # 파일 라인 일부만 취함
                take_lines = lines[:max(0, remain_lines)]

                content = "".join(take_lines)

                # 문자 예산도 적용
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

                # 카운트 업데이트
                added_lines = content.count("\n") + 1 if content else 0
                added_chars = len(content)

                # 혹시라도 0이면 스킵
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

        # 상위 루프도 중단 조건 체크
        if total_lines >= cfg.max_total_lines or total_chars >= cfg.max_total_chars:
            break

    merged = "\n".join(chunks)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(merged)

    print("✅ 텍스쳐 파일 생성 완료!")
    print(f"   - 저장 경로: {output_file}")
    print(f"   - 포함 파일 수: {included_files}")
    print(f"   - 스킵 파일 수: {skipped_files}")
    print(f"   - 누적 라인 수(대략): {total_lines}")
    print(f"   - 누적 문자 수: {total_chars}")

    return {
        "output_file": output_file,
        "included_files": included_files,
        "skipped_files": skipped_files,
        "total_lines": total_lines,
        "total_chars": total_chars,
    }


# =========================
# LLM 분석
# =========================
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
            max_time=cfg.max_time_seconds,  # 1분 내 목표
        )

    gen_ids = out[0][model_inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def analyze_to_json(cfg: Optional[AnalyzerConfig] = None) -> Dict[str, Any]:
    """
    전체 파이프라인:
    1) 폴더 -> project_full_context.txt 생성
    2) 모델 로딩(1회 캐시)
    3) 분석 -> project_flows.json 저장
    """
    if cfg is None:
        cfg = AnalyzerConfig()

    # 1) 텍스쳐 생성
    build_project_full_context(TARGET_FOLDER_PATH, OUTPUT_FILENAME, cfg)

    # INPUT_FILE 이름 유지(네 요구)
    if OUTPUT_FILENAME != INPUT_FILE:
        # 너가 “절대 파일명 무시하지 말라”고 해서 여기서 강제 일치시키지 않고 경고만
        print("⚠️ 경고: OUTPUT_FILENAME과 INPUT_FILE명이 다릅니다.")
        print(f"   - OUTPUT_FILENAME: {OUTPUT_FILENAME}")
        print(f"   - INPUT_FILE: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        # 혹시 OUTPUT_FILENAME만 생성됐고 INPUT_FILE이 다르면 여기서 막히니까 안내
        raise FileNotFoundError(f"❌ 파일 없음: {INPUT_FILE} (OUTPUT_FILENAME={OUTPUT_FILENAME} 생성됨)")

    # 2) 모델 로딩(캐시)
    tokenizer, model = load_model_once()

    # 3) 코드 로드 (예전 방식: 큰 파일 전체 X, 지금은 이미 라인/문자 제한된 텍스쳐)
    print(f"📂 코드 분석 시작: '{INPUT_FILE}'")
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        code_context = f.read()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze the source code to visualize the logic flow. Generate the Nested JSON Structure:\n\n{code_context}"}
    ]

    print("🧠 Qwen 1.5B가 '고속 정밀 모드(Greedy)'로 분석 중입니다... (빠름 & 결과 고정)")
    t0 = time.time()
    response = _generate_once(tokenizer, model, messages, cfg)
    dt = time.time() - t0

    json_str = _extract_json(response)

    # 4) 1차 파싱 시도
    try:
        data = json.loads(json_str)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 분석 완료! -> {OUTPUT_JSON} ({dt:.1f}s)")
        return data
    except Exception:
        # 5) 실패 시: “JSON만 고쳐라” 복구 패스 (입력이 아주 짧아져서 빠름)
        print("⚠️ JSON 파싱 실패. 복구 패스(짧은 입력)로 재시도합니다.")
        repair_messages = [
            {"role": "system", "content": "Return ONLY valid JSON. Do not add any commentary."},
            {"role": "user", "content": f"Fix this into valid JSON only:\n\n{json_str[:8000]}"}
        ]

        repair_cfg = AnalyzerConfig(
            max_total_lines=cfg.max_total_lines,
            max_total_chars=cfg.max_total_chars,
            max_time_seconds=min(20, cfg.max_time_seconds),
            max_new_tokens=2048,
            repetition_penalty=cfg.repetition_penalty
        )

        repaired = _generate_once(tokenizer, model, repair_messages, repair_cfg)
        repaired_json_str = _extract_json(repaired)

        try:
            data = json.loads(repaired_json_str)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ 분석 완료(복구 성공)! -> {OUTPUT_JSON}")
            return data
        except Exception:
            # 디버깅 저장
            with open("raw_model_output.txt", "w", encoding="utf-8") as f:
                f.write(response)
            with open("raw_model_json_attempt.txt", "w", encoding="utf-8") as f:
                f.write(json_str)
            print("❌ JSON 파싱 최종 실패.")
            print("🧾 저장됨: raw_model_output.txt / raw_model_json_attempt.txt")
            raise


# =========================
# 트리 출력(유지)
# =========================
def print_tree(node, prefix="", is_last=True):
    connector = "└── " if is_last else "├── "

    if "method" in node:
        name = f"[{node['method']}] {node['url']} ({node.get('description', '')})"
    elif "function" in node:
        name = f"ƒ {node['function']} - {node.get('description', '')}"
    elif "category" in node:
        name = f"📂 Category: {node.get('categoryName', node['category'])}"
    else:
        name = "Unknown Node"

    print(prefix + connector + name)

    children = node.get("children", [])
    if "endpoints" in node:
        children = node["endpoints"]

    count = len(children)
    for i, child in enumerate(children):
        new_prefix = prefix + ("    " if is_last else "│   ")
        print_tree(child, new_prefix, i == count - 1)


def visualize_json_structure():
    if not os.path.exists(OUTPUT_JSON):
        print("❌ JSON 파일이 없습니다.")
        return

    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n🌳 API Call Graph Analysis Result\n" + "=" * 40)
    if "api" in data:
        for cat in data["api"]:
            print_tree(cat)
    else:
        print("⚠️ 'api' 키를 찾을 수 없습니다. JSON 구조를 확인하세요.")


# =========================
# CLI 실행
# =========================
if __name__ == "__main__":
    cfg = AnalyzerConfig(
        max_total_lines=2000,  
        max_total_chars=50_000,
        max_time_seconds=200,
        max_new_tokens=4096,
        repetition_penalty=1.1
    )
    data = analyze_to_json(cfg)
    visualize_json_structure()
