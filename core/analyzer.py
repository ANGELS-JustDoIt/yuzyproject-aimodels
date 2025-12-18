# project_root/core/analyzer.py
import os
import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Output file names
# =========================
INPUT_FILE = "project_full_context.txt"
OUTPUT_JSON = "project_flows.json"

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

SYSTEM_PROMPT = """
You are a Senior Backend Architect.
Your job is to read backend source code and return a precise call-graph JSON.

RULES:
1. Follow all HTTP routes from router → middleware → controller → service/repository → database.
2. For each HTTP endpoint, include every major function that is called in order.
3. Never invent functions that clearly do not exist in the code.
4. Do NOT create recursive calls (a function must not directly call itself).

JSON FORMAT (return ONLY this JSON, no extra text):
{
  "api": [
    {
      "category": "auth",
      "categoryName": "Auth Feature",
      "endpoints": [
        {
          "method": "POST",
          "url": "/auth/login",
          "function": "login",
          "file": "controller/auth.mjs",
          "description": "Handle user login request and return JWT token",
          "children": [
            {
              "function": "isAuth",
              "file": "middleware/auth.mjs",
              "description": "Validate JWT token",
              "children": []
            },
            {
              "function": "findByUserid",
              "file": "data/auth.mjs",
              "description": "Load user record from database",
              "children": []
            }
          ]
        }
      ]
    }
  ]
}

REQUIREMENTS:
- Top-level MUST be an object with key "api".
- "api" MUST be an array; each item MUST have:
  - "category" (string)
  - "categoryName" (string)
  - "endpoints" (array)
- Each endpoint MUST have at least: "method", "url", "function", "children".
- "children" MUST be an array (can be empty only if the node is truly a leaf).
"""

# =========================
# Explanation prompt (2nd pass)
# =========================
EXPLAIN_PROMPT = """
You are a backend tutor for beginners.
Given the call-graph JSON of an API server, explain each endpoint in simple English.

Return ONLY valid JSON. No markdown. No commentary.

For every endpoint in the input JSON:
- oneLineSummary: a one-sentence summary of what this endpoint does.
- stepByStep: at least 6 short bullet-style steps, describing the full request flow.
- dataFlow: list of objects { "var", "from", "to", "meaning" } that describe how key data moves.
- pitfalls: 3 short bullet points about common mistakes or gotchas.
- glossary: 5 terms with explanations (for example: middleware, controller, service, repository, model).

Schema you MUST return:
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
          "dataFlow": [
            { "var": "...", "from": "...", "to": "...", "meaning": "..." }
          ],
          "pitfalls": ["...", "...", "..."],
          "glossary": [
            { "term": "...", "meaning": "..." }
          ]
        }
      ]
    }
  ]
}
"""

# =========================
# Config
# =========================
@dataclass
class AnalyzerConfig:
    max_total_lines: int = 2000
    max_total_chars: int = 50_000
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
        trust_remote_code=True,
    )
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token
    _MODEL.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ 모델 로딩 완료! (device={device})")
    return _TOKENIZER, _MODEL


def _extract_json(text: str) -> str:
    """
    Extract the first JSON object from the model output by taking everything
    from the first '{' to the last '}' after stripping markdown fences.
    """
    cleaned = re.sub(r"```(json)?|```", "", text.strip(), flags=re.MULTILINE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return cleaned[start : end + 1]


def _generate_once(tokenizer, model, messages: List[Dict[str, str]], cfg: AnalyzerConfig) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
        )

    gen_ids = out[0][model_inputs.input_ids.shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def analyze_from_text(
    context_text: str,
    out_dir: str,
    cfg: Optional[AnalyzerConfig] = None
) -> Dict[str, Any]:
    """
    Take merged source-code text from the frontend and:
    1) Save it to out_dir/project_full_context.txt
    2) Ask the LLM to generate the call-graph JSON
    3) Save the JSON to out_dir/project_flows.json and return it as dict
    """
    if cfg is None:
        cfg = AnalyzerConfig()

    os.makedirs(out_dir, exist_ok=True)

    # 1) Save raw context (for inspection)
    context_path = os.path.join(out_dir, INPUT_FILE)
    with open(context_path, "w", encoding="utf-8") as f:
        f.write(context_text)

    # 2) Load model (cached)
    tokenizer, model = load_model_once()

    # 3) Run analysis
    prompt_text = context_text[: cfg.max_total_chars]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze the following backend project source code and return ONLY the "
                'JSON object that follows the specified \"api\" schema.\n\n'
                f"{prompt_text}"
            ),
        },
    ]

    raw = _generate_once(tokenizer, model, messages, cfg)

    # 4) Parse & save
    json_str = _extract_json(raw)
    data = json.loads(json_str)

    out_json_path = os.path.join(out_dir, OUTPUT_JSON)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def explain_from_analysis(
    analysis_json: Dict[str, Any],
    out_dir: str,
    cfg: Optional[AnalyzerConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = AnalyzerConfig(max_new_tokens=2048)

    tokenizer, model = load_model_once()

    messages = [
        {"role": "system", "content": EXPLAIN_PROMPT},
        {
            "role": "user",
            "content": json.dumps(analysis_json, ensure_ascii=False)[:80000],
        },
    ]

    raw = _generate_once(tokenizer, model, messages, cfg)

    json_str = _extract_json(raw)
    data = json.loads(json_str)

    out_path = os.path.join(out_dir, "project_explain.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data
