# project_root/core/analyzer.py
import os
import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Output file names
# =========================
INPUT_FILE = "project_full_context.txt"
OUTPUT_JSON = "project_flows.json"
WARN_FILE = "analyzer_warnings.txt"

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# =========================
# SYSTEM PROMPT (precision + structure + example)
# =========================
SYSTEM_PROMPT = r"""
You are a Senior Backend Architect and Static Code Analyst.
Your task is to read backend source code and return a precise, verifiable HTTP request call-graph as JSON.

=====================
ABSOLUTE OUTPUT RULES
=====================
- Return ONLY one valid JSON object. No markdown. No commentary.
- Strict JSON only (double quotes, no trailing commas).
- Top-level MUST be: { "api": [...] }  (api is an array)

- EVERY endpoint MUST include:
  "method", "url", "function", "file", "description", "children"

- CRITICAL:
  EVERY node object in the entire tree MUST include "children".
  If a node is a leaf, it MUST include: "children": [].
  Never omit "children". Never use null.

=====================
TRACE ACCURACY RULES
=====================
1) ONLY output functions/middleware/files that are explicitly present in the provided code text.
   - If a function name does NOT appear verbatim in the code text, DO NOT output it.
   - Do NOT invent helpers like "findUserid" or "getUserFromReq" if they don't exist.
2) NEVER invent layers (service/repository) if code jumps directly controller → data.
3) Middleware:
   - Only include middleware if it is explicitly attached on that route line.
   - Middleware arrays (e.g., validateSignup, validatePost) MUST be treated as ONE middleware node (do not expand).
4) Do NOT fabricate controller → controller calls.
   - A controller node's children should be non-controller (data/repository) unless code explicitly calls another controller.
5) Conditional logic (if / ternary) is NOT a function call.
   - You MAY represent branching using a pseudo node: "IF:<condition>"
   - Only use IF when BOTH branches call different real functions.

=====================
ROUTING CHAIN RULES
=====================
For each endpoint, trace the request chain in this order when present in code:

ENTRY (app.mjs)
→ ROUTER_MOUNT (app.use base path)
→ ROUTE_HANDLER (router.METHOD path)
→ Middleware (in declared order)
→ Controller function
→ Data/Repository function(s)
→ (DB is implicit via mongoose methods inside data layer; do not invent DB node unless there is a named function)

If you cannot confidently identify the ENTRY or ROUTER_MOUNT from code, omit those pseudo nodes rather than guessing.

=====================
PATH CONSISTENCY RULES
=====================
- Router mount paths MUST exactly match app.use paths (e.g., app.use("/auth", authRouter)).
- Endpoint URLs MUST be mountPath + routerPath.
- Do NOT add/remove trailing slashes. "/post" != "/post/"

=====================
NODE MODEL
=====================
All nodes MUST follow this exact shape:

{
  "function": "<name>",
  "file": "<relative path>",
  "description": "<factual description>",
  "children": []
}

Allowed pseudo-node prefixes (only when verifiable):
- ENTRY:
- ROUTER_MOUNT:
- ROUTE_HANDLER:
- IF:

=====================
OUTPUT SCHEMA (MUST MATCH)
=====================
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
          "description": "Authenticate user and return JWT token",
          "children": [
            {
              "function": "ENTRY:app",
              "file": "app.mjs",
              "description": "Server entry: mounts routers and middleware",
              "children": [
                {
                  "function": "ROUTER_MOUNT:/auth",
                  "file": "app.mjs",
                  "description": "Mount auth router at /auth",
                  "children": [
                    {
                      "function": "ROUTE_HANDLER:POST /auth/login",
                      "file": "router/auth.mjs",
                      "description": "Bind POST /auth/login to middleware + controller.login",
                      "children": [
                        {
                          "function": "validateLogin",
                          "file": "router/auth.mjs",
                          "description": "Request validation middleware array (treat as a single node)",
                          "children": []
                        },
                        {
                          "function": "login",
                          "file": "controller/auth.mjs",
                          "description": "Controller: verify credentials",
                          "children": [
                            {
                              "function": "findByUserid",
                              "file": "data/auth.mjs",
                              "description": "Fetch user by userid from database",
                              "children": []
                            },
                            {
                              "function": "createJwtToken",
                              "file": "controller/auth.mjs",
                              "description": "Create JWT token using user id",
                              "children": []
                            }
                          ]
                        }
                      ]
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

=====================
FINAL SELF-CHECK (SILENT)
=====================
- No invented functions.
- Middleware only if explicitly attached.
- No controller → controller chains unless explicit.
- All nodes have children.
- JSON parses successfully.
"""

# =========================
# Explanation prompt (2nd pass)
# =========================
EXPLAIN_PROMPT = r"""
You are a backend tutor for beginners.
Given the call-graph JSON of an API server, explain each endpoint in simple English.

Return ONLY valid JSON. No markdown. No commentary.

For every endpoint in the input JSON:
- oneLineSummary: a one-sentence summary of what this endpoint does.
- stepByStep: at least 6 short bullet-style steps, describing the full request flow.
- dataFlow: list of objects { "var", "from", "to", "meaning" } that describe how key data moves.
- pitfalls: 3 short bullet points about common mistakes or gotchas.
- glossary: 5 terms with explanations (for example: middleware, controller, repository, schema, JWT).

IMPORTANT:
- Use ENTRY/ROUTER_MOUNT/ROUTE_HANDLER nodes to describe routing flow clearly.
- Mention middleware arrays (validateLogin/validateSignup/validatePost) as "validation middleware".
- Do not invent details not implied by the call-graph.

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
        )

    gen_ids = out[0][model_inputs.input_ids.shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ============================================================
# Post-processing "insurance"
#   1) children 누락 자동 보정 (재귀)
#   2) controller -> controller 자동 경고 로그
#   3) 함수명 존재 간단 검사 (pseudo node 제외)
# ============================================================
def _is_routing_pseudo_node(func_name: str) -> bool:
    return isinstance(func_name, str) and func_name.startswith(
        ("ENTRY:", "ROUTER_MOUNT:", "ROUTE_HANDLER:", "IF:")
    )


def _is_controller_file(path: str) -> bool:
    if not isinstance(path, str):
        return False
    p = path.replace("\\", "/").lower()
    return "/controller/" in p or p.startswith("controller/") or p.endswith("/controller")


def _ensure_children_recursive(node: Any) -> None:
    """
    Ensure all dict nodes contain "children" as a list, recursively.
    This prevents UI crashes like `node.children.map` when LLM omits the field.
    """
    if isinstance(node, dict):
        if "children" not in node or node["children"] is None:
            node["children"] = []
        elif not isinstance(node["children"], list):
            node["children"] = []

        for child in node["children"]:
            _ensure_children_recursive(child)

    elif isinstance(node, list):
        for item in node:
            _ensure_children_recursive(item)


def _collect_warnings_and_fix(data: Dict[str, Any], context_text: str) -> List[str]:
    warnings: List[str] = []
    ctx = context_text or ""

    # 1) Fix schema
    _ensure_children_recursive(data)

    def walk(node: Dict[str, Any], parent: Optional[Dict[str, Any]] = None):
        if not isinstance(node, dict):
            return

        fn = node.get("function", "")
        file_ = node.get("file", "")

        # 2) controller -> controller warnings
        if parent and isinstance(parent, dict):
            p_fn = parent.get("function", "")
            p_file = parent.get("file", "")

            if _is_controller_file(p_file) and _is_controller_file(file_):
                if (not _is_routing_pseudo_node(str(p_fn))) and (not _is_routing_pseudo_node(str(fn))):
                    warnings.append(
                        f"[WARN controller->controller] {p_fn} ({p_file}) -> {fn} ({file_})"
                    )

        # 3) function existence check (simple insurance)
        if isinstance(fn, str) and fn and (not _is_routing_pseudo_node(fn)):
            # Too-short names tend to false alarm (e.g., "me", "id")
            if len(fn) >= 4:
                pattern = rf"\b{re.escape(fn)}\b"
                if re.search(pattern, ctx) is None:
                    warnings.append(f"[WARN function not found in code] {fn} (file={file_})")

        children = node.get("children", [])
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    walk(child, node)

    # Traverse endpoints
    api_list = data.get("api", [])
    if isinstance(api_list, list):
        for api_item in api_list:
            if not isinstance(api_item, dict):
                continue
            endpoints = api_item.get("endpoints", [])
            if not isinstance(endpoints, list):
                continue
            for ep in endpoints:
                if isinstance(ep, dict):
                    walk(ep, None)

    return warnings


def analyze_from_text(
    context_text: str,
    out_dir: str,
    cfg: Optional[AnalyzerConfig] = None
) -> Dict[str, Any]:
    """
    Take merged source-code text from the frontend and:
    1) Save it to out_dir/project_full_context.txt
    2) Ask the LLM to generate the call-graph JSON
    3) Post-process (insurance) + Save JSON + warnings
    """
    if cfg is None:
        cfg = AnalyzerConfig()

    os.makedirs(out_dir, exist_ok=True)

    # 1) Save raw context
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
                'JSON object that follows the specified "api" schema.\n\n'
                f"{prompt_text}"
            ),
        },
    ]

    raw = _generate_once(tokenizer, model, messages, cfg)

    # 4) Parse
    json_str = _extract_json(raw)
    data = json.loads(json_str)

    # 4.5) Post-process insurance
    warnings = _collect_warnings_and_fix(data, context_text)
    warn_path = os.path.join(out_dir, WARN_FILE)
    with open(warn_path, "w", encoding="utf-8") as wf:
        wf.write("\n".join(warnings))

    # 5) Save JSON
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
        {"role": "user", "content": json.dumps(analysis_json, ensure_ascii=False)[:80000]},
    ]

    raw = _generate_once(tokenizer, model, messages, cfg)

    json_str = _extract_json(raw)
    data = json.loads(json_str)

    out_path = os.path.join(out_dir, "project_explain.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data
