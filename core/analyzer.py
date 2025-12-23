# project_root/core/analyzer.py
import os
import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

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
- CRITICAL JSON RULES:
  * NO trailing commas (e.g., "key": "value",} is WRONG, use "key": "value"})
  * JSON property keys and string values must use double quotes
  * Escape special characters in strings: \n for newline, \" for quote, \\ for backslash
  * Ensure all brackets and braces are properly closed
  * If a string contains a newline, use \n not an actual newline character
  * MOST IMPORTANT: When including code in the "code" field, use SINGLE QUOTES for strings inside the code
    This prevents JSON parsing errors from unescaped double quotes.
    Example WRONG: "code": "return res.status(404).json({ message: \"일치하는 사용자가 없음\" });"
    Example CORRECT: "code": "return res.status(404).json({ message: '일치하는 사용자가 없음' });"
    Example WRONG: "code": "const token = authHeader.split(\" \")[1];"
    Example CORRECT: "code": "const token = authHeader.split(' ')[1];"
  * In JavaScript code strings, prefer single quotes for string literals to avoid JSON escaping issues
  * Only escape double quotes if you absolutely must use them in the code string

- CRITICAL: You MUST find and analyze ALL API endpoints in the provided code.
  - Search for ALL router.METHOD calls (router.get, router.post, router.put, router.delete, router.patch, etc.)
  - Each unique endpoint (method + url combination) MUST be included as a separate entry
  - Do NOT skip any endpoints - analyze every single route handler you find
  - Group endpoints by category (e.g., "auth", "post") based on their router mount path

- EVERY endpoint MUST include:
  "method", "url", "mainFlow", "detailFunctions"

- CRITICAL:
  - "mainFlow" contains the routing chain (app.mjs → router → controller) - this is the LEFT side visualization
  - "detailFunctions" contains all detailed functions (middleware, repository, helper functions) - this is the RIGHT side visualization
  - EVERY function MUST include "code" field with the ACTUAL source code from the provided text
  - Extract the EXACT code snippet for each function from the provided code text

=====================
TRACE ACCURACY RULES
=====================
1) ONLY output functions/middleware/files that are explicitly present in the provided code text.
   - If a function name does NOT appear verbatim in the code text, DO NOT output it.
   - Do NOT invent helpers like "findUserid" or "getUserFromReq" if they don't exist.
2) Extract the EXACT code for each function from the provided code text.
   - For functions, extract the complete function definition including body.
   - For middleware arrays, extract the complete array definition.
   - For router mounts, extract the exact line like "app.use(\"/auth\", authRouter);"
3) Middleware:
   - Only include middleware if it is explicitly attached on that route line.
   - Middleware arrays (e.g., validateSignup, validatePost) MUST be included in detailFunctions with full code.
4) Do NOT fabricate controller → controller calls.
5) Conditional logic (if / ternary) is NOT a function call.

=====================
ROUTING CHAIN RULES (mainFlow)
=====================
For each endpoint, trace the request chain in this order when present in code:

1. app.mjs - router mount (e.g., "app.use('/auth', authRouter);")
2. router file - route handler (e.g., "router.post('/signup', validateSignup, authController.signup);")
3. controller file - controller function (the main handler function)

IMPORTANT: In all code strings (in "code" field), use SINGLE QUOTES for JavaScript string literals.
This prevents JSON parsing errors. Example:
- WRONG: "code": "router.post(\"/signup\", ...)"
- CORRECT: "code": "router.post('/signup', ...)"
- WRONG: "code": "return res.status(404).json({ message: \"일치하는 사용자가 없음\" });"
- CORRECT: "code": "return res.status(404).json({ message: '일치하는 사용자가 없음' });"

mainFlow should be a flat array showing the routing chain, NOT nested.

=====================
DETAIL FUNCTIONS (detailFunctions)
=====================
Include ALL functions that are called or referenced in the main flow:
- Middleware functions (validateSignup, validateLogin, isAuth, validate, etc.)
- Repository/Data functions (findByUserid, createUser, getById, etc.)
- Helper functions (createJwtToken, etc.)

For each function in detailFunctions:
- Extract the COMPLETE function code from the provided text
- Include the function name, parameters, body, and return statement
- For middleware arrays, include the complete array definition

=====================
CODE EXTRACTION RULES
=====================
- Extract code EXACTLY as it appears in the provided text
- Include function signature, body, and all code
- For multi-line functions, include all lines
- Preserve indentation and formatting
- If a function is not found in the code, omit it (do not invent)

=====================
PATH CONSISTENCY RULES
=====================
- Router mount paths MUST exactly match app.use paths (e.g., app.use("/auth", authRouter)).
- Endpoint URLs MUST be mountPath + routerPath.
- Do NOT add/remove trailing slashes. "/post" != "/post/"

=====================
OUTPUT SCHEMA (MUST MATCH)
=====================
You MUST include ALL endpoints found in the code. Example structure:

{
  "api": [
    {
      "category": "auth",
      "categoryName": "Auth Feature",
      "endpoints": [
        {
          "method": "POST",
          "url": "/auth/signup",
          "mainFlow": [
            {
              "file": "app.mjs",
              "code": "app.use('/auth', authRouter);",
              "description": "Mount auth router at /auth path"
            },
            {
              "file": "router/auth.mjs",
              "code": "router.post('/signup', validateSignup, authController.signup);",
              "description": "Route POST /signup to validateSignup middleware and signup controller"
            },
            {
              "file": "controller/auth.mjs",
              "function": "signup",
              "code": "export async function signup(req, res, next) {\n  const { userid, password, name, email, url } = req.body;\n  const found = await authRepository.findByUserid(userid);\n  if (found) {\n    return res.status(409).json({ message: `${userid}이 이미 있습니다` });\n  }\n  const hashed = bcrypt.hashSync(password, config.bcrypt.saltRounds);\n  const user = await authRepository.createUser({\n    userid,\n    password: hashed,\n    name,\n    email,\n    url,\n  });\n  const token = await createJwtToken(user.id);\n  res.status(201).json({ token, userid });\n}",
              "description": "Controller: handle user signup, check duplicates, hash password, create user, return JWT token"
            }
          ],
          "detailFunctions": [
            {
              "function": "validateSignup",
              "file": "router/auth.mjs",
              "code": "const validateSignup = [\n  ...validateLogin,\n  body('name').trim().notEmpty().withMessage('name을 입력'),\n  body('email').trim().isEmail().withMessage('이메일 형식 확인'),\n  validate,\n];",
              "description": "Validation middleware array for signup: validates userid, password, name, and email"
            },
            {
              "function": "validateLogin",
              "file": "router/auth.mjs",
              "code": "const validateLogin = [\n  body('userid')\n    .trim()\n    .isLength({ min: 4 })\n    .withMessage('아이디 최소 4자이상 입력')\n    .matches(/^[a-zA-Z0-9]+$/)\n    .withMessage('아이디에 특수문자는 사용 불가'),\n  body('password')\n    .trim()\n    .isLength({ min: 4 })\n    .withMessage('비밀번호 최소 4자 이상 입력'),\n  validate,\n];",
              "description": "Validation middleware array for login: validates userid (min 4 chars, alphanumeric) and password (min 4 chars)"
            },
            {
              "function": "validate",
              "file": "middleware/validator.mjs",
              "code": "export const validate = (req, res, next) => {\n  const errors = validationResult(req);\n  if (errors.isEmpty()) {\n    return next();\n  }\n  return res.status(400).json({ message: errors.array()[0].msg });\n};",
              "description": "Middleware: check validation results, call next() if valid, return 400 error if invalid"
            },
            {
              "function": "findByUserid",
              "file": "data/auth.mjs",
              "code": "export async function findByUserid(userid) {\n  return User.findOne({ userid });\n}",
              "description": "Repository: find user by userid from database"
            },
            {
              "function": "createUser",
              "file": "data/auth.mjs",
              "code": "export async function createUser(user) {\n  return new User(user).save().then((data) => data.id);\n}",
              "description": "Repository: create new user in database and return user id"
            },
            {
              "function": "createJwtToken",
              "file": "controller/auth.mjs",
              "code": "async function createJwtToken(id) {\n  return jwt.sign({ id }, config.jwt.secretKey, {\n    expiresIn: config.jwt.expiresInSec,\n  });\n}",
              "description": "Helper: create JWT token with user id"
            }
          ]
        },
        {
          "method": "POST",
          "url": "/auth/login",
          "mainFlow": [
            {
              "file": "app.mjs",
              "code": "app.use('/auth', authRouter);",
              "description": "Mount auth router at /auth path"
            },
            {
              "file": "router/auth.mjs",
              "code": "router.post('/login', validateLogin, authController.login);",
              "description": "Route POST /login to validateLogin middleware and login controller"
            },
            {
              "file": "controller/auth.mjs",
              "function": "login",
              "code": "export async function login(req, res, next) {\n  const { userid, password } = req.body;\n  const user = await authRepository.findByUserid(userid);\n  if (!user) {\n    res.status(401).json(`${userid}를 찾을 수 없음`);\n  }\n  const isValidPassword = await bcrypt.compare(password, user.password);\n  if (!isValidPassword) {\n    return res.status(401).json({ message: '아이디 또는 비밀번호 확인' });\n  }\n  const token = await createJwtToken(user.id);\n  res.status(200).json({ token, userid });\n}",
              "description": "Controller: handle user login, check credentials, generate JWT token"
            }
          ],
          "detailFunctions": [
            {
              "function": "validateLogin",
              "file": "router/auth.mjs",
              "code": "const validateLogin = [\n  body('userid')\n    .trim()\n    .isLength({ min: 4 })\n    .withMessage('아이디 최소 4자이상 입력')\n    .matches(/^[a-zA-Z0-9]+$/)\n    .withMessage('아이디에 특수문자는 사용 불가'),\n  body('password')\n    .trim()\n    .isLength({ min: 4 })\n    .withMessage('비밀번호 최소 4자 이상 입력'),\n  validate,\n];",
              "description": "Validation middleware array for login"
            },
            {
              "function": "createJwtToken",
              "file": "controller/auth.mjs",
              "code": "async function createJwtToken(id) {\n  return jwt.sign({ id }, config.jwt.secretKey, {\n    expiresIn: config.jwt.expiresInSec,\n  });\n}",
              "description": "Helper: create JWT token with user id"
            }
          ]
        },
        {
          "method": "POST",
          "url": "/auth/me",
          "mainFlow": [
            {
              "file": "app.mjs",
              "code": "app.use('/auth', authRouter);",
              "description": "Mount auth router at /auth path"
            },
            {
              "file": "router/auth.mjs",
              "code": "router.post('/me', isAuth, authController.me);",
              "description": "Route POST /me to isAuth middleware and me controller"
            },
            {
              "file": "controller/auth.mjs",
              "function": "me",
              "code": "export async function me(req, res, next) {\n  const user = await authRepository.findById(req.id);\n  if (!user) {\n    return res.status(404).json({ message: '일치하는 사용자가 없음' });\n  }\n  res.status(200).json({ token: req.token, userid: user.userid });\n}",
              "description": "Controller: authenticate user, retrieve user details"
            }
          ],
          "detailFunctions": [
            {
              "function": "isAuth",
              "file": "middleware/auth.mjs",
              "code": "export const isAuth = async (req, res, next) => {\n  const authHeader = req.get('Authorization');\n  if (!(authHeader && authHeader.startsWith('Bearer '))) {\n    return res.status(401).json(AUTH_ERROR);\n  }\n  const token = authHeader.split(' ')[1];\n  jwt.verify(token, config.jwt.secretKey, async (error, decoded) => {\n    if (error) {\n      return res.status(401).json(AUTH_ERROR);\n    }\n    const user = await authRepository.findById(decoded.id);\n    if (!user) {\n      return res.status(401).json(AUTH_ERROR);\n    }\n    req.id = user.id;\n    next();\n  });\n};",
              "description": "Middleware: verify JWT token, set user ID in request"
            },
            {
              "function": "findById",
              "file": "data/auth.mjs",
              "code": "export async function findById(id) {\n  return User.findById(id);\n}",
              "description": "Repository: find user by ID from database"
            }
          ]
        }
          ]
        },
        {
          "method": "POST",
          "url": "/auth/login",
          "mainFlow": [...],
          "detailFunctions": [...]
        },
        {
          "method": "POST",
          "url": "/auth/me",
          "mainFlow": [...],
          "detailFunctions": [...]
        }
      ]
    },
    {
      "category": "post",
      "categoryName": "Post Feature",
      "endpoints": [
        {
          "method": "GET",
          "url": "/post",
          "mainFlow": [...],
          "detailFunctions": [...]
        },
        {
          "method": "GET",
          "url": "/post/:id",
          "mainFlow": [...],
          "detailFunctions": [...]
        },
        {
          "method": "POST",
          "url": "/post",
          "mainFlow": [...],
          "detailFunctions": [...]
        },
        {
          "method": "PUT",
          "url": "/post/:id",
          "mainFlow": [...],
          "detailFunctions": [...]
        },
        {
          "method": "DELETE",
          "url": "/post/:id",
          "mainFlow": [...],
          "detailFunctions": [...]
        }
      ]
    }
  ]
}

IMPORTANT: You must find and include ALL endpoints. Look for:
- router.get(...)
- router.post(...)
- router.put(...)
- router.delete(...)
- router.patch(...)
- Any other router.METHOD(...) calls

Each endpoint must be a separate entry in the endpoints array.

=====================
FINAL SELF-CHECK (SILENT)
=====================
- ALL endpoints from the code are included (check every router.METHOD call).
- No invented functions.
- All code snippets are extracted from provided text.
- mainFlow shows routing chain (app → router → controller).
- detailFunctions includes all middleware, repository, and helper functions.
- Every function has "code" field with actual source code.
- JSON parses successfully.
- Count the endpoints: if you find 8 router.METHOD calls, you must have 8 endpoints in the output.
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
    max_new_tokens: int = 16384  # 모든 API 엔드포인트를 포함하기 위해 충분히 큰 값
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


def _fix_unescaped_quotes_in_strings(json_str: str) -> str:
    """
    Fix unescaped quotes inside JSON string values using a more robust approach.
    This uses regex to find string values and fix quotes inside them.
    """
    # Pattern to match JSON string values: "key": "value"
    # We'll process each string value separately
    def fix_string_value(match):
        key_part = match.group(1)  # "key":
        value = match.group(2)  # the string value (without outer quotes)
        
        # Escape any unescaped quotes in the value
        # But be careful not to escape already escaped quotes
        fixed_value = re.sub(r'(?<!\\)"(?![,:\s}\]\\n])', '\\"', value)
        
        return f'{key_part}"{fixed_value}"'
    
    # Match pattern: "key": "value" where value might contain unescaped quotes
    # This is a simplified pattern - it might not catch all cases
    pattern = r'("(?:code|description|message|function|file)"\s*:\s*")([^"]*(?:"[^",}\]]*)*)"'
    
    # Try to fix common cases where quotes appear in string values
    # More aggressive: find strings that contain unescaped quotes
    result = json_str
    # This is a heuristic - find likely problematic patterns
    # Pattern: "key": "text"text" where the middle quote should be escaped
    result = re.sub(
        r'("(?:code|description|message)"\s*:\s*")([^"]*)"([^",}\]]+)"([^"]*")',
        lambda m: m.group(1) + m.group(2) + '\\"' + m.group(3) + '\\"' + m.group(4),
        result
    )
    
    return result


def _extract_json(text: str) -> str:
    """
    Extract the first JSON object from the model output by taking everything
    from the first '{' to the last '}' after stripping markdown fences.
    Also attempts to fix common JSON issues.
    """
    cleaned = re.sub(r"```(json)?|```", "", text.strip(), flags=re.MULTILINE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    
    json_str = cleaned[start : end + 1]
    
    # Try to fix common JSON issues
    # 1. Remove trailing commas before closing brackets/braces (more aggressive)
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # 2. Try to parse first
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        # If parsing fails, try to fix common issues
        # Fix unescaped quotes in string values (heuristic)
        json_str = _fix_unescaped_quotes_in_strings(json_str)
        
        # Try parsing again
        try:
            json.loads(json_str)
            return json_str
        except:
            # If still fails, return the original (will be handled by caller)
            return json_str


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
                "CRITICAL: You MUST find and analyze ALL API endpoints in the code. "
                "Search for every router.get, router.post, router.put, router.delete, router.patch call. "
                "Each unique endpoint (method + url) must be included as a separate entry. "
                "Do NOT skip any endpoints.\n\n"
                "CRITICAL FOR JSON PARSING: In all code strings (in the 'code' field), use SINGLE QUOTES for JavaScript string literals. "
                "This prevents JSON parsing errors. Convert all double quotes in code to single quotes. "
                "Example: Use 'text' instead of \"text\" in code strings.\n\n"
                f"{prompt_text}"
            ),
        },
    ]

    raw = _generate_once(tokenizer, model, messages, cfg)

    # 시간 기반 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # aimodels 디렉토리 경로 (프로젝트 루트 기준)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aimodels_dir = os.path.join(project_root, "aimodels")
    os.makedirs(aimodels_dir, exist_ok=True)
    
    # 4) Raw 출력 저장 (항상 저장)
    raw_output_path = os.path.join(aimodels_dir, f"{timestamp}_raw_output.txt")
    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"✅ Raw 출력 저장: {raw_output_path}")

    # 5) Parse with error handling and retry
    max_retries = 3
    data = None
    last_error = None
    json_str = None
    
    for attempt in range(max_retries):
        try:
            json_str = _extract_json(raw)
            # 추출된 JSON 문자열 저장
            json_str_path = os.path.join(aimodels_dir, f"{timestamp}_extracted_json.json")
            with open(json_str_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"✅ 추출된 JSON 저장: {json_str_path}")
            
            data = json.loads(json_str)
            break
        except json.JSONDecodeError as e:
            last_error = e
            print(f"❌ JSON 파싱 시도 {attempt + 1}/{max_retries} 실패: {e}")
            print(f"   에러 위치: line {e.lineno}, column {e.colno}")
            
            # 에러 위치 주변의 텍스트를 출력하여 디버깅
            if json_str:
                lines = json_str.split('\n')
                error_line_idx = e.lineno - 1
                if 0 <= error_line_idx < len(lines):
                    error_line = lines[error_line_idx]
                    print(f"   에러 라인: {error_line[:200]}")
                    if e.colno < len(error_line):
                        print(f"   에러 위치 표시: {' ' * min(e.colno - 1, 100)}^")
            
            if attempt < max_retries - 1:
                # Try to fix common issues
                try:
                    # Try to fix the JSON with more aggressive fixes
                    json_str = _extract_json(raw)
                    # Remove trailing commas more aggressively
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    
                    # Fix unescaped quotes in JSON string values
                    # This is a character-by-character approach that's more accurate
                    def fix_unescaped_quotes(text):
                        """Fix unescaped quotes inside JSON string values"""
                        result = []
                        i = 0
                        in_string = False
                        escape_next = False
                        string_start_colon = False  # Track if we're in a value (after colon)
                        
                        while i < len(text):
                            char = text[i]
                            
                            if escape_next:
                                result.append(char)
                                escape_next = False
                                i += 1
                                continue
                            
                            if char == '\\':
                                result.append(char)
                                escape_next = True
                                i += 1
                                continue
                            
                            if char == '"':
                                if not in_string:
                                    # Check if this starts a value (after colon) or a key
                                    lookback = ''.join(result[-5:]) if len(result) >= 5 else ''.join(result)
                                    if ':' in lookback:
                                        string_start_colon = True
                                    in_string = True
                                    result.append(char)
                                else:
                                    # Inside a string - determine if this is end or unescaped quote
                                    lookahead = text[i+1:min(i+10, len(text))]
                                    # If next char suggests end of string (:, ,, }, ], whitespace)
                                    if lookahead and (lookahead[0] in [':', ',', '}', ']', '\n', '\r'] or
                                                      (lookahead[0] == ' ' and len(lookahead) > 1 and lookahead[1] in [':', ',', '}', ']'])):
                                        # End of string
                                        in_string = False
                                        string_start_colon = False
                                        result.append(char)
                                    else:
                                        # Likely unescaped quote in string content
                                        # Escape it
                                        result.append('\\"')
                                i += 1
                                continue
                            
                            result.append(char)
                            i += 1
                        
                        return ''.join(result)
                    
                    # Apply the fix
                    json_str = fix_unescaped_quotes(json_str)
                    
                    # Additional pass: fix specific patterns that are common
                    # Pattern: { message: "text" } inside code strings
                    json_str = re.sub(
                        r'("code"\s*:\s*"[^"]*?\{[^}]*?message\s*:\s*)"([^"]+?)"([^}]*?\}[^"]*?")',
                        lambda m: m.group(1) + '\\"' + m.group(2) + '\\"' + m.group(3),
                        json_str,
                        flags=re.DOTALL
                    )
                    
                    # Try parsing again
                    data = json.loads(json_str)
                    print(f"✅ 자동 수정 성공!")
                    break
                except Exception as fix_error:
                    print(f"   자동 수정 시도 실패: {fix_error}")
                    continue
            else:
                # Last attempt failed, save error info
                error_path = os.path.join(aimodels_dir, f"{timestamp}_json_error.txt")
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write(f"JSON 파싱 에러:\n{str(last_error)}\n\n")
                    f.write(f"에러 위치: line {last_error.lineno}, column {last_error.colno}\n\n")
                    f.write(f"Raw output:\n{raw}\n\n")
                    if json_str:
                        f.write(f"Extracted JSON (실패):\n{json_str}\n")
                print(f"❌ JSON 파싱 실패 - 에러 정보 저장: {error_path}")
                raise ValueError(f"JSON 파싱 실패 (시도 {max_retries}회): {last_error}")
    
    if data is None:
        raise ValueError("JSON 파싱에 실패했습니다.")
    
    # 6) 파싱 성공한 JSON 저장
    parsed_json_path = os.path.join(aimodels_dir, f"{timestamp}_parsed_json.json")
    with open(parsed_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 파싱된 JSON 저장: {parsed_json_path}")

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
        cfg = AnalyzerConfig(max_new_tokens=8192)  # 설명 생성도 충분한 토큰 필요

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
