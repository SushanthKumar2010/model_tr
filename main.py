import os
import sys
import asyncio
import json
import base64
import time
import hmac
import hashlib
from collections import defaultdict
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from google import genai
from google.genai import types

# =====================================================
# LOGGING HELPER
# =====================================================
# Vercel's serverless Python runtime captures stderr more reliably than stdout,
# and unbuffered writes show up faster. Use this for all RAG-critical logging.
def rag_log(msg: str):
    print(msg, file=sys.stderr, flush=True)

# =====================================================
# ENV CHECK
# =====================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables must be set")

TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY")
if not TURNSTILE_SECRET_KEY:
    raise RuntimeError("TURNSTILE_SECRET_KEY environment variable not set")

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

# =====================================================
# RAG
# =====================================================
EMBED_MODEL    = "models/gemini-embedding-001"   # MUST match ingest.py
EMBED_DIMS     = 768                             # MUST match ingest.py
TOP_K_CHUNKS   = 6                               # more chunks for better grounding
MIN_SIMILARITY = 0.5                             # gemini-embedding-001 produces lower scores than OpenAI

_gemini_rag_client = None

def _get_rag_client():
    global _gemini_rag_client
    if _gemini_rag_client is None:
        _gemini_rag_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_rag_client

async def _embed_question(question: str) -> list[float]:
    loop = asyncio.get_event_loop()
    def _run():
        result = _get_rag_client().models.embed_content(
            model=EMBED_MODEL,
            contents=[question],
            config={"output_dimensionality": EMBED_DIMS},
        )
        return result.embeddings[0].values
    return await loop.run_in_executor(None, _run)

async def _retrieve_chunks(embedding, board, class_level, subject) -> list[dict]:
    payload = {
        "query_embedding": embedding,
        "match_count":     TOP_K_CHUNKS,
        "match_board":     board or None,
        "match_class":     class_level or None,
        "match_subject":   subject or None,
    }
    rag_log(f"[RAG] POSTing to match_chunks RPC with board={board} class={class_level} subject={subject}")
    async with httpx.AsyncClient(timeout=8) as http:
        resp = await http.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_chunks",
            headers=_supabase_headers(),
            json=payload,
        )
    if resp.status_code != 200:
        rag_log(f"[RAG] RPC error {resp.status_code}: {resp.text[:500]}")
        return []
    rows = resp.json()
    rag_log(f"[RAG] RPC returned {len(rows)} raw chunks; similarities: "
            f"{[round(r.get('similarity', 0), 2) for r in rows[:5]]}")
    return [c for c in rows if c.get("similarity", 0) >= MIN_SIMILARITY]

async def build_rag_context(question, board=None, class_level=None, subject=None) -> str:
    rag_log(f"[RAG] === build_rag_context called ===")
    rag_log(f"[RAG] question={question[:80]!r}  board={board}  class={class_level}  subject={subject}")
    try:
        rag_log(f"[RAG] Step 1: embedding question...")
        embedding = await _embed_question(question)
        rag_log(f"[RAG] Step 1 OK: got embedding of dim {len(embedding)}")

        rag_log(f"[RAG] Step 2: calling match_chunks RPC...")
        chunks = await _retrieve_chunks(embedding, board, class_level, subject)
        rag_log(f"[RAG] Step 2 OK: got {len(chunks)} chunks after threshold filter")

        if not chunks:
            rag_log(f"[RAG] No chunks above threshold ({MIN_SIMILARITY}) — no context added")
            return ""
        rag_log(f"[RAG] {len(chunks)} chunks passed threshold (top: {chunks[0].get('similarity',0):.2f})")
        lines = ["[TEXTBOOK REFERENCE — use this to ground your answer]"]
        for i, c in enumerate(chunks, 1):
            lines.append(f"\n--- Source {i}: {c.get('board','')} Class {c.get('class_level','')} {c.get('subject','')} {c.get('chapter','')} ---")
            lines.append(c["content"])
        lines.append("\n[END TEXTBOOK REFERENCE]")
        return "\n".join(lines)
    except Exception as e:
        import traceback
        rag_log(f"[RAG] Exception: {type(e).__name__}: {e}")
        rag_log(f"[RAG] Traceback:\n{traceback.format_exc()}")
        return ""

# =====================================================
# APP INIT
# =====================================================
app = FastAPI(title="AI Tutor Backend", version="3.11")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://teengro.in", "https://teengro.vercel.app", "http://localhost", "http://localhost:3000", "http://127.0.0.1"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# RATE LIMITER
# =====================================================

class RateLimiter:
    def __init__(self):
        self._store: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> tuple[bool, int]:
        now = time.time()
        self._store[key] = [t for t in self._store[key] if now - t < window_seconds]
        if len(self._store[key]) >= max_requests:
            oldest = self._store[key][0]
            retry_after = int(window_seconds - (now - oldest)) + 1
            return False, retry_after
        self._store[key].append(now)
        return True, 0

    def get_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


limiter = RateLimiter()

LIMITS = {
    "ask"            : (15, 60),
    "generate_image" : (5,  60),
    "login"          : (5,  900),
    "signup"         : (3,  3600),
    "reset_password" : (3,  3600),
}

def check_rate_limit(request: Request, endpoint: str):
    ip = limiter.get_ip(request)
    max_req, window = LIMITS[endpoint]
    allowed, retry_after = limiter.is_allowed(f"{endpoint}:{ip}", max_req, window)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"error": "Too many requests. Please slow down.", "retry_after_seconds": retry_after},
            headers={"Retry-After": str(retry_after)},
        )

# =====================================================
# DAILY TOKEN TRACKER
# =====================================================

DEFAULT_TOKEN_LIMIT = 50_000
WARNING_THRESHOLDS  = [50, 75, 90, 100]
_token_cache: dict[str, dict] = {}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _supabase_headers() -> dict:
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }


async def _load_from_supabase(user_id: str) -> dict:
    today = _today()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                headers=_supabase_headers(),
                params={
                    "select": "daily_tokens_used,daily_tokens_date,daily_token_limit,plan,plan_expires_at",
                    "id": f"eq.{user_id}",
                    "limit": "1",
                },
            )
        if resp.status_code == 200:
            rows = resp.json()
            if rows:
                row = rows[0]
                db_date = (row.get("daily_tokens_date") or "")[:10]
                tokens_used = row.get("daily_tokens_used") or 0
                token_limit = row.get("daily_token_limit") or DEFAULT_TOKEN_LIMIT
                plan = row.get("plan") or "free"
                plan_expires_at = row.get("plan_expires_at")
                if plan != "free" and plan_expires_at:
                    expires = datetime.fromisoformat(plan_expires_at.replace("Z", "+00:00"))
                    if datetime.now(timezone.utc) > expires:
                        plan = "free"
                tokens_for_today = tokens_used if db_date == today else 0
                return {"date": today, "tokens_used": tokens_for_today, "token_limit": token_limit, "plan": plan, "warned_at": []}
    except Exception as e:
        print(f"[Token] Supabase load error for {user_id}: {e}")
    return {"date": today, "tokens_used": 0, "token_limit": DEFAULT_TOKEN_LIMIT, "plan": "free", "warned_at": []}


async def _save_to_supabase(user_id: str, tokens_used: int) -> None:
    today = _today()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                headers=_supabase_headers(),
                params={"id": f"eq.{user_id}"},
                json={"daily_tokens_used": tokens_used, "daily_tokens_date": today},
            )
    except Exception as e:
        print(f"[Token] Supabase save error for {user_id}: {e}")


async def get_token_record(user_id: str) -> dict:
    today = _today()
    record = await _load_from_supabase(user_id)
    cached = _token_cache.get(user_id)
    if cached and cached["date"] == today:
        record["warned_at"] = cached.get("warned_at", [])
    _token_cache[user_id] = record
    return record


async def is_limit_reached(user_id: str) -> bool:
    record = await get_token_record(user_id)
    return record["tokens_used"] >= record.get("token_limit", DEFAULT_TOKEN_LIMIT)


async def get_usage(user_id: str) -> dict:
    record = await get_token_record(user_id)
    tokens_used = record["tokens_used"]
    token_limit = record.get("token_limit", DEFAULT_TOKEN_LIMIT)
    percent = round((tokens_used / token_limit) * 100, 1)
    return {
        "tokens_used": tokens_used,
        "tokens_limit": token_limit,
        "tokens_remaining": max(0, token_limit - tokens_used),
        "percent_used": percent,
        "plan": record.get("plan", "free"),
    }


async def add_tokens(user_id: str, count: int) -> list[int]:
    record = await get_token_record(user_id)
    token_limit = record.get("token_limit", DEFAULT_TOKEN_LIMIT)
    old_percent = (record["tokens_used"] / token_limit) * 100
    record["tokens_used"] = min(record["tokens_used"] + count, token_limit)
    new_percent = (record["tokens_used"] / token_limit) * 100
    new_warnings = []
    for threshold in WARNING_THRESHOLDS:
        if old_percent < threshold <= new_percent and threshold not in record["warned_at"]:
            record["warned_at"].append(threshold)
            new_warnings.append(threshold)
    asyncio.create_task(_save_to_supabase(user_id, record["tokens_used"]))
    return new_warnings


# =====================================================
# AUTH HELPERS
# =====================================================

def _b64url_decode(s: str) -> bytes:
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s)


def extract_user_id(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ", 1)[1]
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        if SUPABASE_JWT_SECRET:
            signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
            expected_sig = hmac.new(
                SUPABASE_JWT_SECRET.encode("utf-8"),
                signing_input,
                hashlib.sha256,
            ).digest()
            actual_sig = _b64url_decode(sig_b64)
            if not hmac.compare_digest(expected_sig, actual_sig):
                print("[JWT] Signature verification FAILED — token rejected")
                return None
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        exp = payload.get("exp")
        if exp and datetime.now(timezone.utc).timestamp() > exp:
            print("[JWT] Token expired — rejected")
            return None
        return payload.get("sub")
    except Exception as e:
        print(f"[JWT] Error: {e}")
        return None


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# =====================================================
# CONFIG
# =====================================================
ALLOWED_BOARDS = {"ICSE", "CBSE", "NCERT"}

SUPPORTED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf",
    "text/plain",
}

import importlib.metadata
try:
    _genai_version = importlib.metadata.version("google-genai")
except Exception:
    _genai_version = "unknown"
print(f"[Startup] google-genai version: {_genai_version}")

client = genai.Client(api_key=GEMINI_API_KEY)

# =====================================================
# SYSTEM INSTRUCTION
# =====================================================
SYSTEM_INSTRUCTION = (
    "You are Teengro, a friendly AI tutor for Indian school students. "
    "FORMATTING RULES — follow these without exception: "
    "NEVER use **asterisks** or markdown bold (**, __, ##, >). "
    "The ONLY way to highlight text is with $dollar signs$ like this: $Answer: x = 3$. "
    "If you feel like typing **, type $ instead. Asterisks are completely forbidden. "
    "RESPONSE STYLE — follow these without exception: "
    "Never open with filler phrases like 'Sure!', 'Great question!', 'Of course!', "
    "'I'd be happy to help', 'Certainly!', or any introductory sentence. "
    "Start the answer immediately and directly."
)

# =====================================================
# SUBJECT-SPECIFIC PROMPTS
# =====================================================

SUBJECT_PROMPTS = {
    "Maths": (
        "Show step-by-step working in plain text (e.g. 2 x 3 = 6, x^2 + 5x + 6 = 0). "
        "Highlight final answer like $Answer: x = 3$. Break each step onto its own line."
    ),
    "Physics": (
        "Format: Given → Formula → Calculation → Answer with units. "
        "Write formulas in plain text (e.g. F = m x a). "
        "Highlight key formula like $F = m x a$ and final answer like $Answer: 10 N$."
    ),
    "Chemistry": (
        "Balance equations. Use -> for reactions (e.g. 2H_2 + O_2 -> 2H_2O). "
        "Write subscripts with _ (e.g. H_2O). "
        "Highlight reactions like $2H_2 + O_2 -> 2H_2O$."
    ),
    "Biology": (
        "Highlight key definitions like $Mitosis is cell division$. "
        "Explain step by step in simple language."
    ),
    "English Literature": (
        "Reference the text. Highlight themes like $The poem explores loss$. "
        "Use clear language a student can reproduce in an exam."
    ),
    "English Grammar": (
        "State the rule first, then give a simple example. "
        "Highlight correct forms like $Passive: The cake was eaten by him$."
    ),
    "History": (
        "Include dates and context. Highlight key facts like $French Revolution began in 1789$."
    ),
    "Economics": (
        "Define terms first. Highlight like $GDP is total value of goods and services produced$."
    ),
    "Geography": (
        "Include location context. Highlight like $Himalayas are young fold mountains$."
    ),
    "Computer Applications": (
        "Write code with proper indentation. Highlight syntax like $int x = 5;$. "
        "Explain each line simply."
    ),
}

DEFAULT_SUBJECT_PROMPT = "Explain clearly. Highlight key points like $this$."

# =====================================================
# BASE PROMPT
# =====================================================
def get_base_prompt(board, class_level, subject, chapter, question, model_choice="t1"):
    board_context = {
        "ICSE": "CISCE curriculum — detailed, descriptive answers.",
        "CBSE": "follows NCERT textbooks — conceptual clarity.",
        "NCERT": "standard national curriculum used across India.",
    }.get(board, board)

    subject_hint = SUBJECT_PROMPTS.get(subject, DEFAULT_SUBJECT_PROMPT)

    if model_choice == "t2":
        mode_instruction = (
            "Give a thorough, step-by-step explanation. "
            "Cover the concept fully — explain the why, not just the what. "
            "Include examples, common mistakes to avoid, and an exam tip if relevant."
        )
    else:
        mode_instruction = (
            "Give a short, direct answer. "
            "Only what's needed to answer the question — no extra context unless essential. "
            "If it's a definition, one clear sentence. If it's a calculation, just the steps."
        )

    return f"""You are Teengro, a friendly AI tutor for Indian school students.

STUDENT: Board={board} ({board_context}) | Class={class_level} | Subject={subject} | Chapter={chapter}
QUESTION: \"\"\"{question}\"\"\"

MODE: {mode_instruction}

BOARD ACCURACY (mandatory):
- Answer must match {board} Class {class_level} syllabus exactly.
- CBSE → NCERT approach. ICSE → CISCE descriptive style. NCERT → standard national approach.
- Never mix boards. Never go above or below Class {class_level} level.
- Use only terminology and formulas from {board} Class {class_level} textbooks.

OFF-TOPIC GUARDRAIL:
- If the question is unrelated to studies, give a short friendly reply then redirect:
  "Anyway, back to {subject} — you've got {board} Class {class_level} to conquer!"
- Never lecture or shame. If you can link the topic to syllabus (e.g. cricket → statistics), do it.

SIMPLICITY:
- Plain simple English a Class {class_level} student can follow.
- Explain what a formula means before using it.
- Numbered steps. No walls of text.
- Math in plain text: powers as x^2, subscripts as H_2O, multiply as x, divide as /, sqrt(), fractions as a/b.
- Write "pi" not π, "theta" not θ.
- If you have more than 3 lines of equations, stop and explain in words first.

HIGHLIGHTING (use $ sparingly — max 2-3 highlights per answer):
- Key formulas: $F = m x a$
- Definitions: $Photosynthesis converts light to chemical energy$
- Final answers: $Answer: x = 5$
- Exam tips: $Common 3-mark question in {board} exams$

SUBJECT FORMAT: {subject_hint}"""

# =====================================================
# HEALTH ROUTES
# =====================================================
@app.get("/")
def root():
    return {"status": "running", "version": "3.11"}

@app.get("/health")
def health():
    return {"status": "healthy", "api_key_present": bool(GEMINI_API_KEY), "timestamp": datetime.utcnow().isoformat()}

# =====================================================
# TOKEN USAGE ROUTE
# =====================================================
@app.get("/api/token-usage")
async def get_token_usage(request: Request):
    user_id = extract_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return await get_usage(user_id)

# =====================================================
# TURNSTILE VERIFICATION
# =====================================================
@app.post("/api/verify-turnstile")
async def verify_turnstile(request: Request, payload: dict):
    token = (payload.get("token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Turnstile token required")
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.post(
                "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                data={"secret": TURNSTILE_SECRET_KEY, "response": token},
            )
        result = resp.json()
        return {"success": result.get("success", False)}
    except Exception as e:
        print(f"[Turnstile] Verification error: {e}")
        return {"success": False}

# =====================================================
# AUTH ROUTES (rate limited)
# =====================================================
@app.post("/api/login")
async def login(request: Request, payload: dict):
    check_rate_limit(request, "login")
    return {"message": "Login route reached"}

@app.post("/api/signup")
async def signup(request: Request, payload: dict):
    check_rate_limit(request, "signup")
    return {"message": "Signup route reached"}

@app.post("/api/reset-password")
async def reset_password(request: Request, payload: dict):
    check_rate_limit(request, "reset_password")
    return {"message": "Reset password route reached"}

# =====================================================
# FILE UPLOAD HELPER
# =====================================================
async def upload_file_to_gemini(raw_bytes: bytes, mime_type: str, name: str):
    import io
    try:
        file_obj = io.BytesIO(raw_bytes)
        file_obj.name = name
        uploaded = client.files.upload(
            file=file_obj,
            config=types.UploadFileConfig(mime_type=mime_type, display_name=name)
        )
        max_attempts = 30
        for attempt in range(max_attempts):
            if uploaded.state.name == "ACTIVE":
                return uploaded.uri, uploaded.name
            if uploaded.state.name == "FAILED":
                raise Exception(f"File processing failed: {name}")
            await asyncio.sleep(1)
            uploaded = client.files.get(name=uploaded.name)
        raise Exception(f"File processing timeout for: {name}")
    except Exception as e:
        raise Exception(f"Failed to upload {name}: {str(e)}")

# =====================================================
# MAIN ASK ROUTE
# =====================================================
@app.post("/api/ask")
async def ask_question(request: Request, payload: dict):

    check_rate_limit(request, "ask")
    user_id = extract_user_id(request)

    if user_id and await is_limit_reached(user_id):
        usage = await get_usage(user_id)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_limit_reached",
                "message": "You've used your daily question limit. It resets at midnight UTC.",
                "tokens_used": usage["tokens_used"],
                "tokens_limit": usage["tokens_limit"],
                "tokens_remaining": usage["tokens_remaining"],
                "resets_at": "midnight UTC",
            }
        )

    board        = (payload.get("board")        or "CBSE").strip().upper()
    class_level  = (payload.get("class_level")  or "10").strip()
    subject      = (payload.get("subject")      or "General").strip()
    chapter      = (payload.get("chapter")      or "General").strip()
    question     = (payload.get("question")     or "").strip()
    model_choice = (payload.get("model")        or "t1").lower()
    files        =  payload.get("files")        or []

    rag_log(f"[ASK] Incoming: board={board} class={class_level} subject={subject} q={question[:80]!r}")

    if board == "SSLC":
        board = "NCERT"

    if board not in ALLOWED_BOARDS:
        raise HTTPException(status_code=400, detail=f"Invalid board. Must be one of: {', '.join(ALLOWED_BOARDS)}")

    if not question and not files:
        raise HTTPException(status_code=400, detail="Question or file required")

    if not question and files:
        question = "Please analyse this and answer any questions based on it."

    record = await get_token_record(user_id) if user_id else {"plan": "free"}
    user_plan = record.get("plan", "free")

    if model_choice == "t2" and user_plan == "pro":
        model_name = "gemini-3.1-pro-preview"
    else:
        model_name = "gemini-3.1-flash-lite-preview"

    # ── RAG: fetch relevant textbook chunks ──
    rag_context = await build_rag_context(question, board, class_level, subject)
    rag_log(f"[ASK] RAG context length: {len(rag_context)} chars")
    prompt_text = rag_context + "\n\n" + get_base_prompt(board, class_level, subject, chapter, question, model_choice)

    input_token_estimate = estimate_tokens(prompt_text + question)

    contents = []
    uploaded_file_uris = []
    file_errors = []

    for f in files:
        mime = f.get("mimeType", "")
        b64  = f.get("base64",   "")
        name = f.get("name",     "file")

        if not b64 or not mime:
            continue
        if mime not in SUPPORTED_MIME_TYPES:
            file_errors.append(f"File '{name}' ({mime}) is unsupported")
            continue

        try:
            raw_bytes = base64.b64decode(b64)
        except Exception as e:
            file_errors.append(f"Could not decode '{name}': {str(e)}")
            continue

        if mime == "application/pdf":
            try:
                uri, file_name = await upload_file_to_gemini(raw_bytes, "application/pdf", name)
                contents.append(types.Part.from_uri(uri=uri, mime_type="application/pdf"))
                uploaded_file_uris.append(file_name)
                input_token_estimate += estimate_tokens(raw_bytes.decode("utf-8", errors="replace"))
            except Exception as e:
                try:
                    contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                except Exception:
                    file_errors.append(f"Could not process PDF '{name}': {str(e)}")
        elif mime == "text/plain":
            try:
                text_content = raw_bytes.decode("utf-8", errors="replace")
                prompt_text += f"\n\n--- Content of {name} ---\n{text_content}\n--- End of {name} ---"
                input_token_estimate += estimate_tokens(text_content)
            except Exception as e:
                file_errors.append(f"Could not read text file '{name}': {str(e)}")
        else:
            try:
                contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
            except Exception as e:
                file_errors.append(f"Could not process image '{name}': {str(e)}")

    if file_errors:
        prompt_text += "\n\n[Note: Some files could not be processed: " + "; ".join(file_errors) + "]"

    contents.append(types.Part.from_text(text=prompt_text))

    async def stream():
        output_chars = 0
        try:
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                ),
            )
            loop = asyncio.get_event_loop()

            def get_next():
                return next(response_stream, None)

            chunk_count = 0
            while True:
                try:
                    chunk = await loop.run_in_executor(None, get_next)
                    if chunk is None:
                        break
                    text = chunk.text or ""
                    if text:
                        chunk_count += 1
                        output_chars += len(text)
                        yield f"data: {json.dumps(text)}\n\n"
                except StopIteration:
                    break
                except Exception as chunk_error:
                    print(f"Chunk error: {chunk_error}")
                    if chunk_count == 0:
                        raise chunk_error
                    break

            output_token_estimate = estimate_tokens(" " * output_chars)
            total_tokens = input_token_estimate + output_token_estimate

            if user_id:
                new_warnings = await add_tokens(user_id, total_tokens)
                usage = await get_usage(user_id)
                usage_payload = {
                    "type": "token_update",
                    "tokens_used": usage["tokens_used"],
                    "tokens_limit": usage["tokens_limit"],
                    "tokens_remaining": usage["tokens_remaining"],
                    "percent_used": usage["percent_used"],
                    "new_warnings": new_warnings,
                }
                yield f"event: usage\ndata: {json.dumps(usage_payload)}\n\n"

            yield "event: end\ndata: done\n\n"

        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower():
                yield f"event: error\ndata: API quota exceeded. Please try again later.\n\n"
            elif "api key" in error_msg.lower():
                yield f"event: error\ndata: API configuration error. Please contact support.\n\n"
            else:
                yield f"event: error\ndata: {error_msg}\n\n"
        finally:
            for file_name in uploaded_file_uris:
                try:
                    client.files.delete(name=file_name)
                except:
                    pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# =====================================================
# IMAGE GENERATION ROUTE
# =====================================================
@app.post("/api/generate-image")
async def generate_image(request: Request, payload: dict):

    check_rate_limit(request, "generate_image")
    user_id = extract_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if await is_limit_reached(user_id):
        raise HTTPException(
            status_code=429,
            detail={"error": "daily_limit_reached", "message": "You've used your daily limit. It resets at midnight UTC."}
        )

    raw_prompt   = (payload.get("prompt")      or "").strip()
    board        = (payload.get("board")       or "CBSE").strip().upper()
    class_level  = (payload.get("class_level") or "10").strip()
    subject      = (payload.get("subject")     or "General").strip()

    if board == "SSLC":
        board = "NCERT"

    if not raw_prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    engineered_prompt = f"""
Create an accurate educational diagram for {board} Class {class_level} {subject}.
Topic: {raw_prompt}

CRITICAL TEXT ACCURACY REQUIREMENTS (most important):
- Every single word, label, formula, and term must be spelled PERFECTLY and be scientifically correct
- Double-check all spelling before rendering — no typos allowed whatsoever
- Use only standard, correct terminology as taught in {board} Class {class_level} textbooks
- All mathematical symbols, formulas, and equations must be 100% accurate
- Labels must use the exact correct English spelling (e.g. "Hypotenuse" not "Hypotunse", "Adjacent" not "Adjecent")

Style requirements:
- Clean, minimal, flat design — like a professional textbook or Khan Academy illustration
- White background
- Clear, readable labels in simple sans-serif font
- Crisp lines and shapes, no textures or noise
- Muted colours (blues, teals, soft greens) — no neon or gradients
- No photorealism, no 3D rendering, no shading
- Include a clear, correctly spelled title and all important labels
- Plenty of whitespace, uncluttered layout
""".strip()

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=engineered_prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
        )

        image_bytes = None
        mime_type = "image/png"
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_bytes = part.inline_data.data
                mime_type = part.inline_data.mime_type or "image/png"
                break

        if not image_bytes:
            raise Exception("No image returned in response")

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        if user_id:
            await add_tokens(user_id, 500)

        return {"image": b64_image, "mime_type": mime_type}

    except Exception as e:
        error_msg = str(e)
        print(f"[ImageGen] Error: {error_msg}")
        if "quota" in error_msg.lower() or "429" in error_msg:
            raise HTTPException(status_code=429, detail="Image generation quota exceeded.")
        if "safety" in error_msg.lower() or "block" in error_msg.lower():
            raise HTTPException(status_code=400, detail="Image blocked by safety filter. Try rephrasing your request.")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {error_msg}")

# =====================================================
# LOCAL RUN
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
