import os
import asyncio
import json
import base64
import time
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from google import genai
from google.genai import types

# =====================================================
# ENV CHECK
# =====================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# =====================================================
# APP INIT
# =====================================================
app = FastAPI(title="AI Tutor Backend", version="3.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# RATE LIMITER (per IP, per endpoint)
# =====================================================

class RateLimiter:
    """
    Simple in-memory rate limiter. Tracks requests per IP per endpoint.
    Resets automatically after the window expires.
    """

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
    "ask"            : (15, 60),    # 15 AI queries per minute per IP
    "generate_image" : (5,  60),    # 5 image generations per minute per IP
    "login"          : (5,  900),   # 5 attempts per 15 minutes per IP
    "signup"         : (3,  3600),  # 3 signups per hour per IP
    "reset_password" : (3,  3600),  # 3 resets per hour per IP
}

def check_rate_limit(request: Request, endpoint: str):
    ip = limiter.get_ip(request)
    max_req, window = LIMITS[endpoint]
    allowed, retry_after = limiter.is_allowed(f"{endpoint}:{ip}", max_req, window)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Too many requests. Please slow down.",
                "retry_after_seconds": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )

# =====================================================
# DAILY TOKEN TRACKER (per Supabase user)
# =====================================================

DAILY_TOKEN_LIMIT = 50_000      # tokens per user per day
WARNING_THRESHOLDS = [50, 75, 90]  # warn at these % of daily limit

class DailyTokenTracker:
    """
    Tracks token usage per Supabase user ID, resets at UTC midnight.

    Storage format:
    {
        "user_id": {
            "date": "2026-04-05",      # UTC date — resets when this changes
            "tokens_used": 12400,
            "warned_at": [50, 75]      # thresholds already notified
        }
    }

    Note: In-memory only — resets if Render restarts.
    To persist across restarts, write tokens_used to a Supabase table.
    """

    def __init__(self):
        self._store: dict[str, dict] = {}

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _get_record(self, user_id: str) -> dict:
        today = self._today()
        record = self._store.get(user_id)

        # New user or new UTC day — reset
        if not record or record["date"] != today:
            self._store[user_id] = {
                "date": today,
                "tokens_used": 0,
                "warned_at": [],
            }

        return self._store[user_id]

    def get_usage(self, user_id: str) -> dict:
        record = self._get_record(user_id)
        tokens_used = record["tokens_used"]
        percent = round((tokens_used / DAILY_TOKEN_LIMIT) * 100, 1)

        return {
            "tokens_used": tokens_used,
            "tokens_limit": DAILY_TOKEN_LIMIT,
            "tokens_remaining": max(0, DAILY_TOKEN_LIMIT - tokens_used),
            "percent_used": percent,
        }

    def is_limit_reached(self, user_id: str) -> bool:
        record = self._get_record(user_id)
        return record["tokens_used"] >= DAILY_TOKEN_LIMIT

    def add_tokens(self, user_id: str, count: int) -> list[int]:
        """
        Adds tokens used for a request.
        Returns list of NEW warning thresholds just crossed e.g. [50] or [75, 90].
        """
        record = self._get_record(user_id)
        old_percent = (record["tokens_used"] / DAILY_TOKEN_LIMIT) * 100
        record["tokens_used"] = min(record["tokens_used"] + count, DAILY_TOKEN_LIMIT)
        new_percent = (record["tokens_used"] / DAILY_TOKEN_LIMIT) * 100

        new_warnings = []
        for threshold in WARNING_THRESHOLDS:
            if old_percent < threshold <= new_percent and threshold not in record["warned_at"]:
                record["warned_at"].append(threshold)
                new_warnings.append(threshold)

        return new_warnings


token_tracker = DailyTokenTracker()


def extract_user_id(request: Request) -> str | None:
    """
    Extracts Supabase user ID from the Authorization header.
    Header: "Bearer <supabase_jwt>"
    Decodes the JWT payload (no signature check needed — just reading sub claim).
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ", 1)[1]

    try:
        payload_b64 = token.split(".")[1]
        # Pad base64 to multiple of 4
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(base64.b64decode(payload_b64).decode("utf-8"))
        return payload.get("sub")  # Supabase user ID lives in "sub"
    except Exception:
        return None


def estimate_tokens(text: str) -> int:
    """~4 characters per token — good enough for daily budgeting."""
    return max(1, len(text) // 4)


# =====================================================
# CONFIG
# =====================================================
ALLOWED_BOARDS = {"ICSE", "CBSE", "SSLC"}

SUPPORTED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf",
    "text/plain",
}

client = genai.Client(api_key=GEMINI_API_KEY)

# =====================================================
# SUBJECT-SPECIFIC PROMPTS
# =====================================================

SUBJECT_PROMPTS = {
    "Maths": """
Show working steps. Highlight final answer: $x = 5$
""",
    "Physics": """
Format: Given -> Formula -> Calculation -> $Answer with units$
Highlight key formulas and definitions.
""",
    "Chemistry": """
Balance equations. Use -> for reactions. Example: 2H_2 + O_2 -> 2H_2O
Highlight important reactions and definitions.
""",
    "Biology": """
Highlight key definitions: $Mitosis is the process of cell division$
Explain processes step by step.
""",
    "English Literature": """
Reference the text. Highlight key themes: $The poem explores the theme of loss and longing$
""",
    "English Grammar": """
State the rule. Highlight correct forms: $The passive voice is: The cake was eaten by him$
""",
    "History": """
Include dates. Highlight key facts: $The French Revolution began in 1789$
""",
    "Economics": """
Define terms. Highlight definitions: $GDP is the total value of goods and services produced$
""",
    "Geography": """
Include location context. Highlight key facts: $The Himalayas are young fold mountains$
""",
    "Computer Applications": """
Write code in plain text. Highlight syntax: $int x = 5;$
"""
}

DEFAULT_SUBJECT_PROMPT = """
Explain clearly. Highlight key points with $...$
"""

# =====================================================
# BASE PROMPT
# =====================================================
def get_base_prompt(board, class_level, subject, chapter, question):
    return f"""You are a {board} Class {class_level} {subject} teacher.

Student asked: \"\"\"{question}\"\"\"

FORMATTING RULES:
- Plain text only. No LaTeX, no Markdown, no HTML.
- Powers: use ^ like x^2, a^3, r^2
- Subscripts: use _ like H_2O, a_n, x_1  
- Arrows: use -> for reactions
- Greek letters: write as words (pi, theta, alpha)
- Square root: write as sqrt() like sqrt(2)
- Fractions: write as a/b or (a+b)/c
- Multiply: use x or just write together (2 x pi x r or 2pir)

HIGHLIGHTING with $ signs:
- Wrap KEY formulas: $x = (-b + sqrt(b^2 - 4ac)) / 2a$
- Wrap KEY definitions: $Photosynthesis is the process by which plants make food using sunlight$
- Wrap FINAL answers: $The answer is 25 cm$
- Wrap IMPORTANT points: $This is a very common exam question$
- Use sparingly - only the most important stuff

Keep answer short, clear, exam-focused for {board} Class {class_level}.
"""

# =====================================================
# HEALTH ROUTES
# =====================================================
@app.get("/")
def root():
    return {"status": "running", "version": "3.4"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "api_key_present": bool(GEMINI_API_KEY),
        "timestamp": datetime.utcnow().isoformat()
    }

# =====================================================
# TOKEN USAGE ROUTE
# Call this on page load to show the user their current quota.
# =====================================================
@app.get("/api/token-usage")
async def get_token_usage(request: Request):
    user_id = extract_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    return token_tracker.get_usage(user_id)

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
            config=types.UploadFileConfig(
                mime_type=mime_type,
                display_name=name,
            )
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

    # ── IP rate limit ──
    check_rate_limit(request, "ask")

    # ── Require authenticated user ──
    user_id = extract_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # ── Block if daily limit already reached ──
    if token_tracker.is_limit_reached(user_id):
        usage = token_tracker.get_usage(user_id)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_limit_reached",
                "message": "You've used your daily question limit. It resets at midnight UTC.",
                "tokens_used": usage["tokens_used"],
                "tokens_limit": usage["tokens_limit"],
                "resets_at": "midnight UTC",
            }
        )

    board        = (payload.get("board")        or "ICSE").strip().upper()
    class_level  = (payload.get("class_level")  or "10").strip()
    subject      = (payload.get("subject")      or "General").strip()
    chapter      = (payload.get("chapter")      or "General").strip()
    question     = (payload.get("question")     or "").strip()
    model_choice = (payload.get("model")        or "t1").lower()
    files        =  payload.get("files")        or []

    if board not in ALLOWED_BOARDS:
        raise HTTPException(status_code=400, detail="Invalid board")

    if not question and not files:
        raise HTTPException(status_code=400, detail="Question or file required")

    if not question and files:
        question = "Please analyse this and answer any questions based on it."

    # ── Model select ──
    if model_choice == "t2":
        model_name = "gemini-3.1-pro-preview"
    else:
        model_name = "gemini-3.1-flash-lite-preview"

    # ── Build prompt ──
    prompt_text = get_base_prompt(board, class_level, subject, chapter, question)
    subject_prompt = SUBJECT_PROMPTS.get(subject, DEFAULT_SUBJECT_PROMPT)
    prompt_text += "\n" + subject_prompt

    # ── Estimate input tokens ──
    input_token_estimate = estimate_tokens(prompt_text + question)

    # ── Process files ──
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

    # ── Stream response ──
    async def stream():
        output_chars = 0

        try:
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
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

            # ── Tally total tokens and check warning thresholds ──
            output_token_estimate = estimate_tokens(" " * output_chars)
            total_tokens = input_token_estimate + output_token_estimate
            new_warnings = token_tracker.add_tokens(user_id, total_tokens)
            usage = token_tracker.get_usage(user_id)

            # Send usage event — frontend listens for this to show warnings/progress
            usage_payload = {
                "type": "token_update",
                "tokens_used": usage["tokens_used"],
                "tokens_limit": usage["tokens_limit"],
                "tokens_remaining": usage["tokens_remaining"],
                "percent_used": usage["percent_used"],
                # New thresholds just crossed, e.g. [50] or [75, 90] or []
                # Frontend should show a toast/banner for each one
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
# IMAGE GENERATION ROUTE (rate limited + token tracked)
# =====================================================
@app.post("/api/generate-image")
async def generate_image(request: Request, payload: dict):

    check_rate_limit(request, "generate_image")

    user_id = extract_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if token_tracker.is_limit_reached(user_id):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_limit_reached",
                "message": "You've used your daily limit. It resets at midnight UTC.",
            }
        )

    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )

        image_bytes = response.generated_images[0].image.image_bytes
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Flat 500 token cost for image generation
        token_tracker.add_tokens(user_id, 500)

        return {
            "image": b64_image,
            "mime_type": "image/png",
        }

    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            raise HTTPException(status_code=429, detail="Image generation quota exceeded.")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {error_msg}")

# =====================================================
# LOCAL RUN
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
