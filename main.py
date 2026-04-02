import os
import asyncio
import json
import base64
from datetime import datetime

from fastapi import FastAPI, HTTPException
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
app = FastAPI(title="AI Tutor Backend", version="3.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
MATHS RULES:
- Write sqrt(x) NOT \\sqrt{{x}}
- Write pi NOT \\pi
- Write theta NOT \\theta  
- Write x^2 for squares, x^3 for cubes
- Write fractions as (a+b)/c NOT \\frac
- Show all working steps
- Highlight only final answer: $x = 5$
""",

    "Physics": """
PHYSICS RULES:
- State formula first, then substitute values
- Write units clearly: m/s^2, kg, N
- Format: Given → Formula → Calculation → Answer
- Highlight only final answer with units: $v = 20 m/s$
""",

    "Chemistry": """
CHEMISTRY RULES:
- Chemical formulas: H_2O, CO_2, H_2SO_4
- Reactions use ->: 2H_2 + O_2 -> 2H_2O
- State symbols in brackets: (s), (l), (g), (aq)
- Balance all equations
- Highlight key equations: $2Na + 2H_2O -> 2NaOH + H_2$
""",

    "Biology": """
BIOLOGY RULES:
- Use proper scientific terms
- Define key terms clearly
- Explain processes step-by-step
- Highlight only key definitions: $Photosynthesis is the process...$
""",

    "English Literature": """
ENGLISH LITERATURE RULES:
- Reference text/chapter specifically
- Use quotes where relevant
- Explain themes and literary devices
- Structure: Introduction → Analysis → Conclusion
""",

    "English Grammar": """
ENGLISH GRAMMAR RULES:
- State grammar rule clearly
- Show correct and incorrect examples
- For transformations, show step-by-step
- Highlight the correct form: $The letter was written by him$
""",

    "History": """
HISTORY & CIVICS RULES:
- Include relevant dates
- Name key figures and roles
- Explain causes and effects
- Structure chronologically
- Highlight key facts: $The treaty was signed in 1947$
""",

    "Economics": """
ECONOMICS RULES:
- Define terms precisely
- Use Indian economy examples
- For numericals: Formula → Substitution → Answer
- Describe graphs in words (axes, curves, shifts)
- Highlight formulas: $GDP = C + I + G + (X-M)$
""",

    "Geography": """
GEOGRAPHY RULES:
- Include location context
- Use geographical terms correctly
- Mention climate, vegetation, resources
- Include statistics from syllabus
""",

    "Computer Applications": """
COMPUTER APPLICATIONS RULES:
- Write code in plain text, indented properly
- Use correct Java syntax
- Explain logic before/after code
- Use camelCase for variables
- Highlight syntax: $int x = 5;$
"""
}

# Default prompt for subjects not in the list
DEFAULT_SUBJECT_PROMPT = """
GENERAL RULES:
- Explain clearly and accurately
- Use syllabus examples
- Highlight only final answers: $answer$
"""

# =====================================================
# BASE PROMPT (common for all subjects)
# =====================================================
def get_base_prompt(board, class_level, subject, chapter, question):
    return f"""You are an expert {board} Class {class_level} {subject} teacher.

Board: {board} | Class: {class_level} | Subject: {subject} | Chapter: {chapter}

Student's Question:
\"\"\"{question}\"\"\"

CRITICAL FORMATTING RULES — FOLLOW EXACTLY:

BANNED (NEVER USE):
- NO LaTeX: no \\frac, \\sqrt, \\pm, \\theta, \\pi, \\sin, \\cos, \\tan, \\times, \\div, \\cdot, \\neq, \\leq, \\geq, \\infty, \\sum, \\int, \\alpha, \\beta, \\gamma, \\delta
- NO Markdown: no *, **, #, ##, `, ```
- NO HTML tags
- NO emojis

REQUIRED FORMAT:
- Write theta as: theta
- Write pi as: pi
- Write plus/minus as: +/-
- Write square root as: sqrt(x)
- Write fractions as: (a + b) / 2 or a/b
- Write multiplication as: 2 x 3 or 2(3) or ab
- Write powers as: x^2, x^3, a^n (caret symbol)
- Write subscripts as: a_1, x_2, H_2O (underscore)
- Write chemical arrows as: -> (becomes →)

HIGHLIGHTING:
- To highlight a formula, wrap it with $ on each side
- Example: The answer is $x = 5$
- Only highlight FINAL ANSWERS, not everything
- Maximum 1-2 highlights per section

ANSWERING STYLE:
- Be friendly and conversational
- Keep it SHORT and CLEAR
- Focus on {board} Class {class_level} exam pattern
- Show step-by-step working for numericals

IMAGE/FILE HANDLING:
- If image attached, read and solve all questions
- Match {board} Class {class_level} {subject} syllabus
"""

# =====================================================
# HEALTH ROUTES
# =====================================================
@app.get("/")
def root():
    return {"status": "running", "version": "3.2"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "api_key_present": bool(GEMINI_API_KEY),
        "timestamp": datetime.utcnow().isoformat()
    }

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
async def ask_question(payload: dict):

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

    # =====================================================
    # MODEL SELECT
    # =====================================================
    if model_choice == "t2":
        model_name = "gemini-3.1-pro-preview"
    else:
        model_name = "gemini-3.1-flash-lite-preview"

    # =====================================================
    # BUILD PROMPT: Base + Subject-specific
    # =====================================================
    prompt_text = get_base_prompt(board, class_level, subject, chapter, question)
    
    # Add subject-specific prompt
    subject_prompt = SUBJECT_PROMPTS.get(subject, DEFAULT_SUBJECT_PROMPT)
    prompt_text += "\n" + subject_prompt

    # =====================================================
    # PROCESS FILES
    # =====================================================
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
                contents.append(types.Part.from_uri(
                    uri=uri,
                    mime_type="application/pdf"
                ))
                uploaded_file_uris.append(file_name)
            except Exception as e:
                try:
                    contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                except Exception:
                    file_errors.append(f"Could not process PDF '{name}': {str(e)}")

        elif mime == "text/plain":
            try:
                text_content = raw_bytes.decode("utf-8", errors="replace")
                prompt_text += f"\n\n--- Content of {name} ---\n{text_content}\n--- End of {name} ---"
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

    # =====================================================
    # STREAM RESPONSE
    # =====================================================
    async def stream():
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
                        yield f"data: {json.dumps(text)}\n\n"
                except StopIteration:
                    break
                except Exception as chunk_error:
                    print(f"Chunk error: {chunk_error}")
                    if chunk_count == 0:
                        raise chunk_error
                    break

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
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# =====================================================
# LOCAL RUN
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
