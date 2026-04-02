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
MATHS-SPECIFIC RULES:
- Show ALL working steps clearly
- Write equations using: x^2 for powers, fractions as a/b
- Use ^ for exponents: x^2, x^{{n+1}}, 10^{{-3}}
- Highlight final answers with $answer$
- For geometry: state theorems used
- For algebra: show each simplification step
- For trigonometry: use sin, cos, tan with degree symbol (30°)
- Always verify answer where possible
""",

    "Physics": """
PHYSICS-SPECIFIC RULES:
- State the relevant formula first
- List given quantities with units
- Show substitution step clearly
- Use ^ for powers: m/s^2, kg m^2, 10^{{-19}}
- Include units in final answer
- Highlight formula and final answer with $text$
- For numericals: Given → Formula → Substitution → Answer
- Mention SI units where relevant
""",

    "Chemistry": """
CHEMISTRY-SPECIFIC RULES:
- For equations: use subscripts H_2O, CO_2, Na_2SO_4
- Multi-char subscripts: C_{{6}}H_{{12}}O_{{6}}
- Charges as superscripts: Fe^{{3+}}, Cu^{{2+}}, OH^-
- Reaction arrows: use -> (renders as →)
- Always balance chemical equations
- State symbols: (s) (l) (g) (aq) after compounds
- Example: 2H_2 + O_2 -> 2H_2O
- Example: CaCO_3 -> CaO + CO_2
- Highlight important reactions/definitions with $text$
""",

    "Biology": """
BIOLOGY-SPECIFIC RULES:
- Use proper scientific terminology
- Define key terms clearly
- For diagrams: describe parts systematically
- Mention functions along with structures
- Use examples from the syllabus
- Highlight definitions and key points with $text$
- For processes: explain step-by-step in order
""",

    "English Literature": """
ENGLISH LITERATURE-SPECIFIC RULES:
- Reference the text/chapter specifically
- Include relevant quotes where helpful
- Explain themes, characters, literary devices
- Use formal analytical language
- Structure: Introduction → Analysis → Conclusion
- Highlight key quotes and terms with $text$
- Connect to broader themes when relevant
""",

    "English Grammar": """
ENGLISH GRAMMAR-SPECIFIC RULES:
- State the grammar rule clearly
- Give correct and incorrect examples
- For transformations: show step-by-step
- For tenses: name the tense used
- For voice/narration: show the conversion process
- Highlight rules and correct forms with $text$
""",

    "History": """
HISTORY & CIVICS-SPECIFIC RULES:
- Include dates and timeline where relevant
- Name key figures and their roles
- Explain causes and consequences
- For civics: quote constitutional provisions if needed
- Structure answers chronologically when appropriate
- Highlight important dates, names, events with $text$
""",

    "Economics": """
ECONOMICS-SPECIFIC RULES:
- Define economic terms precisely
- Use real-world examples from Indian economy where relevant
- For numerical problems: show formula and substitution
- Explain concepts like demand, supply, GDP, inflation clearly
- Use graphs descriptions when needed (describe axes, curves, shifts)
- Highlight key definitions, formulas, and concepts with $text$
- Connect theory to practical applications
""",

    "Geography": """
GEOGRAPHY-SPECIFIC RULES:
- Include location context where relevant
- Use geographical terminology correctly
- For map-based: describe positions clearly
- Mention climate, vegetation, resources as relevant
- Include statistical data from syllabus
- Highlight key terms and facts with $text$
""",

    "Computer Applications": """
COMPUTER APPLICATIONS-SPECIFIC RULES:
- Write code in plain text, properly indented
- For Java: use correct syntax and conventions
- Explain logic before/after code
- Mention output where helpful
- For theory: define terms precisely
- Highlight keywords, syntax, definitions with $text$
- Variable names in camelCase for Java
"""
}

# Default prompt for subjects not in the list
DEFAULT_SUBJECT_PROMPT = """
GENERAL RULES:
- Explain concepts clearly and accurately
- Use examples from the syllabus
- Structure answer logically
- Highlight important points with $text$
"""

# =====================================================
# BASE PROMPT (common for all subjects)
# =====================================================
def get_base_prompt(board, class_level, subject, chapter, question):
    return f"""You are an expert {board} Class {class_level} {subject} teacher.

Board: {board} | Class: {class_level} | Subject: {subject} | Chapter: {chapter}

Student's Question:
\"\"\"{question}\"\"\"

Answer strictly according to {board} syllabus and exam pattern.

CORE RULES:
1. PLAIN TEXT ONLY — no Markdown, HTML, LaTeX, emojis
2. Use ^ for superscripts: x^2, 10^{{-19}}, Fe^{{3+}}
3. Use _ for subscripts: H_2O, CO_2, C_{{6}}H_{{12}}O_{{6}}
4. Use -> for arrows (renders as →)
5. HIGHLIGHTING (use sparingly):
   - Wrap ONLY final answers or key formulas with $...$
   - Example: $x = 5$ or $F = ma$
   - Do NOT highlight section titles, explanations, or general text
   - Maximum 2-3 highlights per response
   - Must have both opening and closing $
6. Keep answer SHORT, CLEAR, EXAM-ORIENTED
7. Be friendly and conversational
8. Frame answer as board examiner expects

FILE/IMAGE RULES:
- Analyse any attached image/PDF carefully
- If question paper: solve ALL visible questions
- Relate content to {board} Class {class_level} {subject} syllabus
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
