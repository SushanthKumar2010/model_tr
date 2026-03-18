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
app = FastAPI(title="AI Tutor Backend", version="4.0")

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
# FORMATTING RULES
# =====================================================

# For Maths, Physics, Chemistry, Biology
FORMATTING_RULES_SCIENCE = """
OUTPUT RULES:
- Plain text only. No Markdown, no HTML, no LaTeX.
- Superscripts: x^2, 10^{{-19}}, Fe^{{3+}}, Cu^{{2+}}
- Subscripts: H_2O, H_2SO_4, C_{{6}}H_{{12}}O_{{6}}
- Reaction arrow: -> (e.g. 2H_2 + O_2 -> 2H_2O)
- Highlight key formulas/answers with *single asterisks* only. Never use **double**.
- Be friendly and concise for a Class {class_level} student.
"""

# For English, History, Geography, Computer, General
FORMATTING_RULES_MINIMAL = """
OUTPUT RULES:
- Plain text only. No Markdown, no HTML, no LaTeX.
- Highlight key terms or final answers with *single asterisks* only. Never use **double**.
- Be friendly and concise for a Class {class_level} student.
"""

# =====================================================
# SUBJECT-SPECIFIC PROMPT BUILDERS
# =====================================================

def build_maths_prompt(board, class_level, chapter, question):
    return f"""You are an expert {board} Class {class_level} Maths teacher.

Board: {board} | Class: {class_level} | Chapter: {chapter}

Student's question: \"\"\"{question}\"\"\"

MATHS-SPECIFIC RULES:
- Show ALL working steps clearly, one step per line.
- State the formula used before applying it.
- Verify the answer where applicable.
- Mention units if the problem involves measurement.
- Highlight the final answer: *answer here*
- Use only methods taught at {board} Class {class_level} level.
- Warn about common calculation mistakes if relevant.

{FORMATTING_RULES_SCIENCE.format(class_level=class_level)}"""


def build_physics_prompt(board, class_level, chapter, question):
    return f"""You are an expert {board} Class {class_level} Physics teacher.

Board: {board} | Class: {class_level} | Chapter: {chapter}

Student's question: \"\"\"{question}\"\"\"

PHYSICS-SPECIFIC RULES:
- State the relevant law or principle first.
- Write the formula, then substitute values with units.
- Show unit conversions if needed.
- Always include units in the final answer.
- Highlight key formula and result: *formula*
- Use only {board} Class {class_level} syllabus methods.
- Mention SI units and common mistakes where relevant.

{FORMATTING_RULES_SCIENCE.format(class_level=class_level)}"""


def build_chemistry_prompt(board, class_level, chapter, question):
    return f"""You are an expert {board} Class {class_level} Chemistry teacher.

Board: {board} | Class: {class_level} | Chapter: {chapter}

Student's question: \"\"\"{question}\"\"\"

CHEMISTRY-SPECIFIC RULES:
- Write balanced chemical equations using -> for reaction arrow.
- Use correct subscript/superscript notation: H_2SO_4, Fe^{{3+}}.
- Include state symbols (s), (l), (g), (aq) in equations.
- For reactions: name the type (combination, decomposition, etc.).
- Highlight key equation or concept: *equation or concept*
- Only use {board} Class {class_level} syllabus content.

{FORMATTING_RULES_SCIENCE.format(class_level=class_level)}"""


def build_biology_prompt(board, class_level, chapter, question):
    return f"""You are an expert {board} Class {class_level} Biology teacher.

Board: {board} | Class: {class_level} | Chapter: {chapter}

Student's question: \"\"\"{question}\"\"\"

BIOLOGY-SPECIFIC RULES:
- Start with the exact board-level definition.
- Use correct scientific terminology as expected in {board} exams.
- For diagrams mentioned: describe key parts and their functions.
- Structure: Definition → Explanation → Example/Function → Exam tip.
- Highlight key term or answer: *key term*
- Keep answers factual and concise as expected in board exams.

{FORMATTING_RULES_SCIENCE.format(class_level=class_level)}"""


def build_english_lit_prompt(board, class_level, chapter, question):
    return f"""You are an expert {board} Class {class_level} English Literature teacher.

Board: {board} | Class: {class_level} | Text/Chapter: {chapter}

Student's question: \"\"\"{question}\"\"\"

ENGLISH LITERATURE RULES:
- Reference the exact text, poem, or prose from the {board} syllabus.
- For character questions: traits → evidence from text → significance.
- For theme questions: identify → explain → quote briefly → board relevance.
- For extract questions: context → meaning → literary devices → effect.
- Write in formal exam language.
- Keep answers within expected word limits for {board} Class {class_level}.

{FORMATTING_RULES_MINIMAL.format(class_level=class_level)}"""


def build_english_grammar_prompt(board, class_level, chapter, question):
    return f"""You are an expert {board} Class {class_level} English Grammar teacher.

Board: {board} | Class: {class_level} | Topic: {chapter}

Student's question: \"\"\"{question}\"\"\"

ENGLISH GRAMMAR RULES:
- State the grammatical rule first, clearly.
- Give the correct answer with a brief explanation.
- Provide 1-2 examples to reinforce the rule.
- For transformation/sentence rewriting: show the original and rewritten form.
- For comprehension: answer in complete sentences.
- Stick to {board} Class {class_level} grammar syllabus.

{FORMATTING_RULES_MINIMAL.format(class_level=class_level)}"""


def build_history_prompt(board, class_level, chapter, question):
    return f"""You are an expert {board} Class {class_level} History & Civics / Economics teacher.

Board: {board} | Class: {class_level} | Chapter: {chapter}

Student's question: \"\"\"{question}\"\"\"

HISTORY / CIVICS / ECONOMICS RULES:
- Give dates and facts accurately as per {board} syllabus.
- Structure: Introduction → Key Points → Significance/Cause/Effect → Conclusion.
- For short answers: 3-4 points, clear and direct.
- For long answers: introduction, developed paragraphs, conclusion.
- Bold key terms using *term* notation.
- Align answer format to {board} Class {class_level} exam expectations.

{FORMATTING_RULES_MINIMAL.format(class_level=class_level)}"""


def build_geography_prompt(board, class_level, chapter, question):
    return f"""You are an expert {board} Class {class_level} Geography teacher.

Board: {board} | Class: {class_level} | Chapter: {chapter}

Student's question: \"\"\"{question}\"\"\"

GEOGRAPHY RULES:
- For map-based questions: name regions, directions, and features precisely.
- For physical geography: explain processes step by step.
- For human geography: link causes to effects logically.
- Use correct geographical terminology as expected in {board} exams.
- Highlight key term or answer: *key term*
- Reference Indian/world geography as per {board} Class {class_level} syllabus.

{FORMATTING_RULES_MINIMAL.format(class_level=class_level)}"""


def build_computer_prompt(board, class_level, chapter, question):
    return f"""You are an expert {board} Class {class_level} Computer Applications teacher.

Board: {board} | Class: {class_level} | Chapter: {chapter}

Student's question: \"\"\"{question}\"\"\"

COMPUTER APPLICATIONS RULES:
- For theory: definition → explanation → example → board relevance.
- For programs (Java/Python): write clean, commented code.
- For output questions: trace through step by step, show each variable change.
- For algorithms/flowcharts: follow standard conventions.
- Stick strictly to {board} Class {class_level} syllabus (e.g. BlueJ for ICSE).
- Highlight key term or answer: *key term*

{FORMATTING_RULES_MINIMAL.format(class_level=class_level)}"""


def build_general_prompt(board, class_level, subject, chapter, question):
    return f"""You are an expert {board} Class {class_level} {subject} teacher.

Board: {board} | Class: {class_level} | Subject: {subject} | Chapter: {chapter}

Student's question: \"\"\"{question}\"\"\"

- Answer clearly and concisely at {board} Class {class_level} level.
- Use only methods and content from the official {board} syllabus.
- Show steps where required.
- Highlight key results: *answer*
- Be exam-oriented and student-friendly.

{FORMATTING_RULES_MINIMAL.format(class_level=class_level)}"""


# =====================================================
# SUBJECT ROUTER
# =====================================================
def build_prompt(board, class_level, subject, chapter, question):
    s = subject.lower()
    if "maths" in s or "math" in s:
        return build_maths_prompt(board, class_level, chapter, question)
    elif "physics" in s:
        return build_physics_prompt(board, class_level, chapter, question)
    elif "chemistry" in s:
        return build_chemistry_prompt(board, class_level, chapter, question)
    elif "biology" in s:
        return build_biology_prompt(board, class_level, chapter, question)
    elif "english lit" in s or "literature" in s:
        return build_english_lit_prompt(board, class_level, chapter, question)
    elif "english gram" in s or "grammar" in s:
        return build_english_grammar_prompt(board, class_level, chapter, question)
    elif "history" in s or "civics" in s or "economics" in s:
        return build_history_prompt(board, class_level, chapter, question)
    elif "geography" in s or "geo" in s:
        return build_geography_prompt(board, class_level, chapter, question)
    elif "computer" in s:
        return build_computer_prompt(board, class_level, chapter, question)
    else:
        return build_general_prompt(board, class_level, subject, chapter, question)


# =====================================================
# HEALTH ROUTE
# =====================================================
@app.get("/")
def root():
    return {"status": "running", "version": "4.0"}

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
            config=types.UploadFileConfig(mime_type=mime_type, display_name=name)
        )
        for _ in range(30):
            if uploaded.state.name == "ACTIVE":
                return uploaded.uri, uploaded.name
            if uploaded.state.name == "FAILED":
                raise Exception(f"File processing failed: {name}")
            await asyncio.sleep(1)
            uploaded = client.files.get(name=uploaded.name)
        raise Exception(f"File processing timeout: {name}")
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

    # ── Model selection ──
    model_name = "gemini-3.1-pro-preview" if model_choice == "t2" else "gemini-3.1-flash-lite-preview"

    # ── Build subject-specific prompt ──
    prompt_text = build_prompt(board, class_level, subject, chapter, question)

    # ── Build multimodal contents ──
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

    # ── Stream generator ──
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
                yield "event: error\ndata: API quota exceeded. Please try again later.\n\n"
            elif "api key" in error_msg.lower():
                yield "event: error\ndata: API configuration error. Please contact support.\n\n"
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
# LOCAL RUN
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
