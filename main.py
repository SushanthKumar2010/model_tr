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
app = FastAPI(title="AI Tutor Backend", version="3.0")

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

# Supported MIME types Gemini can process
SUPPORTED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf",
    "text/plain",
}

client = genai.Client(api_key=GEMINI_API_KEY)

# =====================================================
# HEALTH ROUTE
# =====================================================
@app.get("/")
def root():
    return {"status": "running"}

# =====================================================
# SSE ASK ROUTE  (multimodal)
# =====================================================
@app.post("/api/ask")
async def ask_question(payload: dict):

    board        = (payload.get("board")        or "ICSE").strip().upper()
    class_level  = (payload.get("class_level")  or "10").strip()
    subject      = (payload.get("subject")      or "General").strip()
    chapter      = (payload.get("chapter")      or "General").strip()
    question     = (payload.get("question")     or "").strip()
    model_choice = (payload.get("model")        or "t1").lower()
    files        =  payload.get("files")        or []   # list of {name, mimeType, base64}

    if board not in ALLOWED_BOARDS:
        raise HTTPException(status_code=400, detail="Invalid board")

    if not question and not files:
        raise HTTPException(status_code=400, detail="Question or file required")

    # If no question text but files exist, add a default prompt
    if not question and files:
        question = "Please analyse this and answer any questions based on it."

    # =====================================================
    # MODEL SELECT
    # =====================================================
    if model_choice == "t2":
        model_name = "gemini-3-pro-preview"
    else:
        model_name = "gemini-2.5-flash-lite"

    # =====================================================
    # PROMPT
    # =====================================================
    prompt_text = f"""
You are an expert {board} Class {class_level} teacher.

Board: {board}
Subject: {subject}
Chapter: {chapter}

A student from Class {class_level} has asked the following question:

\"\"\"{question}\"\"\"

Your task is to answer strictly according to the {board} syllabus and exam pattern.

REQUIREMENTS:
- Explain the concept clearly and correctly.
- Use only {board} Class {class_level} level methods.
- Show all important steps and working where required (Maths, Physics, Chemistry).
- Keep the explanation concise but conceptually strong.
- Mention a common mistake ONLY if it is relevant.
- Focus on how answers are expected in board exams.

STRICT ANSWERING RULES (VERY IMPORTANT):

1. Use PLAIN TEXT ONLY.

- NO Markdown
- NO HTML
- NO LaTeX
- NO emojis
- NO special formatting commands

2. Allowed mathematical symbols ONLY:
- Degrees: 30°
- Fractions: 1/2
- Equals sign: =
- Plus or minus: + −
- Square root: √

3. Do NOT use:
- LaTeX-style syntax (\\sin, \\frac, ^, _)
- Markdown symbols (**, ##, -, etc.)

4. Write mathematics in NORMAL SCHOOL STYLE.
Example: sin 30° = 1/2

5. Keep the answer:
- SHORT
- CLEAR
- CONCEPTUALLY DEEP
- STRICTLY exam-oriented

6. Follow this STRUCTURE exactly:
- Core idea
- Explanation in 2 to 4 lines
- ONE simple value or example if useful
- Final answer or result

7. IMPORTANT HIGHLIGHTING RULES:
- Highlight EVERY important formulas, definitions, or final answers
- Use ONLY SINGLE ASTERISKS like *this*
- NEVER use double asterisks **
- NEVER over-highlight
- The MAIN FINAL RESULT must be inside single asterisks

8. Do NOT mention:
- AI
- Instructions
- Formatting rules
- Any external syllabus or board

9. Language must be:
- Simple
- Calm
- Clear
- Suitable for Class {class_level} students

10. Output structure:
- while giving the output don't be dry, instead be friendly and conversational with the students and generate
long and valuable answers. And also mention the thing which user says in the input "ex: class 10, ICSE, maths, ..." and you should generate output with reference to this.

11. BOARD ALIGNMENT RULE

* Answer ONLY what is officially taught at Class {class_level} level for the given {board}.
* Do NOT use higher-class shortcuts, advanced tricks, or competitive exam logic.
* If ICSE and CBSE approaches differ, follow the method strictly accepted in the given {board}.

12. EXAM ANSWER EXPECTATION

* Frame the answer exactly how a board examiner expects it.
* Use proper terminology used in {board} textbooks.
* Avoid casual wording that cannot earn marks in an exam.

13. STEP MARKING AWARENESS

* Write steps in the correct logical order used for marking.
* Do NOT skip steps that usually carry marks, even if the math looks simple.

14. DEFINITIONS RULE

* If the question involves a definition, law, principle, or statement,
* Start with the *exact definition* in simple board language.
* Do NOT paraphrase important definitions loosely.

15. DERIVATION RULE (If Applicable)

* If the question asks for a derivation,
* Write it in the standard school sequence.
* Do NOT compress or over-explain.
* End with the required final expression clearly.

16. NUMERICALS RULE

* Always write:
* Given values
* Formula used
* Substitution
* Final answer with unit (if applicable)
* Units must match board standards.

17. DIAGRAM REFERENCE RULE

* If a diagram is normally required in board exams,
* Mention "A neat labelled diagram should be drawn"
* Briefly explain using words only (no drawing).

18. COMMON MISTAKE RULE

* Mention a common mistake ONLY if students frequently lose marks because of it.
* Keep it to ONE short line.

19. WORD LIMIT DISCIPLINE

* Do NOT add extra theory beyond what is needed to score full marks.
* No storytelling, no motivation talk, no unrelated facts.

20. SUBJECT-SPECIFIC STRICTNESS

* Maths: logical steps, no skipped working.
* Physics: formula, substitution, unit correctness.
* Chemistry: correct reactions, conditions, symbols, and names.
* Biology: keyword-based answers, no vague explanations.

21. LANGUAGE CONTROL

* Use simple school-level English.
* No fancy vocabulary.
* Every sentence should help gain marks.

22. FINAL ANSWER EMPHASIS

* The final result or conclusion MUST be clearly stated at the end.
* The examiner should be able to find the answer immediately.

23. NO ASSUMPTIONS RULE

* Do NOT assume what the student knows.
* Explain briefly but clearly, exactly at Class {class_level} level.

24. ICSE AND CBSE EQUALITY RULE

* Treat ICSE, CBSE, & SSLC with equal seriousness.
* Do NOT favor NCERT wording unless the board is CBSE.
* Do NOT favor concise answers unless the board is ICSE.

25. SSLC RULES

* if the board is selected as SSLC, understand that it is related to KARNATKA BOARD
* if this board is selected, give answers with reference to the latest SSLC KARNATAKA BOARD syllabus

26. FILE / IMAGE RULES (applies when files are attached)

* If an image is attached, carefully read and analyse it before answering.
* If it is a question paper or worksheet, solve ALL visible questions step by step.
* If it is a diagram, explain what the diagram shows in exam-appropriate language.
* If it is a PDF, treat it as a document and answer based on its content.
* Always relate the file content back to {board} Class {class_level} {subject} syllabus.
* If the image quality is poor or unclear, state that briefly and answer based on what is visible.
"""

    # =====================================================
    # BUILD GEMINI CONTENTS (multimodal)
    # Images  → inline_data (fast, no upload needed)
    # PDFs    → Gemini File API (handles large files, proper page parsing)
    # Text    → appended to prompt directly
    # =====================================================
    contents = []
    uploaded_file_uris = []   # track for optional cleanup

    for f in files:
        mime = f.get("mimeType", "")
        b64  = f.get("base64",   "")
        name = f.get("name",     "file")

        if not b64 or not mime:
            continue

        if mime not in SUPPORTED_MIME_TYPES:
            prompt_text += f"\n\n[Note: File '{name}' ({mime}) is unsupported and was skipped.]"
            continue

        try:
            raw_bytes = base64.b64decode(b64)
        except Exception as e:
            prompt_text += f"\n\n[Note: Could not decode file '{name}': {e}]"
            continue

        if mime == "application/pdf":
            # ── PDF: upload via File API, then reference by URI ──
            try:
                import io
                file_obj = io.BytesIO(raw_bytes)
                file_obj.name = name  # some SDK versions use this

                uploaded = client.files.upload(
                    file=file_obj,
                    config=types.UploadFileConfig(
                        mime_type="application/pdf",
                        display_name=name,
                    )
                )

                # Wait for file to be ACTIVE (usually instant for small PDFs)
                import time
                max_wait = 20  # seconds
                waited = 0
                while uploaded.state.name == "PROCESSING" and waited < max_wait:
                    time.sleep(1)
                    waited += 1
                    uploaded = client.files.get(name=uploaded.name)

                if uploaded.state.name == "ACTIVE":
                    contents.append(types.Part.from_uri(
                        uri=uploaded.uri,
                        mime_type="application/pdf"
                    ))
                    uploaded_file_uris.append(uploaded.name)
                else:
                    prompt_text += f"\n\n[Note: PDF '{name}' could not be processed (state: {uploaded.state.name}).]"

            except Exception as e:
                # Fallback: try inline if File API fails
                prompt_text += f"\n\n[Note: PDF upload failed ({e}), trying inline.]"
                try:
                    contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                except Exception:
                    prompt_text += f"\n\n[Note: Could not process PDF '{name}' at all.]"

        elif mime == "text/plain":
            # ── Plain text: decode and embed directly in prompt ──
            try:
                text_content = raw_bytes.decode("utf-8", errors="replace")
                prompt_text += f"\n\n--- Content of {name} ---\n{text_content}\n--- End of {name} ---"
            except Exception as e:
                prompt_text += f"\n\n[Note: Could not read text file '{name}': {e}]"

        else:
            # ── Images: inline_data (jpeg, png, gif, webp) ──
            try:
                contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
            except Exception as e:
                prompt_text += f"\n\n[Note: Could not process image '{name}': {e}]"

    # Text prompt goes last
    contents.append(types.Part.from_text(text=prompt_text))

    # =====================================================
    # STREAM GENERATOR
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

            while True:
                chunk = await loop.run_in_executor(None, get_next)
                if chunk is None:
                    break
                text = chunk.text or ""
                if text:
                    yield f"data: {json.dumps(text)}\n\n"

            yield "event: end\ndata: done\n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

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
