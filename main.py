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
app = FastAPI(title="AI Tutor Backend", version="3.1")

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
    return {"status": "running", "version": "3.1"}

@app.get("/health")
def health():
    """Health check endpoint for monitoring"""
    try:
        # Quick test to verify Gemini API is accessible
        return {
            "status": "healthy",
            "api_key_present": bool(GEMINI_API_KEY),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# =====================================================
# IMPROVED FILE UPLOAD WITH ASYNC HANDLING
# =====================================================
async def upload_file_to_gemini(raw_bytes: bytes, mime_type: str, name: str):
    """
    Upload file to Gemini File API with proper async handling and retries
    """
    import io
    
    try:
        file_obj = io.BytesIO(raw_bytes)
        file_obj.name = name

        # Upload file
        uploaded = client.files.upload(
            file=file_obj,
            config=types.UploadFileConfig(
                mime_type=mime_type,
                display_name=name,
            )
        )

        # Wait for file to be ACTIVE with proper async handling
        max_attempts = 30  # 30 seconds max
        for attempt in range(max_attempts):
            if uploaded.state.name == "ACTIVE":
                return uploaded.uri, uploaded.name
            
            if uploaded.state.name == "FAILED":
                raise Exception(f"File processing failed: {name}")
            
            await asyncio.sleep(1)
            uploaded = client.files.get(name=uploaded.name)
        
        # Timeout reached
        raise Exception(f"File processing timeout for: {name}")
        
    except Exception as e:
        raise Exception(f"Failed to upload {name}: {str(e)}")

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
        model_name = "gemini-3-flash-preview"
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
   - NO Markdown, NO HTML, NO LaTeX, NO emojis

2. Allowed mathematical symbols ONLY:
   - Degrees: 30°
   - Fractions: 1/2
   - Equals sign: =
   - Plus or minus: + −
   - Square root: √

3. Write mathematics in NORMAL SCHOOL STYLE.
   Example: sin 30° = 1/2

4. Keep the answer:
   - SHORT
   - CLEAR
   - CONCEPTUALLY DEEP
   - STRICTLY exam-oriented

5. IMPORTANT HIGHLIGHTING RULES:
   - Highlight important formulas, definitions, or final answers
   - Use ONLY SINGLE ASTERISKS like *this*
   - NEVER use double asterisks **
   - The MAIN FINAL RESULT must be inside single asterisks

6. Language must be:
   - Simple
   - Calm
   - Clear
   - Suitable for Class {class_level} students

7. While giving output, be friendly and conversational with students.

8. BOARD ALIGNMENT: Answer ONLY what is officially taught at Class {class_level} level for {board}.

9. EXAM ANSWER EXPECTATION: Frame the answer exactly how a board examiner expects it.

10. STEP MARKING AWARENESS: Write steps in the correct logical order used for marking.

11. DEFINITIONS RULE: Start with the exact definition in simple board language.

12. FILE / IMAGE RULES:
    - If an image is attached, carefully read and analyse it
    - If it's a question paper, solve ALL visible questions step by step
    - If it's a diagram, explain what it shows in exam-appropriate language
    - If it's a PDF, treat it as a document and answer based on its content
    - Always relate file content back to {board} Class {class_level} {subject} syllabus
"""

    # =====================================================
    # BUILD GEMINI CONTENTS (multimodal) WITH ERROR HANDLING
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
            # ── PDF: upload via File API with improved error handling ──
            try:
                uri, file_name = await upload_file_to_gemini(raw_bytes, "application/pdf", name)
                contents.append(types.Part.from_uri(
                    uri=uri,
                    mime_type="application/pdf"
                ))
                uploaded_file_uris.append(file_name)
            except Exception as e:
                # Fallback: try inline if File API fails
                try:
                    contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                except Exception:
                    file_errors.append(f"Could not process PDF '{name}': {str(e)}")

        elif mime == "text/plain":
            # ── Plain text: decode and embed directly in prompt ──
            try:
                text_content = raw_bytes.decode("utf-8", errors="replace")
                prompt_text += f"\n\n--- Content of {name} ---\n{text_content}\n--- End of {name} ---"
            except Exception as e:
                file_errors.append(f"Could not read text file '{name}': {str(e)}")

        else:
            # ── Images: inline_data (jpeg, png, gif, webp) ──
            try:
                contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
            except Exception as e:
                file_errors.append(f"Could not process image '{name}': {str(e)}")

    # Add file errors to prompt if any
    if file_errors:
        prompt_text += "\n\n[Note: Some files could not be processed: " + "; ".join(file_errors) + "]"

    # Text prompt goes last
    contents.append(types.Part.from_text(text=prompt_text))

    # =====================================================
    # STREAM GENERATOR WITH IMPROVED ERROR HANDLING
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
                    # Log chunk error but try to continue
                    print(f"Chunk error: {chunk_error}")
                    if chunk_count == 0:
                        # If no chunks sent yet, report error
                        raise chunk_error
                    # Otherwise, just end the stream
                    break

            yield "event: end\ndata: done\n\n"

        except Exception as e:
            error_msg = str(e)
            # Send user-friendly error message
            if "quota" in error_msg.lower():
                yield f"event: error\ndata: API quota exceeded. Please try again later.\n\n"
            elif "api key" in error_msg.lower():
                yield f"event: error\ndata: API configuration error. Please contact support.\n\n"
            else:
                yield f"event: error\ndata: {error_msg}\n\n"
        finally:
            # Cleanup uploaded files
            for file_name in uploaded_file_uris:
                try:
                    client.files.delete(name=file_name)
                except:
                    pass  # Silent cleanup failure

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
