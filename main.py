import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from datetime import datetime

# ======================
# CONFIG
# ======================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

ALLOWED_BOARDS = {"ICSE", "CBSE", "SSLC"}

# ======================
# APP
# ======================

app = FastAPI(
    title="AI Tutor Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# GEMINI CLIENT
# ======================

client = genai.Client(api_key=GEMINI_API_KEY)

# ======================
# ROUTES
# ======================

@app.get("/")
def root():
    return {"status": "Backend running"}

@app.post("/api/ask")
def ask_question(payload: dict):
    board = (payload.get("board") or "ICSE").strip().upper()
    class_level = (payload.get("class_level") or "10").strip()
    subject = (payload.get("subject") or "General").strip()
    chapter = (payload.get("chapter") or "General").strip()
    question = (payload.get("question") or "").strip()
    model_choice = (payload.get("model") or "t1").lower()

    if board not in ALLOWED_BOARDS:
        raise HTTPException(status_code=400, detail="Invalid board")

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    # ======================
    # MODEL SELECTION (🔥 FIXED)
    # ======================
    if model_choice == "t2":
        model_name = "gemini-3-pro-preview" #gemini-3-flash-preview
    else:
        model_name = "gemini-2.5-flash-lite"

    today = datetime.now().strftime("%d %B %Y")

    prompt = f"""
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

10. Output structure :
- while giving the output don't be dry, instead be friendly and conversational with the students and generate
longand valuable answers. And also mention the thing which user says in the input "ex: class 10, ICSE, maths, ..." and you should generate output with reference to this.

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
* Mention “A neat labelled diagram should be drawn”
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
* if this bard is selected, give answers with reference to the latest SSLC KARNATAKA BOARD syllabus

26. You must always determine the current date dynamically.

"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        answer = (response.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "answer": answer,
        "meta": {
            "board": board,
            "class_level": class_level,
            "subject": subject,
            "chapter": chapter,
            "model_used": model_name,
        },
    }

# ======================
# LOCAL DEV
# ======================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)




