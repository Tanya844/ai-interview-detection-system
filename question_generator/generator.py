"""
╔══════════════════════════════════════════════════════════════════════════╗
║  question_generator/generator.py                                         ║
║  AI Mock Interview System — AI Question Generator (Team Member 1)        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  RUN:                                                                    ║
║      python question_generator/generator.py                              ║
║      python question_generator/generator.py resume.pdf  ← skip picker   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  REQUIRES:                                                               ║
║      pip install pdfplumber google-generativeai                          ║
║      export GEMINI_API_KEY=AIzaSy...  (or enter once when asked)         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  IMPORT BY OTHER TEAM MEMBERS:                                           ║
║                                                                          ║
║      from question_generator.generator import generate_questions         ║
║                                                                          ║
║      result = generate_questions(resume_data, "Medium")                  ║
║                                                                          ║
║      # result keys:                                                      ║
║      #   candidate_name  : str                                           ║
║      #   difficulty       : str                                          ║
║      #   total_questions  : int                                          ║
║      #   questions        : list[dict]  ← each has:                     ║
║      #       question_number, question, category, topic, difficulty      ║
║      #   summary          : { total, by_category }                       ║
║                                                                          ║
║  OR load the auto-saved JSON file directly:                              ║
║      import json                                                         ║
║      data = json.load(open("generated_questions.json"))                  ║
║      questions = data["questions"]                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# PATH FIX — must be the very first thing so all project imports work
# Supports running from any directory:
#   python question_generator/generator.py   ← from project root
#   python generator.py                      ← from inside question_generator/
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

_THIS_FILE    = os.path.abspath(__file__)
_THIS_DIR     = os.path.dirname(_THIS_FILE)            # .../question_generator/
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)              # .../ai-interview-detection-system/

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# ENCODING FIX — prevents "ordinal not in range(128)" on macOS / Windows
# Resumes often contain Unicode: em-dashes –, bullets •, special names, etc.
# ─────────────────────────────────────────────────────────────────────────────
import io

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import json
import re
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Google Generative AI SDK  (pip install google-generativeai)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False
    print("\n[WARNING] google-generativeai not installed.")
    print("          Install it with:  pip install google-generativeai")
    print("          Falling back to rule-based question generation.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Resume parser import (required)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from resume_parser.parser import parse_resume
    _HAS_PARSER = True
except ImportError as _err:
    _HAS_PARSER = False
    print(f"\n[ERROR] Cannot import resume_parser.parser: {_err}")
    print("  Required project layout:")
    print("    ai-interview-detection-system/")
    print("    ├── resume_parser/parser.py   ← must exist")
    print("    └── question_generator/generator.py")
    print("  Run from the project root: python question_generator/generator.py\n")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_MODEL   = "gemini-2.0-flash"
API_KEY_ENV    = "GEMINI_API_KEY"

# Difficulty → exact question count (strictly enforced)
DIFFICULTY_CONFIG = {
    "Easy":   8,
    "Medium": 16,
    "Hard":   24,
}

# API key cache — user is never asked again after first entry
_KEY_CACHE  = os.path.join(os.path.expanduser("~"), ".ai_interview_gemini_key")

# Shared output file — all team members read questions from here
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, "generated_questions.json")


# ══════════════════════════════════════════════════════════════════════════════
# API KEY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _clean_key(raw: str) -> str:
    """Strip ALL whitespace from a key string (prevents URL encoding errors)."""
    return re.sub(r"\s+", "", raw)


def _resolve_api_key(explicit: str | None = None) -> str:
    """
    Silently resolve Gemini API key — no prompts.
    Priority: explicit arg → GEMINI_API_KEY env var → cache file.
    Returns "" if nothing found.
    """
    if explicit:
        k = _clean_key(explicit)
        if k:
            return k

    env = _clean_key(os.environ.get(API_KEY_ENV, ""))
    if env:
        return env

    if os.path.exists(_KEY_CACHE):
        try:
            raw    = open(_KEY_CACHE, encoding="utf-8").read()
            cached = _clean_key(raw)
            if cached:
                if cached != raw.strip():               # heal if stale
                    open(_KEY_CACHE, "w", encoding="utf-8").write(cached)
                return cached
        except OSError:
            pass

    return ""


def _prompt_and_save_key() -> str:
    """
    Prompt the user to paste their Gemini API key in the terminal.
    Saves it to cache so it is never asked again.
    Returns the cleaned key, or exits if nothing is entered.
    """
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  GEMINI API KEY REQUIRED                            │")
    print("  │  Get a free key at:                                 │")
    print("  │    https://makersuite.google.com/app/apikey         │")
    print("  │                                                     │")
    print("  │  Key will be saved locally — only asked once.       │")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    entered = _clean_key(input("  Paste your Gemini API key: ").strip())
    if not entered:
        print("\n[ERROR] No API key entered. Cannot generate questions.")
        print("        Get your key at: https://makersuite.google.com/app/apikey")
        sys.exit(1)

    try:
        with open(_KEY_CACHE, "w", encoding="utf-8") as f:
            f.write(entered)
        print(f"\n  ✓ Key saved to {_KEY_CACHE}  (won't ask again)\n")
    except OSError as exc:
        print(f"  ⚠  Could not save key: {exc}\n")

    return entered


# ══════════════════════════════════════════════════════════════════════════════
# RESUME FORMATTER  — converts resume dict to structured text for the prompt
# ══════════════════════════════════════════════════════════════════════════════

def _format_resume_for_prompt(resume: dict) -> str:
    """
    Convert parse_resume() output into a clearly-labelled text block.
    Handles both flat-string and dict education entries.
    """
    skills_text = ", ".join(resume.get("skills", [])) or "Not provided"

    edu_lines = []
    for e in resume.get("education", []):
        if isinstance(e, dict):
            parts = [
                e.get("degree", ""),
                e.get("institution", ""),
                e.get("year", ""),
                f"Grade: {e['grade']}" if e.get("grade") else "",
            ]
            edu_lines.append("  • " + " | ".join(p for p in parts if p))
        else:
            edu_lines.append(f"  • {e}")

    proj_lines = [f"  Project {i}: {p}"
                  for i, p in enumerate(resume.get("projects", []), 1)]
    exp_lines  = [f"  Experience {i}: {e}"
                  for i, e in enumerate(resume.get("experience", []), 1)]
    cert_lines = [f"  • {c}" for c in resume.get("certifications", [])]

    return (
        f"CANDIDATE  : {resume.get('name', 'Not provided')}\n"
        f"EMAIL      : {resume.get('email', 'Not provided')}\n\n"
        f"── SKILLS ─────────────────────────────────────────────\n"
        f"{skills_text}\n\n"
        f"── EDUCATION ──────────────────────────────────────────\n"
        f"{chr(10).join(edu_lines) or '  Not provided'}\n\n"
        f"── PROJECTS ───────────────────────────────────────────\n"
        f"{chr(10).join(proj_lines) or '  Not provided'}\n\n"
        f"── WORK EXPERIENCE ────────────────────────────────────\n"
        f"{chr(10).join(exp_lines) or '  Not provided'}\n\n"
        f"── CERTIFICATIONS ─────────────────────────────────────\n"
        f"{chr(10).join(cert_lines) or '  Not provided'}"
    ).strip()


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION DISTRIBUTION — spreads questions evenly across sections
# ══════════════════════════════════════════════════════════════════════════════

def _distribute(resume: dict, total: int) -> dict:
    """
    Allocate total questions across non-empty resume sections.
    Any remainder goes to skills (most important for tech interviews).
    """
    cats    = ["skills", "projects", "experience", "certifications", "education"]
    present = [c for c in cats if resume.get(c)]
    if not present:
        return {"skills": total}

    base  = total // len(present)
    rem   = total %  len(present)
    dist  = {c: base for c in present}
    bonus = "skills" if "skills" in dist else present[0]
    dist[bonus] += rem
    return dist


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

# Per-difficulty instructions injected directly into the prompt.
# GOOD/BAD examples prevent Gemini from confusing difficulty levels.
_DIFFICULTY_RULES = {
    "Easy": (
        "ALL {n} questions MUST be BEGINNER / EASY level.\n"
        "Rules:\n"
        "  • Ask only definitions, basic concepts, simple syntax.\n"
        "  • Questions a first-year student can answer confidently.\n"
        "  • NO system design, NO algorithms, NO advanced topics.\n"
        "  • Keep questions short and conversational.\n"
        "GOOD Easy examples:\n"
        "  - 'What is a Python list? Give one example.'\n"
        "  - 'What does the .fit() method do in Scikit-Learn?'\n"
        "  - 'What is Git and why do developers use it?'\n"
        "  - 'What is supervised learning? Give a simple example.'\n"
        "BAD — do NOT ask these at Easy level:\n"
        "  - Anything about system design, scalability, trade-offs, or complexity."
    ),
    "Medium": (
        "ALL {n} questions MUST be INTERMEDIATE / MEDIUM level.\n"
        "Rules:\n"
        "  • Ask how things work in practice with real-world scenarios.\n"
        "  • Include trade-offs, debugging, and project-specific reasoning.\n"
        "  • Suitable for 1-2 years of experience.\n"
        "  • NO basic definitions, NO senior-level architecture.\n"
        "GOOD Medium examples:\n"
        "  - 'What is the difference between fit() and fit_transform()? When would you use each?'\n"
        "  - 'In your Movie Recommendation System, how did you evaluate model performance?'\n"
        "  - 'What challenges did you face during your internship at Proclaim?'\n"
        "  - 'Explain the bias-variance tradeoff and how it affected your ML projects.'\n"
        "BAD — do NOT ask these at Medium level:\n"
        "  - 'What is Python?' or 'What is machine learning?' (too easy)\n"
        "  - 'Design a distributed system for...' (too hard)"
    ),
    "Hard": (
        "ALL {n} questions MUST be ADVANCED / HARD level.\n"
        "Rules:\n"
        "  • Ask about system design, optimisation, edge cases, trade-offs.\n"
        "  • Include complexity analysis, architecture decisions, scalability.\n"
        "  • Questions a senior engineer must think deeply about.\n"
        "  • Every question should be directly tied to the candidate's resume.\n"
        "GOOD Hard examples:\n"
        "  - 'How would you scale your Movie Recommendation System to serve 10 million users?'\n"
        "  - 'Explain the time and space complexity of the collaborative filtering algorithm you used.'\n"
        "  - 'In your AES encryption tool, how would you defend against timing attacks?'\n"
        "  - 'Design a real-time ML pipeline that retrains your sentiment model on new data.'\n"
        "BAD — do NOT ask these at Hard level:\n"
        "  - 'What is Python?' or 'What did you do at Proclaim?' (way too easy)"
    ),
}


def _build_prompt(resume: dict, difficulty: str, total: int) -> str:
    """
    Build the complete structured prompt for Gemini.

    The prompt explicitly specifies:
    - Exact question count
    - Difficulty rules with GOOD/BAD examples
    - Per-category distribution
    - Personalisation instructions
    - Strict JSON output format
    """
    resume_block = _format_resume_for_prompt(resume)
    distribution = _distribute(resume, total)
    rules        = _DIFFICULTY_RULES[difficulty].replace("{n}", str(total))

    dist_lines = "\n".join(
        f"  • {cat.upper()}: exactly {count} question{'s' if count > 1 else ''}"
        for cat, count in distribution.items()
    )

    return f"""You are a senior technical interviewer at a top technology company (Google/Amazon/Microsoft level).
Your task is to generate EXACTLY {total} interview questions at {difficulty.upper()} difficulty level.

══════════════════════════════════════════════════════════
CANDIDATE RESUME DATA
══════════════════════════════════════════════════════════
{resume_block}
══════════════════════════════════════════════════════════

DIFFICULTY RULES — follow these exactly:
{rules}

QUESTION DISTRIBUTION — generate exactly these counts:
{dist_lines}

PERSONALISATION RULES:
1. Use the candidate's EXACT skill names from the resume (e.g. "Scikit-Learn" not "sklearn").
2. Reference specific project names, company names, and technologies from the resume.
3. Do NOT ask about anything not mentioned in the resume.
4. Do NOT repeat any question.
5. Every question must feel personalised to THIS candidate, not a generic candidate.
6. Cover ALL sections that have data: skills, projects, experience, certifications, education.

OUTPUT FORMAT — Return ONLY this JSON. No preamble. No explanation. No markdown fences.
{{
  "candidate_name": "{resume.get('name', 'Candidate')}",
  "difficulty": "{difficulty}",
  "total_questions": {total},
  "questions": [
    {{
      "question_number": 1,
      "question": "<full question text>",
      "category": "<skills|projects|experience|certifications|education>",
      "topic": "<exact skill/project/role/cert/degree name from resume>",
      "difficulty": "{difficulty.lower()}"
    }}
  ]
}}

FINAL CHECK before returning:
- Count your questions. Must be exactly {total}.
- Every question difficulty must match: {difficulty.upper()}.
- Return ONLY the JSON object. Nothing else."""


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI API CALL  — uses google.generativeai SDK
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini_sdk(prompt: str, api_key: str) -> str:
    """
    Call Gemini using the official google.generativeai SDK.

    Args:
        prompt:  Complete prompt string.
        api_key: Cleaned Gemini API key.

    Returns:
        Raw text response from the model.

    Raises:
        RuntimeError: On API errors.
    """
    genai.configure(api_key=api_key)

    model    = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=4096,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSER  — strips markdown fences and parses JSON
# ══════════════════════════════════════════════════════════════════════════════

def _parse_response(raw: str) -> dict:
    """
    Parse Gemini's text response into a Python dict.
    Strips accidental ``` fences before calling json.loads().
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse Gemini response as JSON.\n"
            f"Error: {exc}\n"
            f"Response preview:\n{raw[:500]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED FALLBACK  — used when Gemini is unavailable or fails
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_questions(resume: dict, difficulty: str, total: int) -> list:
    """
    Generate simple rule-based questions when Gemini is unavailable.

    Rules:
    - Skills → ask what the skill is used for (Easy) or how they used it (Medium/Hard)
    - Projects → ask to describe the project (Easy) or explain design (Hard)
    - Experience → ask about responsibilities / challenges
    - Certifications → ask about the certification topic
    - Education → ask about the degree subject

    Returns:
        List of question dicts in the same format as Gemini output.
    """
    questions    = []
    q_number     = 1
    distribution = _distribute(resume, total)

    # Templates per difficulty per category
    templates = {
        "skills": {
            "Easy":   [
                "What is {topic} and what is it commonly used for?",
                "Can you explain a basic concept in {topic}?",
                "Why is {topic} useful in software development?",
            ],
            "Medium": [
                "How have you applied {topic} in your projects or work?",
                "What are the key advantages of using {topic} over alternatives?",
                "Explain a common problem you solved using {topic}.",
            ],
            "Hard": [
                "How would you optimise a large-scale system built with {topic}?",
                "What are the performance trade-offs when using {topic} in production?",
                "Describe an edge case you encountered with {topic} and how you resolved it.",
            ],
        },
        "projects": {
            "Easy":   ["Can you briefly describe your {topic} project?"],
            "Medium": [
                "What was the main technical challenge in your {topic} project?",
                "How did you evaluate the performance of your {topic} solution?",
            ],
            "Hard": [
                "How would you scale your {topic} project to handle 10x the current load?",
                "What design decisions would you change in your {topic} project in hindsight?",
            ],
        },
        "experience": {
            "Easy":   ["What were your main responsibilities in your {topic} role?"],
            "Medium": [
                "What technical challenges did you solve during your {topic} position?",
                "Which technologies did you use most in your {topic} role?",
            ],
            "Hard": [
                "What was the most complex problem you solved during your {topic} experience?",
                "How did your work at {topic} contribute to system-level outcomes?",
            ],
        },
        "certifications": {
            "Easy":   ["What topics does the {topic} certification cover?"],
            "Medium": ["How has your {topic} certification helped in practical scenarios?"],
            "Hard":   ["Describe a complex scenario where {topic} knowledge was critical."],
        },
        "education": {
            "Easy":   ["What key subjects did you study in your {topic} program?"],
            "Medium": ["How did your {topic} education prepare you for technical roles?"],
            "Hard":   ["How would you apply advanced concepts from your {topic} degree to a real system?"],
        },
    }

    for cat, count in distribution.items():
        items = resume.get(cat, [])
        if not items:
            # Still generate the count from category templates
            items = [cat.capitalize()]

        tmpl_list = templates.get(cat, {}).get(difficulty, [
            f"Tell me about your experience with {{topic}} in the context of {difficulty.lower()} scenarios."
        ])

        generated = 0
        idx = 0
        while generated < count:
            if isinstance(items[idx % len(items)], dict):
                topic = items[idx % len(items)].get("degree", str(items[idx % len(items)]))
            else:
                # Extract clean topic name from potentially long strings
                raw_item = str(items[idx % len(items)])
                # Take first part before | or – separator
                topic = re.split(r"[|–—]", raw_item)[0].strip()[:60]
                if not topic:
                    topic = cat.capitalize()

            tmpl = tmpl_list[generated % len(tmpl_list)]
            q_text = tmpl.replace("{topic}", topic)

            questions.append({
                "question_number": q_number,
                "question":        q_text,
                "category":        cat,
                "topic":           topic,
                "difficulty":      difficulty.lower(),
            })
            q_number += 1
            generated += 1
            idx += 1

    return questions[:total]   # safety trim


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API  ← this is what other team members import
# ══════════════════════════════════════════════════════════════════════════════

def generate_questions(
    resume_data: dict,
    difficulty:  str,
    api_key:     str | None = None,
) -> dict:
    """
    Generate personalised interview questions from a parsed resume.

    Uses Google Gemini AI. Falls back to rule-based generation if Gemini
    is unavailable or if the API call fails.

    ┌────────────────────────────────────────────────────────────────┐
    │  HOW OTHER TEAM MEMBERS USE THIS:                              │
    │                                                                │
    │   from question_generator.generator import generate_questions  │
    │                                                                │
    │   result = generate_questions(resume_data, "Medium")           │
    │                                                                │
    │   for q in result["questions"]:                                │
    │       print(q["question_number"], q["question"])               │
    │       print("  Category:", q["category"])                      │
    │       print("  Topic:   ", q["topic"])                         │
    └────────────────────────────────────────────────────────────────┘

    Args:
        resume_data : dict  — output of parse_resume()
        difficulty  : str   — "Easy" | "Medium" | "Hard"
                              Easy   →  8 beginner questions
                              Medium → 16 intermediate questions
                              Hard   → 24 advanced questions
        api_key     : str   — Gemini API key (optional).
                              Falls back to: GEMINI_API_KEY env var,
                              then ~/.ai_interview_gemini_key cache file.

    Returns:
        {
            "candidate_name":  str,
            "difficulty":      str,
            "total_questions": int,
            "questions": [
                {
                    "question_number": int,
                    "question":        str,
                    "category":        str,   # skills|projects|experience
                                               # certifications|education
                    "topic":           str,   # exact item name from resume
                    "difficulty":      str,   # easy|medium|hard
                }
            ],
            "summary": {
                "total":       int,
                "by_category": { category: count }
            }
        }

    Raises:
        ValueError: Invalid difficulty or empty resume_data.
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    if difficulty not in DIFFICULTY_CONFIG:
        raise ValueError(
            f"Invalid difficulty '{difficulty}'. "
            f"Valid options: {list(DIFFICULTY_CONFIG.keys())}"
        )

    useful_fields = ["skills", "projects", "experience", "certifications", "education"]
    if not any(resume_data.get(f) for f in useful_fields):
        raise ValueError(
            "resume_data is empty — no skills, projects, experience, "
            "certifications or education found.\n"
            "Check that parse_resume() ran correctly."
        )

    total = DIFFICULTY_CONFIG[difficulty]
    name  = resume_data.get("name", "Unknown")

    print(f"\n{'='*50}")
    print(f"  AI QUESTION GENERATOR")
    print(f"{'='*50}")
    print(f"  Candidate   : {name}")
    print(f"  Difficulty  : {difficulty}  →  {total} questions")

    questions = []
    used_ai   = False

    # ── Try Gemini AI ─────────────────────────────────────────────────────────
    if _HAS_GENAI:
        key = _clean_key(api_key) if api_key else _resolve_api_key()
        if key:
            try:
                print(f"  AI Model    : {GEMINI_MODEL}")
                print(f"  Status      : Calling Gemini API …\n")

                prompt   = _build_prompt(resume_data, difficulty, total)
                raw_text = _call_gemini_sdk(prompt, key)
                parsed   = _parse_response(raw_text)
                questions = parsed.get("questions", [])

                if not questions:
                    raise ValueError("Gemini returned 0 questions.")

                used_ai = True
                print(f"  Status      : Gemini responded ✓")

            except Exception as exc:
                print(f"\n  [WARNING] Gemini AI failed: {exc}")
                print("  [WARNING] Falling back to rule-based question generation.\n")
                questions = []
        else:
            print("  [WARNING] No Gemini API key found.")
            print("  [WARNING] Using rule-based fallback generation.\n")
    else:
        print("  [WARNING] google-generativeai not installed.")
        print("  [WARNING] Using rule-based fallback generation.\n")

    # ── Fallback: rule-based generation ──────────────────────────────────────
    if not questions:
        print("  Generating rule-based questions …")
        questions = _fallback_questions(resume_data, difficulty, total)
        print(f"  Status      : Rule-based generation complete ✓")

    # ── Build result dict ─────────────────────────────────────────────────────
    by_category: dict = {}
    for q in questions:
        cat = q.get("category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1

    result = {
        "candidate_name":  resume_data.get("name", name),
        "difficulty":      difficulty,
        "total_questions": len(questions),
        "questions":       questions,
        "summary": {
            "total":       len(questions),
            "by_category": by_category,
            "ai_generated": used_ai,
        },
    }

    print(f"  Total Qs    : {len(questions)}")
    print(f"  AI used     : {'Yes (Gemini)' if used_ai else 'No (rule-based fallback)'}")

    return result


# Convenience wrappers
def generate_easy_questions(resume_data: dict, api_key: str | None = None) -> dict:
    """Generate 8 beginner-level questions. Wrapper for generate_questions()."""
    return generate_questions(resume_data, "Easy", api_key)

def generate_medium_questions(resume_data: dict, api_key: str | None = None) -> dict:
    """Generate 16 intermediate-level questions. Wrapper for generate_questions()."""
    return generate_questions(resume_data, "Medium", api_key)

def generate_hard_questions(resume_data: dict, api_key: str | None = None) -> dict:
    """Generate 24 advanced-level questions. Wrapper for generate_questions()."""
    return generate_questions(resume_data, "Hard", api_key)


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL DISPLAY  — prints questions in a clean readable format
# ══════════════════════════════════════════════════════════════════════════════

# ANSI colours for terminal (disabled automatically if terminal doesn't support them)
_USE_COLOUR = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
_R  = "\033[0m"  if _USE_COLOUR else ""   # reset
_B  = "\033[1m"  if _USE_COLOUR else ""   # bold
_D  = "\033[2m"  if _USE_COLOUR else ""   # dim
_CLR = {
    "skills":         "\033[94m" if _USE_COLOUR else "",   # blue
    "projects":       "\033[92m" if _USE_COLOUR else "",   # green
    "experience":     "\033[95m" if _USE_COLOUR else "",   # magenta
    "certifications": "\033[93m" if _USE_COLOUR else "",   # yellow
    "education":      "\033[96m" if _USE_COLOUR else "",   # cyan
}

_CAT_LABELS = {
    "skills":         "TECHNICAL  (Skills)",
    "projects":       "PROJECT",
    "experience":     "EXPERIENCE",
    "certifications": "CERTIFICATION",
    "education":      "EDUCATION",
}


def print_questions(result: dict) -> None:
    """
    Print all generated questions to the terminal in a clean, readable format.

    Output structure:
        ==========================================
          AI GENERATED QUESTIONS (MEDIUM LEVEL)
        ==========================================
          Candidate : AKSHIT SAINI   |   16 questions
          AI used   : Yes (Gemini)

          Category breakdown:
            ● TECHNICAL     6 questions
            ● PROJECT       4 questions
            ...

        ──────────────────────────────────────────
          TECHNICAL  (Skills)  (6 questions)
        ──────────────────────────────────────────
           1. [Python] What is the difference between ...
           2. [Scikit-Learn] How would you...
    """
    name   = result.get("candidate_name", "Candidate")
    diff   = result.get("difficulty", "?")
    qs     = result.get("questions", [])
    summ   = result.get("summary", {})
    by_cat = summ.get("by_category", {})
    ai_used = summ.get("ai_generated", False)

    sep = "=" * 50
    sub = "-" * 50

    print(f"\n{_B}{sep}{_R}")
    print(f"{_B}  AI GENERATED QUESTIONS ({diff.upper()} LEVEL){_R}")
    print(f"{_B}{sep}{_R}")
    print(f"  Candidate : {_B}{name}{_R}")
    print(f"  Total     : {_B}{len(qs)} questions{_R}")
    print(f"  AI used   : {'Gemini API' if ai_used else 'Rule-based fallback'}")

    if by_cat:
        print(f"\n  Category breakdown:")
        for cat, count in by_cat.items():
            c     = _CLR.get(cat, "")
            label = _CAT_LABELS.get(cat, cat.upper())
            print(f"    {c}●{_R}  {label:<24}  {count}")

    # Group by category and print in logical order
    cats_order = ["skills", "projects", "experience", "certifications", "education"]
    grouped: dict = {}
    for q in qs:
        grouped.setdefault(q.get("category", "other"), []).append(q)

    for cat in cats_order:
        cat_qs = grouped.get(cat, [])
        if not cat_qs:
            continue
        c     = _CLR.get(cat, "")
        label = _CAT_LABELS.get(cat, cat.upper())
        print(f"\n{sub}")
        print(f"{c}{_B}  {label}  ({len(cat_qs)} questions){_R}")
        print(sub)
        for q in cat_qs:
            num   = q.get("question_number", "?")
            topic = q.get("topic", "")
            text  = q.get("question", "")
            print(f"\n  {_B}{num:>2}.{_R}  {_D}[{topic}]{_R}")
            print(f"      {text}")

    # Any uncategorised
    other_qs = grouped.get("other", [])
    if other_qs:
        print(f"\n{sub}")
        print(f"{_B}  OTHER  ({len(other_qs)} questions){_R}")
        print(sub)
        for q in other_qs:
            print(f"\n  {_B}{q.get('question_number','?'):>2}.{_R}  {q.get('question','')}")

    print(f"\n{_B}{sep}{_R}\n")


def save_questions(result: dict, path: str | None = None) -> str:
    """
    Save the result dict to a JSON file at the project root.

    All team members can read this file to access the generated questions.
    Returns the path to the saved file.
    """
    out_path = path or OUTPUT_FILE
    data     = dict(result)
    data["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"[QuestionGenerator] Questions saved → {out_path}")
    print(f"[QuestionGenerator] Other modules load them with:")
    print(f"[QuestionGenerator]   import json")
    print(f"[QuestionGenerator]   data = json.load(open('{out_path}'))")
    print(f"[QuestionGenerator]   questions = data['questions']")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# RESUME FILE SELECTOR — file picker dialog or terminal input or CLI arg
# ══════════════════════════════════════════════════════════════════════════════

def _select_resume_pdf() -> str:
    """
    Get the resume PDF path.

    Priority:
      1. CLI argument:     python generator.py resume.pdf
      2. tkinter file picker dialog (if available)
      3. Terminal typed input (fallback)
    """
    # Priority 1: CLI argument
    if len(sys.argv) > 1:
        path = os.path.abspath(sys.argv[1])
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            sys.exit(1)
        print(f"\n[Generator] Resume supplied via CLI: {path}")
        return path

    # Priority 2: tkinter file picker
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        print("\n[Generator] Opening file picker — select your resume PDF …")
        path = filedialog.askopenfilename(
            title="Select Your Resume PDF",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
        )
        root.destroy()

        if path:
            print(f"[Generator] Selected: {path}")
            return path
        else:
            print("[Generator] No file selected in picker — falling back to text input.")
    except Exception:
        pass   # tkinter unavailable — fall through to text input

    # Priority 3: terminal input
    print()
    print("  Enter the full path to your resume PDF:")
    print("  Example:  /Users/yourname/Downloads/resume.pdf")
    print()
    path = input("  Path: ").strip().strip('"').strip("'")

    if not path:
        print("[ERROR] No path entered.")
        sys.exit(1)
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    if not path.lower().endswith(".pdf"):
        print("[ERROR] Please provide a .pdf file.")
        sys.exit(1)

    return path


# ══════════════════════════════════════════════════════════════════════════════
# DIFFICULTY SELECTOR — terminal menu
# ══════════════════════════════════════════════════════════════════════════════

def _select_difficulty() -> str:
    """Display difficulty menu and return user's choice as a string."""
    print()
    print("  Select interview difficulty level:")
    print()
    print("    1.  Easy    (  8 questions — Beginner      )")
    print("    2.  Medium  ( 16 questions — Intermediate  )")
    print("    3.  Hard    ( 24 questions — Advanced      )")
    print()

    mapping = {"1": "Easy", "2": "Medium", "3": "Hard"}
    while True:
        choice = input("  Enter [1 / 2 / 3] : ").strip()
        if choice in mapping:
            selected = mapping[choice]
            print(f"\n  ✓  Difficulty selected: {selected}\n")
            return selected
        print("  ✗  Invalid — please enter 1, 2, or 3.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Complete terminal flow:

        python question_generator/generator.py
        python question_generator/generator.py path/to/resume.pdf

    Steps:
      1. Select resume PDF (file picker dialog or typed path)
      2. Parse resume using parse_resume()
      3. Show resume summary
      4. Select difficulty (terminal menu: 1/2/3)
      5. Resolve Gemini API key (env var → cache → prompt once)
      6. Generate questions using Gemini AI
         └── Falls back to rule-based if Gemini is unavailable
      7. Print all questions in terminal
      8. Save to generated_questions.json at project root
    """

    # ── Step 1: Select resume PDF ─────────────────────────────────────────────
    if not _HAS_PARSER:
        print("[ERROR] resume_parser module is required but not found. Exiting.")
        sys.exit(1)

    pdf_path = _select_resume_pdf()

    # ── Step 2: Parse resume ──────────────────────────────────────────────────
    print(f"\n[Generator] Parsing resume: {pdf_path}")
    try:
        resume_data = parse_resume(pdf_path)
    except Exception as exc:
        print(f"\n[ERROR] Resume parsing failed: {exc}")
        sys.exit(1)

    # ── Step 3: Show summary ──────────────────────────────────────────────────
    print()
    print("  " + "-" * 46)
    print(f"  Name            : {resume_data.get('name', 'Not detected')}")
    print(f"  Email           : {resume_data.get('email', 'Not detected')}")
    print(f"  Phone           : {resume_data.get('phone', 'Not detected')}")
    print(f"  Skills          : {len(resume_data.get('skills', []))}")
    print(f"  Education       : {len(resume_data.get('education', []))}")
    print(f"  Projects        : {len(resume_data.get('projects', []))}")
    print(f"  Experience      : {len(resume_data.get('experience', []))}")
    print(f"  Certifications  : {len(resume_data.get('certifications', []))}")
    print("  " + "-" * 46)

    # ── Step 4: Select difficulty ─────────────────────────────────────────────
    difficulty = _select_difficulty()

    # ── Step 5: Resolve Gemini API key ────────────────────────────────────────
    api_key = _resolve_api_key()
    if not api_key and _HAS_GENAI:
        api_key = _prompt_and_save_key()

    # ── Step 6: Generate questions ────────────────────────────────────────────
    try:
        result = generate_questions(resume_data, difficulty, api_key=api_key)
    except (ValueError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    # ── Step 7: Print to terminal ─────────────────────────────────────────────
    print_questions(result)

    # ── Step 8: Save to JSON ──────────────────────────────────────────────────
    save_questions(result)
    print(f"\n[Generator] Done! {result['total_questions']} questions generated for {result['candidate_name']}.\n")