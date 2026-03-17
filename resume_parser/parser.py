"""
╔══════════════════════════════════════════════════════════════════╗
║  resume_parser/parser.py                                         ║
║  AI Mock Interview System — Resume Parser Module  v3.0           ║
╠══════════════════════════════════════════════════════════════════╣
║  RUN DIRECTLY      python parser.py                              ║
║  IMPORT            from resume_parser.parser import parse_resume ║
║                    data = parse_resume("resume.pdf")             ║
╠══════════════════════════════════════════════════════════════════╣
║  INSTALL           pip install pdfplumber                        ║
║  tkinter ships with Python on Windows / macOS.                   ║
║  Linux:  sudo apt-get install python3-tk                         ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────
import re
import json
import os
import sys
import tkinter as tk
from tkinter import filedialog

# ── Third-party ───────────────────────────────────────────────────────────
try:
    import pdfplumber
except ImportError:
    print("[ERROR] pdfplumber not installed.  Run:  pip install pdfplumber")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════

WIN_W, WIN_H   = 500, 580
WIN_BG         = "#0E1120"
CARD_BG        = "#161929"
CARD_BORDER    = "#232741"

TXT_TITLE      = "#E8EEFF"
TXT_BODY       = "#7E86AD"
TXT_HINT       = "#3E4466"
TXT_SUCCESS    = "#4ECBA0"
TXT_WARN       = "#F0A060"
TXT_ERROR      = "#F07070"

BTN_IDLE       = "#3A6BF5"
BTN_HOVER      = "#5C87FF"
BTN_PRESS      = "#2850CC"
BTN_TXT        = "#FFFFFF"

ACCENT_LINE    = "#3A6BF5"
DIVIDER        = "#1F2438"

ICON_DOC       = "#1E2540"
ICON_FOLD      = "#0E1120"
ICON_OUTLINE   = "#3A6BF5"
ICON_LINES     = "#2D3358"
ICON_ARROW     = "#4ECBA0"

AUTO_CLOSE_MS  = 5_000     # milliseconds before the window auto-closes


# ══════════════════════════════════════════════════════════════════════════
# SECTION DETECTION PATTERNS
#
# Each pattern matches BOTH the normal spaced form ("WORK EXPERIENCE") AND
# the squished no-space form ("WORKEXPERIENCE") that some PDF renderers
# produce when extracting text from columnar or tightly-set layouts.
#
# Pattern structure:
#   spaced variant   →  \s+ between words
#   squished variant →  words concatenated, no separator
# ══════════════════════════════════════════════════════════════════════════

_SECTION_PATTERNS: dict[str, re.Pattern] = {

    "summary": re.compile(
        r"^\s*("
        # spaced
        r"summary|professional\s+summary|professional summary"
        r"|profile|career\s+objective|career objective"
        r"|objective|about\s+me|overview"
        # squished
        r"|professionalsummary|careerobjective|aboutme"
        r")\s*$",
        re.IGNORECASE,
    ),

    "education": re.compile(
        r"^\s*("
        # spaced
        r"education|academic\s+background|educational\s+background"
        r"|academic\s+qualifications?|qualifications?|academic\s+details?"
        # squished
        r"|academicbackground|educationalbackground|academicqualifications?"
        r"|academicdetails?"
        r")\s*$",
        re.IGNORECASE,
    ),

    "experience": re.compile(
        r"^\s*("
        # spaced
        r"experience|work\s+experience|professional\s+experience"
        r"|employment\s+history|work\s+history|career\s+history"
        r"|internship\s+experience|industry\s+experience|internships?"
        # squished
        r"|workexperience|professionalexperience|employmenthistory"
        r"|workhistory|careerhistory|internshipexperience|industryexperience"
        r")\s*$",
        re.IGNORECASE,
    ),

    "skills": re.compile(
        r"^\s*("
        # spaced
        r"skills?|technical\s+skills?|core\s+competenc\w+"
        r"|technologies|tools?\s*(?:&|and)?\s*technologies?"
        r"|key\s+skills?|programming\s+languages?|expertise|tech\s+stack"
        r"|technical\s+expertise|areas?\s+of\s+expertise"
        # squished
        r"|technicalskills?|corecompetenc\w+|techstack"
        r"|programmingskills?|programmingknowledge"
        r"|keyskills?|technicalknowledge"
        r")\s*$",
        re.IGNORECASE,
    ),

    "projects": re.compile(
        r"^\s*("
        # spaced
        r"projects?|major\s+projects?|personal\s+projects?"
        r"|academic\s+projects?|key\s+projects?|notable\s+projects?"
        r"|side\s+projects?|portfolio|project\s+work|project\s+experience"
        # squished
        r"|majorprojects?|personalprojects?|academicprojects?"
        r"|keyprojects?|notableprojects?|sideprojects?"
        r"|projectwork|projectexperience"
        r")\s*$",
        re.IGNORECASE,
    ),

    "certifications": re.compile(
        r"^\s*("
        # spaced
        r"certifications?|certificates?|licen[sc]es?"
        r"|certifications?\s*(?:&|and)\s*achievements?"
        r"|achievements?|awards?\s*(?:&|and)?\s*achievements?"
        r"|honours?|accomplishments?|courses?\s*(?:&|and)?\s*certifications?"
        r"|training|training\s*(?:&|and)?\s*certifications?"
        # squished
        r"|certifications?achievements?|achievementsawards?"
        r"|certificationsachievements?|trainingcertifications?"
        r")\s*$",
        re.IGNORECASE,
    ),
}

# Processing order matters: summary and education are checked first so
# that ambiguous headings (e.g. "Profile") don't steal experience lines.
_SECTION_ORDER = [
    "summary", "education", "skills", "experience",
    "projects", "certifications",
]

# ── Skill vocabulary ────────────────────────────────────────────────────────

KNOWN_SKILLS: list[str] = [
    # Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#",
    "go", "golang", "ruby", "php", "swift", "kotlin", "rust", "scala",
    "r", "matlab", "bash", "shell", "dart", "elixir", "perl", "haskell",
    "lua", "groovy", "objective-c", "assembly",
    # Web / Frontend
    "html", "css", "html5", "css3", "react", "react.js", "angular",
    "vue", "vue.js", "next.js", "nuxt", "nuxt.js", "svelte",
    "tailwind", "tailwindcss", "bootstrap", "sass", "scss", "less",
    "jquery", "webpack", "vite", "babel", "redux", "mobx",
    # Backend / Frameworks
    "node.js", "nodejs", "express", "express.js", "django", "flask",
    "fastapi", "spring", "spring boot", "rails", "ruby on rails",
    "laravel", "asp.net", ".net", "nestjs", "gin", "fiber", "fastify",
    # Mobile
    "flutter", "react native", "android", "ios", "xamarin",
    # Data Science & ML / AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data engineering",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "opencv",
    "hugging face", "transformers", "xgboost", "lightgbm", "catboost",
    "data analysis", "data visualization", "feature engineering",
    "reinforcement learning", "generative ai", "llm",
    # Databases
    "sql", "mysql", "postgresql", "postgres", "mongodb", "redis",
    "sqlite", "firebase", "cassandra", "elasticsearch", "dynamodb",
    "oracle", "ms sql", "mssql", "neo4j", "couchdb", "mariadb",
    # Cloud & DevOps
    "aws", "amazon web services", "azure", "gcp", "google cloud",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins",
    "ci/cd", "github actions", "gitlab ci", "circleci", "travis ci",
    "git", "github", "gitlab", "bitbucket", "linux", "unix",
    "nginx", "apache", "prometheus", "grafana", "elk stack",
    # Tools & Practices
    "rest api", "restful", "graphql", "grpc", "microservices",
    "agile", "scrum", "kanban", "jira", "confluence", "trello",
    "figma", "sketch", "adobe xd", "postman", "swagger",
    "tableau", "power bi", "looker", "excel", "google analytics",
    # Security
    "cybersecurity", "penetration testing", "ethical hacking",
    "network security", "owasp", "ssl", "tls",
]


# ══════════════════════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION + PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════

# Known section titles that some PDF renderers squish into a single token
# without spaces (e.g. "WORKEXPERIENCE", "MAJORPROJECTS").
# Each tuple: (squished_regex, canonical_heading_to_insert)
_SQUISHED_HEADINGS: list[tuple] = [
    # Must be ordered longest-match first to avoid partial sub-matches
    (re.compile(r"(?<![A-Z])PROFESSIONALEXPERIENCE(?![A-Z])"),  "PROFESSIONAL EXPERIENCE"),
    (re.compile(r"(?<![A-Z])PROFESSIONALSUMMARY(?![A-Z])"),     "PROFESSIONAL SUMMARY"),
    (re.compile(r"(?<![A-Z])WORKEXPERIENCE(?![A-Z])"),          "WORK EXPERIENCE"),
    (re.compile(r"(?<![A-Z])MAJORPROJECTS?(?![A-Z])"),          "MAJOR PROJECTS"),
    (re.compile(r"(?<![A-Z])PERSONALPROJECTS?(?![A-Z])"),       "PERSONAL PROJECTS"),
    (re.compile(r"(?<![A-Z])ACADEMICPROJECTS?(?![A-Z])"),       "ACADEMIC PROJECTS"),
    (re.compile(r"(?<![A-Z])KEYPROJECTS?(?![A-Z])"),            "KEY PROJECTS"),
    (re.compile(r"(?<![A-Z])TECHNICALSKILLS?(?![A-Z])"),        "TECHNICAL SKILLS"),
    (re.compile(r"(?<![A-Z])PROGRAMMINGSKILLS?(?![A-Z])"),      "TECHNICAL SKILLS"),
    (re.compile(r"(?<![A-Z])CERTIFICATIONS?&?ACHIEVEMENTS?(?![A-Z])"), "CERTIFICATIONS & ACHIEVEMENTS"),
    (re.compile(r"(?<![A-Z])ACHIEVEMENTSAWARDS?(?![A-Z])"),     "CERTIFICATIONS & ACHIEVEMENTS"),
    (re.compile(r"(?<![A-Z])TRAININGCERTIFICATIONS?(?![A-Z])"), "CERTIFICATIONS & ACHIEVEMENTS"),
    (re.compile(r"(?<![A-Z])ACADEMICBACKGROUND(?![A-Z])"),      "EDUCATION"),
    (re.compile(r"(?<![A-Z])CAREEROBJECTIVE(?![A-Z])"),         "PROFESSIONAL SUMMARY"),
    (re.compile(r"(?<![A-Z])ABOUTME(?![A-Z])"),                 "PROFESSIONAL SUMMARY"),
]


def _preprocess_text(raw: str) -> str:
    """
    Fix two common PDF extraction artefacts before section splitting:

    1. SQUISHED HEADINGS — some PDF renderers join adjacent words without
       spaces, e.g. "WORKEXPERIENCE", "MAJORPROJECTS", "TECHNICALSKILLS".
       We scan every line; if a squished heading is found *inside* a line
       we split the line so the heading starts on its own line.

    2. ALL-CAPS SECTION MARKERS embedded mid-line — e.g. a line that starts
       with education content but suddenly contains "WORKEXPERIENCE" in the
       middle.  After step 1 those are already on their own lines.

    Returns the cleaned text with section headings on isolated lines.
    """
    lines = raw.split("\n")
    out: list[str] = []

    for line in lines:
        working = line

        # Check every squished-heading pattern against the current line
        for pattern, canonical in _SQUISHED_HEADINGS:
            match = pattern.search(working)
            if match:
                before = working[:match.start()].strip()
                after  = working[match.end():].strip()
                # Reconstruct: content before (if any) → heading → content after (if any)
                parts = []
                if before:
                    parts.append(before)
                parts.append(canonical)      # isolated heading line
                if after:
                    parts.append(after)
                working = "\n".join(parts)
                break   # one squished heading per line is enough

        out.append(working)

    return "\n".join(out)


def _read_pdf(file_path: str) -> str:
    """
    Read all pages of a PDF with pdfplumber and return combined text.
    Applies _preprocess_text() to normalise squished section headings.

    Raises:
        FileNotFoundError – path does not exist.
        RuntimeError      – file opened but no text extracted (image PDF).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")

    try:
        pages: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)

        if not pages:
            raise RuntimeError(
                "No readable text found. The PDF may be image-based or "
                "password-protected. Try running OCR first (e.g. pytesseract)."
            )

        raw = "\n".join(pages)
        return _preprocess_text(raw)   # ← normalise headings before splitting

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"pdfplumber could not read the file: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION SPLITTING  (regex-based boundary detection)
# ══════════════════════════════════════════════════════════════════════════

def _split_sections(text: str) -> dict[str, str]:
    """
    Walk the resume line by line.  When a line matches one of the compiled
    section-heading patterns the parser switches "active section" and all
    subsequent lines accumulate under that label — until the next heading.

    Lines before the first recognised heading go into 'header' (used for
    name / contact extraction only).

    Returns:
        Dict of { section_name: raw_text_block }.
    """
    lines = text.split("\n")
    buckets: dict[str, list[str]] = {"header": []}
    active = "header"

    for line in lines:
        stripped = line.strip()

        # Try every pattern in order; first match wins.
        matched_section: str | None = None
        for sec in _SECTION_ORDER:
            if _SECTION_PATTERNS[sec].match(stripped):
                matched_section = sec
                break

        if matched_section:
            active = matched_section
            buckets.setdefault(active, [])
        else:
            buckets.setdefault(active, []).append(line)

    # Collapse each list → single text block, stripped of leading/trailing blank lines
    return {k: "\n".join(v).strip() for k, v in buckets.items()}


# ══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL FIELD EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════

def _extract_name(header_text: str) -> str:
    """
    Detect the candidate's name from the 'header' section (top of resume).

    Strategy: the name is almost always one of the first 8 non-empty lines;
    it contains 2–4 title-cased words with no digits, URLs, or contact signals.
    """
    skip_signals = [
        "@", "http", "www.", "linkedin", "github", "phone",
        "email", "resume", "cv", "objective", "summary",
        "profile", "+", "|", "/", "–", "—",
    ]
    lines = [l.strip() for l in header_text.split("\n") if l.strip()]

    for line in lines[:8]:
        low = line.lower()
        if any(sig in low for sig in skip_signals):
            continue
        # Reject lines that contain digits (phone / year) or long sentences
        if re.search(r"\d", line):
            continue
        words = line.split()
        if 2 <= len(words) <= 5 and all(
            re.match(r"^[A-Z][a-zA-Z'\-\.]{0,}$", w) for w in words
        ):
            return line
    return ""


def extract_email(text: str) -> str:
    """Return the first valid email address found in the text."""
    m = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
    )
    return m.group(0) if m else ""


def extract_phone(text: str) -> str:
    """
    Return the first plausible phone number (7–15 digits).
    Handles +91-XXXXX-XXXXX, (123) 456-7890, 10-digit plain, etc.
    """
    for candidate in re.findall(r"(\+?\d[\d\s\-().]{7,}\d)", text):
        digits = re.sub(r"\D", "", candidate)
        if 7 <= len(digits) <= 15:
            return candidate.strip()
    return ""


def extract_skills(full_text: str, skills_section: str = "") -> list[str]:
    """
    Match skills from KNOWN_SKILLS against the resume using word-boundary
    regex.  Searches the dedicated skills section first (preferred), then
    falls back to full-text scan to capture inline mentions.

    Returns a sorted, deduplicated list.
    """
    # Search both spaces: skills section (higher confidence) + full text
    # Combine them so inline mentions are never missed.
    search_corpus = (
        (skills_section + "\n" + full_text) if skills_section else full_text
    ).lower()

    found: set[str] = set()
    for skill in KNOWN_SKILLS:
        pattern = r"(?<![a-zA-Z])" + re.escape(skill) + r"(?![a-zA-Z])"
        if re.search(pattern, search_corpus):
            # Display formatting
            if " " in skill:
                display = skill.capitalize()
            elif skill in {"c", "r", "go"}:
                display = skill.upper()
            else:
                display = skill.title()
            found.add(display)

    return sorted(found)


# ── Bullet classification ────────────────────────────────────────────────
#
# TOP-LEVEL bullets  →  start a new entry
#   • (U+2022)  ● (U+25CF)  ▪ (U+25AA)
#
# SUB-LEVEL bullets  →  detail line, always appended to the current entry
#   ◦ (U+25E6)  ○ (U+25CB)
#
# KEY INSIGHT for squished PDFs:
#   pdfplumber strips ALL leading whitespace, so we CANNOT rely on
#   indentation to tell sub-bullets apart.  Instead we use the bullet
#   character itself:
#     ◦  is ALWAYS a sub-bullet, regardless of indentation.
#     •  is ALWAYS a top-level bullet.
#
# This exactly matches Akshit's resume where pdfplumber produces lines like:
#   • MovieRecommendationSystem|...          ← new project entry
#   ◦ Designedandimplemented...              ← detail of that project
#   ◦ Performeddatapreprocessing...          ← detail of that project

# Strips any bullet char from the start of a cleaned string
_BULLET_STRIP_RE = re.compile(
    r"^[\•\●\▪\◦\○\-\*\–\—\u2022\u25cf\u25aa\u25e6\u25cb]\s*"
)

# TOP-level bullet chars only (filled / solid shapes)
_TOP_BULLET_RE = re.compile(
    r"^[\•\●\▪\u2022\u25cf\u25aa]\s*"
)

# SUB-level bullet chars — hollow shapes that always signal a detail line
_SUB_BULLET_RE = re.compile(
    r"^[\◦\○\u25e6\u25cb]\s*"
)

# Lines that are pure visual noise (separators, standalone page numbers)
_NOISE_RE = re.compile(
    r"^[\-\=\*\.\•\▪\◦\~\_\|]{2,}$"
    r"|^\d{1,3}$"
)


def _clean_section_entries(section_text: str, min_len: int = 8) -> list[str]:
    """
    Convert a raw section text block into a clean list of logical entries.

    Each TOP-level bullet (•●▪) starts a new entry.
    Each SUB-level bullet (◦○) is appended as a detail to the current entry.

    This correctly handles squished PDFs where pdfplumber strips indentation —
    the bullet character itself (filled vs hollow) is the only reliable
    signal for grouping.

    Example input (Akshit's resume after preprocessing):
        • MovieRecommendationSystem|Python,Scikit-learn
        ◦ Designedandimplementeda ML-basedrecommendation system.
        ◦ Performeddatapreprocessing,featureengineering.
        • MovieReviewSentimentAnalysis|Python,NLTK
        ◦ DevelopedanNLP-basedsentimentanalysismodel.

    Expected output (3 grouped entries, not 7 flat ones):
        [
          "MovieRecommendationSystem|... – Designedandimplemented... – Performeddata...",
          "MovieReviewSentimentAnalysis|... – DevelopedanNLP...",
          ...
        ]
    """
    if not section_text:
        return []

    # ── Dense-blob guard ─────────────────────────────────────────────────
    # If the entire section arrived as a single long line (rare edge case),
    # split on top-level bullet chars first.
    newline_count = section_text.count("\n")
    if newline_count < 3 and len(section_text) > 200:
        fragments = re.split(
            r"(?=[\•\●\▪\u2022\u25cf\u25aa])",
            section_text,
        )
        entries = []
        for frag in fragments:
            s = _BULLET_STRIP_RE.sub("", frag).strip()
            if len(s) >= min_len and not _NOISE_RE.match(s):
                entries.append(s)
        if entries:
            return entries
        # Fall through to line-by-line if split produced nothing useful

    # ── Line-by-line grouping ─────────────────────────────────────────────
    entries: list[str] = []
    current: list[str] = []   # accumulates lines for the current entry

    def _flush() -> None:
        """Save the current entry block and reset."""
        if current:
            merged = " ".join(current).strip()
            if len(merged) >= min_len:
                entries.append(merged)
            current.clear()

    for line in section_text.split("\n"):
        s = line.strip()

        # ── Skip blank and decoration lines ──────────────────────────────
        if not s or _NOISE_RE.match(s):
            # A blank line between entries — flush if not already empty
            _flush()
            continue

        # ── Classify the line ─────────────────────────────────────────────
        if _SUB_BULLET_RE.match(s):
            # ◦ → sub-detail: always append to the current entry
            detail = _BULLET_STRIP_RE.sub("", s).strip()
            if detail and current:
                # Append detail with a separator so it stays readable
                current.append("– " + detail)
            # If there's no current entry yet (orphaned sub-bullet), skip it

        elif _TOP_BULLET_RE.match(s):
            # • → new top-level entry: flush previous, start fresh
            _flush()
            clean = _BULLET_STRIP_RE.sub("", s).strip()
            if clean:
                current.append(clean)

        else:
            # Plain line (no bullet) — continuation of the current block
            # Also handles indented sub-bullets whose indentation was kept
            if re.match(r"^\s{2,}", line):
                # Indented line → treat as detail even without a bullet char
                detail = s
                if detail and current:
                    current.append("– " + detail)
            else:
                current.append(s)

    _flush()
    return entries


def _extract_certifications(cert_section: str, full_text: str) -> list[str]:
    """
    Extract certification entries.

    Priority:
      1. Use the dedicated certifications section when detected.
      2. Fall back to keyword-scanning every line of the full text.

    The section text may be a single bullet-packed line (squished PDF)
    so we split on bullet chars as well as newlines.
    """
    if cert_section:
        entries = _clean_section_entries(cert_section)
        if entries:
            return entries

    # Fallback: scan every line of the full text for cert-related keywords
    cert_keywords = re.compile(
        r"\b(certified|certification|certificate|credential|licence|license"
        r"|aws certified|google certified|microsoft certified|azure"
        r"|coursera|udemy|edx|nptel|cisco|comptia|pmp|scrum master"
        r"|ceh|oscp|adobe|workshop|nanodegree|specialization)\b",
        re.IGNORECASE,
    )
    found: list[str] = []
    # Split the full text on both newlines and bullet characters so squished
    # certification lists are also handled.
    for raw_line in re.split(r"[\n\•\●\▪\u2022\u25cf]", full_text):
        s = raw_line.strip().lstrip("-–•●▪◦ ")
        if len(s) >= 10 and cert_keywords.search(s):
            found.append(s)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped




# ══════════════════════════════════════════════════════════════════════════
# EDUCATION EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════

_DEGREE_SPLIT_RE = re.compile(
    r"(?="
    r"(?:B\.?Tech|M\.?Tech"
    r"|B\.?E\.?|M\.?E\.?"
    r"|B\.?Sc\.?|M\.?Sc\.?"
    r"|B\.?C\.?A\.?|M\.?C\.?A\.?"
    r"|B\.?B\.?A\.?|M\.?B\.?A\.?"
    r"|B\.?Com\.?|M\.?Com\.?"
    r"|Ph\.?D\.?|Doctorate"
    r"|Class\s*(?:XII|XI|X|IX|\d{1,2})"
    r"|High\s+School|Senior\s+Secondary|Intermediate|Diploma|Matriculation"
    r"))",
    re.IGNORECASE,
)

_GRADE_RE = re.compile(
    r"(\d{1,2}\.\d{1,2}\s*/\s*10(?:\.\d)?"
    r"|\d{2,3}(?:\.\d{1,2})?\s*%"
    r"|\d{1,2}\.\d{1,2}\s*CGPA"
    r")",
    re.IGNORECASE,
)

_EDU_HEADER_RE = re.compile(
    r"\b(Degree|Institute|Institution|Year|GPA|Percentage|Grade|Marks|Board|Stream)\b",
    re.IGNORECASE,
)


def _split_degree_institution(text: str):
    """Split a fragment like 'B.Tech(CSE-ML&AI) GraphicEraUniversity,Dehradun'
    into (degree, institution) using comma and parenthesis as signals."""
    text   = text.strip()
    tokens = text.split()
    degree_end = 1
    for i in range(1, len(tokens)):
        tok = tokens[i]
        if "," in tok:
            degree_end = i
            break
        if re.search(r"[(\[]", tok):
            degree_end = i + 1
            continue
        if i >= 2:
            degree_end = i
            break
    else:
        degree_end = len(tokens)
    return " ".join(tokens[:degree_end]).strip(), " ".join(tokens[degree_end:]).strip()


def _extract_education(edu_section: str) -> list:
    """
    Parse the education section into a list of structured entry dicts.

    Each dict contains any of:
        "degree"      - qualification name  e.g. "B.Tech(CSE-ML&AI)"
        "institution" - school/university   e.g. "GraphicEraUniversity,Dehradun"
        "year"        - year or range       e.g. "2022-2026" or "2021"
        "grade"       - score               e.g. "7.35/10.0" or "79%"

    Works with both squished blobs (table PDFs) and clean multi-line text.
    Each entry can be used by question_generator to ask targeted questions:
        "What was your GPA in B.Tech?"
        "Which board was your Class XII under?"
        "What percentage did you score in Class X?"
    """
    if not edu_section:
        return []

    flat = " ".join(edu_section.split())
    flat = _EDU_HEADER_RE.sub(" ", flat)
    flat = re.sub(r"\s{2,}", " ", flat).strip()

    fragments = _DEGREE_SPLIT_RE.split(flat)
    fragments = [f.strip() for f in fragments if f.strip() and len(f.strip()) > 4]

    entries = []
    for frag in fragments:
        grade_match = _GRADE_RE.search(frag)
        grade = grade_match.group(0).strip() if grade_match else ""

        year_match = re.search(r"\b(\d{4}\s*[-\u2013]\s*\d{4}|\d{4})\b", frag)
        year = year_match.group(0).strip() if year_match else ""

        remainder = frag
        if grade:
            remainder = remainder.replace(grade_match.group(0), " ")
        if year:
            remainder = remainder.replace(year_match.group(0), " ")
        remainder = re.sub(r"\s{2,}", " ", remainder).strip().strip("-| ,")

        degree, institution = _split_degree_institution(remainder)

        entry: dict = {"degree": degree}
        if institution:
            entry["institution"] = institution
        if year:
            entry["year"] = year
        if grade:
            entry["grade"] = grade

        entries.append(entry)

    return entries


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API  — the only function external modules need
# ══════════════════════════════════════════════════════════════════════════

def parse_resume(file_path: str) -> dict:
    """
    Parse a PDF resume and return all extracted data as a structured dict.

    This is the single public entry point consumed by question_generator
    and any other downstream module.

    Args:
        file_path: Path to a PDF resume file (absolute or relative).

    Returns:
        {
            "name":           str,
            "email":          str,
            "phone":          str,
            "skills":         list[str],
            "education":      list[str],
            "projects":       list[str],
            "experience":     list[str],
            "certifications": list[str],
        }

    Raises:
        FileNotFoundError  – file_path does not exist.
        RuntimeError       – PDF cannot be read or contains no text.

    Example (question_generator usage):
        from resume_parser.parser import parse_resume
        data = parse_resume("uploads/candidate.pdf")
        skills     = data["skills"]
        projects   = data["projects"]
        experience = data["experience"]
    """
    print("\n[ResumeParser] ── Starting extraction ──────────────────────")

    # 1 ── Read raw text from the PDF
    raw_text = _read_pdf(file_path)
    print("[ResumeParser] Text extracted from PDF.")

    # 2 ── Split into labelled sections
    sections = _split_sections(raw_text)
    detected = [k for k in sections if k != "header" and sections[k].strip()]
    print(f"[ResumeParser] Sections detected  : {detected}")

    # 3 ── Extract every field
    name  = _extract_name(sections.get("header", ""))
    email = extract_email(raw_text)
    phone = extract_phone(raw_text)

    skills = extract_skills(
        full_text=raw_text,
        skills_section=sections.get("skills", ""),
    )

    education      = _extract_education(sections.get("education", ""))
    projects       = _clean_section_entries(sections.get("projects", ""))
    experience     = _clean_section_entries(sections.get("experience", ""))
    certifications = _extract_certifications(
        sections.get("certifications", ""), raw_text
    )

    data = {
        "name":           name,
        "email":          email,
        "phone":          phone,
        "skills":         skills,
        "education":      education,
        "projects":       projects,
        "experience":     experience,
        "certifications": certifications,
    }

    print(f"[ResumeParser] Skills      found  : {len(skills)}")
    print(f"[ResumeParser] Education   found  : {len(education)}")
    print(f"[ResumeParser] Projects    found  : {len(projects)}")
    print(f"[ResumeParser] Experience  found  : {len(experience)}")
    print(f"[ResumeParser] Certs       found  : {len(certifications)}")
    print("[ResumeParser] ── Extraction complete ──────────────────────\n")
    return data


# ══════════════════════════════════════════════════════════════════════════
# TERMINAL OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _print_json(data: dict) -> None:
    print("=" * 64)
    print("              EXTRACTED RESUME DATA  (JSON)")
    print("=" * 64)
    print(json.dumps(data, indent=4))
    print("=" * 64)


def _print_summary(data: dict) -> None:
    print("\n── QUICK SUMMARY " + "─" * 46)
    print(f"  Name                  : {data['name']  or 'Not detected'}")
    print(f"  Email                 : {data['email'] or 'Not detected'}")
    print(f"  Phone                 : {data['phone'] or 'Not detected'}")
    print(f"  Skills found          : {len(data['skills'])}")
    print(f"  Education entries     : {len(data['education'])}")
    for edu in data['education']:
        deg  = edu.get('degree','?')
        inst = edu.get('institution','')
        yr   = edu.get('year','')
        gr   = edu.get('grade','')
        line = f"    → {deg}"
        if inst: line += f"  |  {inst}"
        if yr:   line += f"  |  {yr}"
        if gr:   line += f"  |  {gr}"
        print(line)
    print(f"  Projects found        : {len(data['projects'])}")
    print(f"  Experience entries    : {len(data['experience'])}")
    print(f"  Certifications found  : {len(data['certifications'])}")
    print("─" * 64 + "\n")


# ══════════════════════════════════════════════════════════════════════════
# ROUNDED BUTTON  (Frame + Label composite — works on ALL Python/Tk versions)
#
# Why not Canvas?
# tk.Canvas subclasses crash on Python 3.14 + Tk 9 (macOS Homebrew) with
# "TclError: bad argument '<id>': must be name of window" when the Canvas
# Tcl internal command isn't registered yet at pack() time.  Using a plain
# tk.Frame avoids the Canvas lifecycle entirely and is 100% stable.
# ══════════════════════════════════════════════════════════════════════════

class RoundedButton(tk.Frame):
    """
    A Frame-based button widget with hover / press colour states.

    Looks like a pill button because the outer Frame is given a contrasting
    background (the border colour) and the inner Frame carries the fill
    colour with padx/pady inset — creating the illusion of a border radius.

    Compatible with Python 3.8 → 3.14+ and all Tk versions.
    """

    _DISABLED_BG = "#2A2F48"

    def __init__(
        self,
        parent,
        text: str,
        command=None,
        bg_idle: str  = BTN_IDLE,
        bg_hover: str = BTN_HOVER,
        bg_press: str = BTN_PRESS,
        fg: str       = BTN_TXT,
        font_spec: tuple = ("Helvetica", 12, "bold"),
        padx: int = 28,
        pady: int = 12,
        **kwargs,
    ):
        # Outer frame = "border ring" — 1 px darker than the button fill
        border_colour = self._darken(bg_idle)
        super().__init__(parent, bg=border_colour, padx=1, pady=1, **kwargs)

        self._idle    = bg_idle
        self._hover   = bg_hover
        self._press   = bg_press
        self._enabled = True
        self._command = command

        # Inner frame = the visible button body
        self._body = tk.Frame(self, bg=bg_idle, padx=padx, pady=pady,
                              cursor="hand2")
        self._body.pack(fill="both", expand=True)

        # Label centred inside the body
        self._lbl = tk.Label(
            self._body,
            text=text,
            bg=bg_idle,
            fg=fg,
            font=font_spec,
            cursor="hand2",
        )
        self._lbl.pack()

        # Bind hover / press events to both inner widgets so the whole
        # button area responds, not just the label.
        for widget in (self._body, self._lbl):
            widget.bind("<Enter>",           self._on_enter)
            widget.bind("<Leave>",           self._on_leave)
            widget.bind("<ButtonPress-1>",   self._on_press)
            widget.bind("<ButtonRelease-1>", self._on_release)

    # ── Colour helpers ────────────────────────────────────────────────────

    @staticmethod
    def _darken(hex_colour: str, amount: int = 20) -> str:
        """Return a slightly darkened version of a #RRGGBB colour."""
        hex_colour = hex_colour.lstrip("#")
        r, g, b = (int(hex_colour[i:i+2], 16) for i in (0, 2, 4))
        r, g, b = max(0, r - amount), max(0, g - amount), max(0, b - amount)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _apply_colour(self, colour: str) -> None:
        """Repaint the body and label with the given background colour."""
        self._body.config(bg=colour)
        self._lbl.config(bg=colour)
        self.config(bg=self._darken(colour))

    # ── Public API ────────────────────────────────────────────────────────

    def set_enabled(self, state: bool) -> None:
        """Enable or disable the button (visual + interaction)."""
        self._enabled = state
        colour = self._idle if state else self._DISABLED_BG
        self._apply_colour(colour)
        cursor = "hand2" if state else ""
        self._body.config(cursor=cursor)
        self._lbl.config(cursor=cursor)

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_enter(self, _) -> None:
        if self._enabled:
            self._apply_colour(self._hover)

    def _on_leave(self, _) -> None:
        target = self._idle if self._enabled else self._DISABLED_BG
        self._apply_colour(target)

    def _on_press(self, _) -> None:
        if self._enabled:
            self._apply_colour(self._press)

    def _on_release(self, _) -> None:
        if self._enabled:
            self._apply_colour(self._hover)
            if self._command:
                self._command()


# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT UPLOAD ICON  (Canvas — zero image files required)
# ══════════════════════════════════════════════════════════════════════════

def _build_icon(parent) -> tk.Canvas:
    """Draw a document-with-upload-arrow icon using only Canvas primitives."""
    c = tk.Canvas(parent, width=72, height=72,
                  bg=CARD_BG, highlightthickness=0)
    # Document body
    c.create_rectangle(10, 5, 50, 63,
                       fill=ICON_DOC, outline=ICON_OUTLINE, width=2)
    # Dog-ear fold
    c.create_polygon(38, 5, 50, 17, 38, 17,
                     fill=ICON_FOLD, outline=ICON_OUTLINE, width=2)
    # Text-lines on the document
    for y, x_end in [(28, 44), (35, 44), (42, 38), (49, 32)]:
        c.create_line(16, y, x_end, y, fill=ICON_LINES, width=1.5,
                      capstyle="round")
    # Upload arrow (bottom-right area)
    c.create_line(59, 60, 59, 42, fill=ICON_ARROW, width=2.5, capstyle="round")
    c.create_line(54, 48, 59, 42, fill=ICON_ARROW, width=2.5, capstyle="round")
    c.create_line(64, 48, 59, 42, fill=ICON_ARROW, width=2.5, capstyle="round")
    c.create_line(53, 63, 65, 63, fill=ICON_ARROW, width=2.5, capstyle="round")
    return c


# ══════════════════════════════════════════════════════════════════════════
# COUNTDOWN BAR  — thin animated progress strip
# ══════════════════════════════════════════════════════════════════════════

class CountdownBar(tk.Canvas):
    """
    A horizontal bar that drains from full width to zero over
    AUTO_CLOSE_MS milliseconds, giving the user a visual countdown.
    """

    BAR_H = 3

    def __init__(self, parent, total_ms: int, width: int):
        super().__init__(
            parent, width=width, height=self.BAR_H,
            bg=CARD_BG, highlightthickness=0,
        )
        self._total    = total_ms
        self._elapsed  = 0
        self._width    = width
        self._tick     = 50        # update every 50 ms (~20 fps)
        self._rect_id  = self.create_rectangle(
            0, 0, width, self.BAR_H, fill=BTN_IDLE, outline="",
        )
        self._running  = False

    def start(self) -> None:
        self._running = True
        self._tick_fn()

    def stop(self) -> None:
        self._running = False

    def _tick_fn(self) -> None:
        if not self._running:
            return
        self._elapsed += self._tick
        fraction = max(0.0, 1.0 - self._elapsed / self._total)
        self.coords(self._rect_id, 0, 0, int(self._width * fraction), self.BAR_H)
        if fraction > 0:
            self.after(self._tick, self._tick_fn)


# ══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION CLASS
# ══════════════════════════════════════════════════════════════════════════

class ResumeParserApp:
    """
    Dark-themed tkinter window with:
      • Upload card (icon + headline + browse button)
      • Results row (stat pills for skills / projects / experience / certs)
      • Animated countdown bar before auto-close
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._timer_id: str | None = None
        self._build_window()
        self._build_ui()

    # ── Window ─────────────────────────────────────────────────────────────

    def _build_window(self) -> None:
        self.root.title("AI Mock Interview Resume Parser")
        self.root.resizable(False, False)
        self.root.configure(bg=WIN_BG)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x  = (sw - WIN_W) // 2
        y  = (sh - WIN_H) // 2
        self.root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

    # ── UI layout ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Compose all widgets inside a centred card on the dark background."""

        # ── Full-window centering wrapper ──────────────────────────────────
        wrapper = tk.Frame(self.root, bg=WIN_BG)
        wrapper.place(relx=0.5, rely=0.5, anchor="center")

        # ── Card (outer border frame + inner surface) ──────────────────────
        border = tk.Frame(wrapper, bg=CARD_BORDER, padx=2, pady=2)
        border.pack()
        card = tk.Frame(border, bg=CARD_BG, padx=50, pady=36)
        card.pack()

        CARD_INNER_W = WIN_W - 120   # approximate inner card content width

        # ── Blue accent bar at top ─────────────────────────────────────────
        tk.Frame(card, bg=ACCENT_LINE, height=3,
                 width=CARD_INNER_W).pack(pady=(0, 26))

        # ── Upload icon ────────────────────────────────────────────────────
        _build_icon(card).pack(pady=(0, 20))

        # ── Headline ───────────────────────────────────────────────────────
        tk.Label(
            card,
            text="Upload your resume to start\nthe mock interview",
            bg=CARD_BG, fg=TXT_TITLE,
            font=("Georgia", 17, "bold"),
            justify="center",
        ).pack(pady=(0, 8))

        # ── Subtitle ───────────────────────────────────────────────────────
        tk.Label(
            card,
            text="Supported format:  PDF",
            bg=CARD_BG, fg=TXT_HINT,
            font=("Courier", 10),
        ).pack(pady=(0, 24))

        # ── Browse button ──────────────────────────────────────────────────
        self.browse_btn = RoundedButton(
            card,
            text="  Browse Resume  ",
            command=self._on_browse,
            font_spec=("Helvetica", 12, "bold"),
            padx=32,
            pady=13,
        )
        self.browse_btn.pack()

        # ── Status label ───────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="No file selected.")
        self._status_lbl = tk.Label(
            card, textvariable=self._status_var,
            bg=CARD_BG, fg=TXT_BODY,
            font=("Helvetica", 9),
            wraplength=CARD_INNER_W, justify="center",
        )
        self._status_lbl.pack(pady=(16, 0))

        # ── Thin divider ───────────────────────────────────────────────────
        tk.Frame(card, bg=DIVIDER, height=1,
                 width=CARD_INNER_W).pack(pady=(20, 16))

        # ── Countdown label + bar (hidden until auto-close starts) ─────────
        self._countdown_frame = tk.Frame(card, bg=CARD_BG)
        self._countdown_lbl = tk.Label(
            self._countdown_frame,
            text="", bg=CARD_BG, fg=TXT_HINT,
            font=("Courier", 8),
        )
        self._countdown_lbl.pack(pady=(0, 4))
        self._cbar = CountdownBar(
            self._countdown_frame,
            total_ms=AUTO_CLOSE_MS,
            width=CARD_INNER_W,
        )
        self._cbar.pack()

    # ── Browse callback ─────────────────────────────────────────────────────

    def _on_browse(self) -> None:
        """Handle 'Browse Resume' click: open dialog → parse → display."""
        self._set_status("Opening file explorer …", TXT_BODY)
        self.root.update()

        file_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select Your Resume PDF",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
        )

        if not file_path:
            self._set_status("No file selected.", TXT_HINT)
            return

        filename = os.path.basename(file_path)
        self._set_status(f"Parsing  {filename} …", TXT_BODY)
        self.browse_btn.set_enabled(False)
        self.root.update()

        # ── Parse ──────────────────────────────────────────────────────────
        try:
            result = parse_resume(file_path)
        except (FileNotFoundError, RuntimeError) as err:
            self._set_status(f"Error: {err}", TXT_ERROR)
            self.browse_btn.set_enabled(True)
            return

        self.browse_btn.set_enabled(True)

        # ── Terminal output ────────────────────────────────────────────────
        _print_json(result)
        _print_summary(result)

        # ── GUI success state ──────────────────────────────────────────────
        self._set_status("Resume uploaded successfully.", TXT_SUCCESS)
        self._start_countdown()

    # ── Auto-close countdown ──────────────────────────────────────────────

    def _start_countdown(self) -> None:
        """Show the countdown bar and schedule window destruction."""
        self._countdown_frame.pack(pady=(12, 0))
        self._countdown_lbl.config(
            text=f"Window closes automatically in "
                 f"{AUTO_CLOSE_MS // 1000} seconds …"
        )
        self._cbar.start()

        # Cancel any previous timer (user browsed a second resume)
        if self._timer_id:
            self.root.after_cancel(self._timer_id)

        self._timer_id = self.root.after(AUTO_CLOSE_MS, self.root.destroy)

    # ── Helper ────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, colour: str) -> None:
        self._status_var.set(msg)
        self._status_lbl.config(fg=colour)


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Complete program flow
    ─────────────────────
    1.  python parser.py
    2.  Dark GUI window opens — centred, no dialog yet.
    3.  User clicks 'Browse Resume'.
    4.  Native OS file picker opens (PDF only).
    5.  User selects resume.pdf.
    6.  Status → 'Parsing …'
    7.  parse_resume() extracts all fields.
    8.  Terminal prints full JSON + quick summary.
    9.  GUI shows  "Resume uploaded successfully."
    10. Four stat pills appear: Skills · Projects · Experience · Certs
    11. Countdown bar drains over 10 seconds.
    12. Window auto-closes.
    """
    root = tk.Tk()
    ResumeParserApp(root)
    root.mainloop()