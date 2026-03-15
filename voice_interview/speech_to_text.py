"""
Member 2 - Module B: Answer Quality Analyser using Llama (via Ollama)
  + Voice Extension: record_audio → speech_to_text → run_voice_interview
======================================================================
Evaluates whether a candidate's answer is:
  1. High quality / genuinely knowledgeable
  2. Generic / AI-generated / copied
  3. Off-topic / incomplete

Core deps  :  pip install requests
Voice deps :  pip install sounddevice scipy openai-whisper numpy
Run Llama3 :  ollama pull llama3  &&  ollama serve
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing      import Optional

import requests   # pip install requests


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE_URL  = "http://localhost:11434"
LLAMA_MODEL      = "llama3"
REQUEST_TIMEOUT  = 120


# ════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AnswerAnalysis:
    question          : str
    answer            : str
    relevance_score   : float = 0.0
    depth_score       : float = 0.0
    clarity_score     : float = 0.0
    overall_score     : float = 0.0
    ai_generated_prob : float = 0.0
    is_generic        : bool  = False
    suspicious        : bool  = False
    strengths         : list  = field(default_factory=list)
    weaknesses        : list  = field(default_factory=list)
    suggestion        : str   = ""
    verdict           : str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ════════════════════════════════════════════════════════════════════════════
# OLLAMA CLIENT
# ════════════════════════════════════════════════════════════════════════════

def _call_ollama(prompt: str, system: str = "") -> str:
    payload = {
        "model"   : LLAMA_MODEL,
        "stream"  : False,
        "messages": [],
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json    = payload,
            timeout = REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama. Make sure it is running:\n"
            "  ollama serve\n"
            "  ollama pull llama3"
        )


def check_ollama_health() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are an expert technical interview evaluator at a top technology company.
Your job is to assess candidate answers for both quality AND authenticity.
You must detect AI-generated or copy-pasted responses.

ALWAYS respond with ONLY a valid JSON object — no markdown, no explanation.
""".strip()


EVALUATION_PROMPT = """
Evaluate the following interview answer:

QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

ROLE / JOB CONTEXT:
{role}

Return a JSON object with EXACTLY these keys:
{{
  "relevance_score"   : <int 0-10>,
  "depth_score"       : <int 0-10>,
  "clarity_score"     : <int 0-10>,
  "overall_score"     : <float 0-10, weighted average>,
  "ai_generated_prob" : <float 0.0-1.0>,
  "is_generic"        : <bool>,
  "suspicious"        : <bool>,
  "strengths"         : [<str>, ...],
  "weaknesses"        : [<str>, ...],
  "suggestion"        : <str>,
  "verdict"           : <one of: "EXCELLENT","GOOD","AVERAGE","POOR","SUSPICIOUS">
}}

Scoring guide:
- relevance_score : 10 = perfectly addresses the question
- depth_score     : 10 = expert-level technical detail
- clarity_score   : 10 = crystal clear, well-structured
- ai_generated_prob: 0.0 = clearly human; 0.7+ = likely AI/copied
- suspicious      : set true if ai_generated_prob >= 0.6
- verdict SUSPICIOUS if answer feels AI-generated or plagiarised
""".strip()


FOLLOWUP_PROMPT = """
Based on this interview exchange, generate 2 targeted follow-up questions
to probe deeper into the candidate's understanding.

ORIGINAL QUESTION: {question}
CANDIDATE ANSWER: {answer}
EVALUATION VERDICT: {verdict}

Return a JSON array of exactly 2 follow-up question strings.
Example: ["question 1", "question 2"]
""".strip()


# ════════════════════════════════════════════════════════════════════════════
# CORE ANALYSER  ── UNCHANGED ──
# ════════════════════════════════════════════════════════════════════════════

def _parse_json_safe(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def analyse_answer(
    question : str,
    answer   : str,
    role     : str = "Software Engineer",
) -> AnswerAnalysis:
    """
    Main function: send question + answer to Llama and return AnswerAnalysis.
    """
    if not answer.strip():
        return AnswerAnalysis(
            question   = question,
            answer     = answer,
            verdict    = "POOR",
            weaknesses = ["No answer provided"],
        )

    prompt = EVALUATION_PROMPT.format(
        question = question,
        answer   = answer,
        role     = role,
    )

    raw  = _call_ollama(prompt, system=SYSTEM_PROMPT)
    data = _parse_json_safe(raw)

    if not data:
        print(f"[Analyser] ⚠  Could not parse LLM response:\n{raw[:300]}")
        return AnswerAnalysis(question=question, answer=answer, verdict="POOR")

    return AnswerAnalysis(
        question          = question,
        answer            = answer,
        relevance_score   = float(data.get("relevance_score",   0)),
        depth_score       = float(data.get("depth_score",       0)),
        clarity_score     = float(data.get("clarity_score",     0)),
        overall_score     = float(data.get("overall_score",     0)),
        ai_generated_prob = float(data.get("ai_generated_prob", 0.0)),
        is_generic        = bool(data.get("is_generic",  False)),
        suspicious        = bool(data.get("suspicious",  False)),
        strengths         = data.get("strengths",  []),
        weaknesses        = data.get("weaknesses", []),
        suggestion        = data.get("suggestion", ""),
        verdict           = data.get("verdict",    "AVERAGE"),
    )


def generate_followup_questions(analysis: AnswerAnalysis) -> list[str]:
    """
    Generate 2 dynamic follow-up questions based on the candidate's answer.
    Integrates with Member 1's question generator pipeline.
    """
    prompt = FOLLOWUP_PROMPT.format(
        question = analysis.question,
        answer   = analysis.answer,
        verdict  = analysis.verdict,
    )
    raw = _call_ollama(prompt, system=SYSTEM_PROMPT)

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result[:2]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())[:2]
        except Exception:
            pass

    return ["Can you elaborate on that?", "Give me a concrete example."]


# ════════════════════════════════════════════════════════════════════════════
# CHEATING FLAG HELPER  ── UNCHANGED ──
# ════════════════════════════════════════════════════════════════════════════

def get_cheating_signal(analysis: AnswerAnalysis) -> dict:
    """
    Returns a standardised signal dict for the central Cheating Engine.
    Keys match the interface expected by the SVM cheating model (Step 11).
    """
    return {
        "source"           : "answer_analysis",
        "ai_generated_prob": analysis.ai_generated_prob,
        "is_generic"       : int(analysis.is_generic),
        "suspicious"       : int(analysis.suspicious),
        "overall_score"    : analysis.overall_score,
        "verdict"          : analysis.verdict,
    }


# ════════════════════════════════════════════════════════════════════════════
# PRETTY PRINT HELPER  ── UNCHANGED ──
# ════════════════════════════════════════════════════════════════════════════

def print_analysis(a: AnswerAnalysis):
    verdict_color = {
        "EXCELLENT" : "✅",
        "GOOD"      : "🟢",
        "AVERAGE"   : "🟡",
        "POOR"      : "🔴",
        "SUSPICIOUS": "🚨",
    }
    icon = verdict_color.get(a.verdict, "❓")

    print(f"\n{'═'*55}")
    print(f"  ANSWER ANALYSIS  {icon} {a.verdict}")
    print(f"{'═'*55}")
    print(f"  Relevance  : {a.relevance_score:>4.1f} / 10")
    print(f"  Depth      : {a.depth_score:>4.1f} / 10")
    print(f"  Clarity    : {a.clarity_score:>4.1f} / 10")
    print(f"  Overall    : {a.overall_score:>4.1f} / 10")
    print(f"  AI Prob    : {a.ai_generated_prob*100:>4.0f} %")
    print(f"  Suspicious : {'YES ⚠' if a.suspicious else 'No'}")
    print(f"\n  Strengths:")
    for s in a.strengths:
        print(f"    + {s}")
    print(f"\n  Weaknesses:")
    for w in a.weaknesses:
        print(f"    - {w}")
    if a.suggestion:
        print(f"\n  Suggestion : {a.suggestion}")
    print(f"{'═'*55}\n")


# ════════════════════════════════════════════════════════════════════════════
# ▶  NEW — VOICE EXTENSION
# ════════════════════════════════════════════════════════════════════════════
#
#  Three new functions added below.  Nothing above this line was changed.
#
#  Pipeline:
#      record_audio()  →  speech_to_text()  →  run_voice_interview()
#
#  New library imports live here so the rest of the module stays
#  independent — existing functions work even if sounddevice / whisper
#  are not installed yet.
# ════════════════════════════════════════════════════════════════════════════

import threading                                  # built-in
import numpy as np                               # pip install numpy
import sounddevice as sd                         # pip install sounddevice
from scipy.io.wavfile import write as wav_write  # pip install scipy

# ── Voice constants ───────────────────────────────────────────────────────
_SAMPLE_RATE   = 16_000   # Hz  — Whisper requires 16 kHz input
_CHANNELS      = 1        # mono
_DURATION      = 60       # seconds — max recording length
_AUDIO_FILE    = "answer.wav"
_WHISPER_MODEL = "base"   # tiny | base | small | medium | large


def record_audio(
    filename : str = _AUDIO_FILE,
    duration : int = _DURATION,
) -> str:
    """
    Record audio from the default microphone and save it as a WAV file.
    Press ENTER at any time to stop recording early.
    Auto-stops after `duration` seconds if ENTER is not pressed.

    Includes a 1.5s cooldown before opening the stream so macOS has time
    to fully release the audio hardware between consecutive recordings.
    Retries automatically up to 3 times if PortAudio reports a hardware error.

    Args:
        filename : Output file path.  Defaults to "answer.wav".
        duration : Max recording length in seconds.  Defaults to 60.

    Returns:
        Path to the saved WAV file (same as filename).
    """
    # Give macOS time to release the mic from the previous session.
    # PaErrorCode -9986 happens when hardware is still marked as in-use.
    time.sleep(1.5)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        frames       = []
        stop_flag    = threading.Event()
        stream_error = []   # mutable list so inner thread can surface errors

        print(f"\n[Voice] Recording (max {duration}s) - speak now ...")
        print("[Voice]     Press ENTER to stop when done.\n")

        # Background thread: captures audio in 0.5s chunks
        def _record():
            try:
                with sd.InputStream(
                    samplerate = _SAMPLE_RATE,
                    channels   = _CHANNELS,
                    dtype      = "float32",
                ) as stream:
                    chunk_size = int(_SAMPLE_RATE * 0.5)
                    max_chunks = int(duration / 0.5)
                    for _ in range(max_chunks):
                        if stop_flag.is_set():
                            break
                        chunk, _ = stream.read(chunk_size)
                        frames.append(chunk.copy())
            except Exception as exc:
                stream_error.append(exc)   # surface to main thread

        record_thread = threading.Thread(target=_record, daemon=True)
        record_thread.start()

        # Wait briefly to confirm stream opened without error
        time.sleep(0.4)
        if stream_error:
            stop_flag.set()
            record_thread.join()
            if attempt < max_retries:
                print(f"[Voice] Audio hardware busy, retrying in 2s "
                      f"(attempt {attempt}/{max_retries}) ...")
                time.sleep(2.0)
                continue
            else:
                raise RuntimeError(
                    f"[Voice] Could not open microphone after {max_retries} attempts.\n"
                    f"Error: {stream_error[0]}\n"
                    "Tip: close other apps using the mic, then run again."
                )

        # Block until user presses ENTER (or thread finishes on timeout)
        input()
        stop_flag.set()
        record_thread.join()

        if stream_error:
            raise RuntimeError(f"[Voice] Recording error: {stream_error[0]}")

        if not frames:
            raise RuntimeError("[Voice] No audio captured. Check your microphone.")

        # Concatenate chunks -> int16 PCM -> save WAV
        audio_np  = np.concatenate(frames, axis=0)
        audio_int = (audio_np * 32_767).astype(np.int16)

        wav_write(filename, _SAMPLE_RATE, audio_int)
        print(f"\n[Voice] Saved -> {filename}")
        return filename   # success - exit retry loop

    raise RuntimeError("[Voice] Recording failed unexpectedly.")


# ── Whisper model: loaded ONCE at module level, reused for every call ────
try:
    import whisper as _whisper_lib
    print(f"[Whisper] Loading '{_WHISPER_MODEL}' model (once) ...")
    _WHISPER_MODEL_INSTANCE = _whisper_lib.load_model(_WHISPER_MODEL)
    print("[Whisper] Model ready.")
except ImportError:
    _whisper_lib            = None
    _WHISPER_MODEL_INSTANCE = None


def speech_to_text(audio_file: str = _AUDIO_FILE) -> str:
    """
    Transcribe a WAV file to text using the globally loaded Whisper model.
    Model is loaded once at startup, not reloaded on every call.

    Args:
        audio_file : Path to the WAV file to transcribe.

    Returns:
        Transcribed text string.
    """
    if _WHISPER_MODEL_INSTANCE is None:
        raise ImportError(
            "Whisper is not installed.\n"
            "Run:  /usr/bin/python3 -m pip install openai-whisper"
        )

    print(f"[Whisper] Transcribing {audio_file} ...")
    result     = _WHISPER_MODEL_INSTANCE.transcribe(audio_file, fp16=False)
    transcript = result["text"].strip()

    print(f"\n[Whisper] Transcribed answer:")
    print(f"    \"{transcript}\"\n")

    return transcript


def run_voice_interview(
    question : str,
    role     : str = "Software Engineer",
) -> AnswerAnalysis:
    """
    Full end-to-end voice interview pipeline.

    Steps:
        1. Display the question to the candidate.
        2. Record spoken answer        → record_audio()
        3. Transcribe to text          → speech_to_text()
        4. Analyse quality/authenticity → analyse_answer()
        5. Print full analysis          → print_analysis()
        6. Generate follow-up questions → generate_followup_questions()
        7. Return AnswerAnalysis object  (use with get_cheating_signal())

    Args:
        question : Interview question string.
        role     : Target job role for Llama evaluation context.

    Returns:
        AnswerAnalysis dataclass — fully compatible with get_cheating_signal().
    """
    # ── 1. Show question ─────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  QUESTION:")
    print(f"  {question}")
    print(f"{'─'*55}")
    input("  Press ENTER when ready to speak …\n")

    # ── 2. Record audio ──────────────────────────────────────────────────
    audio_path = record_audio()          # saves "answer.wav"

    # ── 3. Transcribe ────────────────────────────────────────────────────
    transcript = speech_to_text(audio_path)

    if not transcript:
        print("[Pipeline] ⚠  No speech detected — marking answer as POOR.")
        empty = AnswerAnalysis(
            question   = question,
            answer     = "",
            verdict    = "POOR",
            weaknesses = ["No speech detected in the recording."],
        )
        print_analysis(empty)
        return empty

    # ── 4. Analyse with Llama ─────────────────────────────────────────────
    analysis = analyse_answer(question, transcript, role=role)

    # ── 5. Print results ──────────────────────────────────────────────────
    print_analysis(analysis)

    # ── 6. Follow-up questions ────────────────────────────────────────────
    followups = generate_followup_questions(analysis)
    print("  Suggested Follow-up Questions:")
    for i, fq in enumerate(followups, 1):
        print(f"    {i}. {fq}")
    print()

    # ── 7. Return for cheating engine ─────────────────────────────────────
    return analysis


# ════════════════════════════════════════════════════════════════════════════
# ▶  NEW — INTERVIEW SESSION + PDF REPORT
# ════════════════════════════════════════════════════════════════════════════
#
#  run_interview_session()  — loops through all questions, stores results,
#                             prints full analysis at the end, generates PDF.
#  generate_pdf_report()    — builds a formatted PDF from stored results.
#
#  Requires:  pip install fpdf2
# ════════════════════════════════════════════════════════════════════════════

import datetime


def _pdf_safe(text: str) -> str:
    """
    Sanitize a string for FPDF core fonts (Latin-1 only).
    Replaces common Unicode punctuation with ASCII equivalents,
    then encodes with errors="replace" as a final safety net.
    """
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2014": "-",    # em dash
        "\u2013": "-",    # en dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...",  # ellipsis
        "\u2022": "*",    # bullet
        "\u2013": "-",    # en dash (duplicate guard)
        "\u2012": "-",    # figure dash
        "\u00b7": "*",    # middle dot
        "\u2713": "OK",   # check mark
        "\u2714": "OK",   # heavy check mark
        "\u2718": "X",    # cross mark
        "\u00e9": "e",    "\u00e8": "e",  "\u00ea": "e",
        "\u00e0": "a",    "\u00e2": "a",  "\u00e4": "a",
        "\u00f4": "o",    "\u00f6": "o",  "\u00f9": "u",
        "\u00fb": "u",    "\u00fc": "u",  "\u00ee": "i",
        "\u00ef": "i",    "\u00e7": "c",  "\u00f1": "n",
    }
    for char, safe in replacements.items():
        text = text.replace(char, safe)
    # Final net: anything still outside Latin-1 becomes "?"
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(
    results     : list[dict],
    output_path : str = "interview_report.pdf",
    candidate   : str = "Candidate",
    role        : str = "Software Engineer",
) -> str:
    """
    Generate a PDF interview report from a list of per-question result dicts.

    Each dict in `results` must contain:
        question, answer, relevance_score, depth_score, clarity_score,
        overall_score, ai_generated_prob, strengths, weaknesses,
        suggestion, verdict

    Args:
        results     : List of result dicts (one per question).
        output_path : Where to save the PDF.
        candidate   : Candidate name shown on the cover.
        role        : Job role shown on the cover.

    Returns:
        Path to the generated PDF file.
    """
    try:
        from fpdf import FPDF          # pip install fpdf2
    except ImportError:
        raise ImportError(
            "fpdf2 is not installed.\n"
            "Run:  /usr/bin/python3 -m pip install fpdf2"
        )

    # ── Colour palette ────────────────────────────────────────────────────
    BRAND   = (52,  73, 130)   # dark blue  - headings
    ACCENT  = (231, 76,  60)   # red        - verdict / warnings
    LIGHT   = (245, 245, 245)  # light grey - row backgrounds
    WHITE   = (255, 255, 255)
    BLACK   = (30,  30,  30)

    # ASCII-only verdict labels (no unicode tick/cross)
    verdict_label = {
        "EXCELLENT" : "[OK] EXCELLENT",
        "GOOD"      : "[OK] GOOD",
        "AVERAGE"   : "[~]  AVERAGE",
        "POOR"      : "[X]  POOR",
        "SUSPICIOUS": "[!]  SUSPICIOUS",
    }

    # ── FPDF subclass with header / footer ────────────────────────────────
    class InterviewPDF(FPDF):
        def header(self):
            self.set_fill_color(*BRAND)
            self.rect(0, 0, 210, 14, "F")
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(*WHITE)
            self.set_xy(0, 3)
            self.cell(0, 8, "AI Mock Interview + Cheating Detection System",
                      align="C")
            self.set_text_color(*BLACK)
            self.ln(10)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 8,
                      f"Page {self.page_no()} - Confidential Interview Report",
                      align="C")
            self.set_text_color(*BLACK)

    # Helper: cell that moves to the next line (replaces ln=True)
    NX, NY = "LMARGIN", "NEXT"

    pdf = InterviewPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 18, 15)

    # ════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.ln(20)

    scores = [r["overall_score"] for r in results]
    avg    = sum(scores) / len(scores) if scores else 0.0

    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*BRAND)
    pdf.cell(0, 12, "Interview Report", align="C", new_x=NX, new_y=NY)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8,
             _pdf_safe(f"Candidate: {candidate}   |   Role: {role}"),
             align="C", new_x=NX, new_y=NY)
    pdf.cell(0, 8,
             f"Date: {datetime.date.today().strftime('%B %d, %Y')}",
             align="C", new_x=NX, new_y=NY)
    pdf.ln(6)

    # Overall score banner
    pdf.set_fill_color(*BRAND)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12,
             f"  Overall Interview Score:  {avg:.1f} / 10",
             fill=True, new_x=NX, new_y=NY)
    pdf.set_text_color(*BLACK)
    pdf.ln(8)

    # Quick score table header
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(*BRAND)
    pdf.set_text_color(*WHITE)
    for col, w in [("Q#", 12), ("Question (brief)", 98),
                   ("Score", 20), ("Verdict", 40)]:
        pdf.cell(w, 8, col, border=0, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(*BLACK)

    # Quick score table rows
    for i, r in enumerate(results, 1):
        pdf.set_fill_color(*(LIGHT if i % 2 == 0 else WHITE))
        q      = _pdf_safe(r["question"])
        brief  = q[:60] + ("..." if len(q) > 60 else "")
        v      = r.get("verdict", "")
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(12, 7, str(i), fill=True, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(98, 7, brief, fill=True)
        pdf.cell(20, 7, f"{r['overall_score']:.1f}", fill=True, align="C")
        pdf.set_text_color(*(ACCENT if v in ("SUSPICIOUS", "POOR") else BRAND))
        pdf.cell(40, 7, v, fill=True, align="C")
        pdf.set_text_color(*BLACK)
        pdf.ln()

    # ════════════════════════════════════════════════════════════════════
    # PER-QUESTION PAGES
    # ════════════════════════════════════════════════════════════════════
    for i, r in enumerate(results, 1):
        pdf.add_page()

        # Question header bar
        pdf.set_fill_color(*BRAND)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 9, f"  Question {i} of {len(results)}",
                 fill=True, new_x=NX, new_y=NY)
        pdf.set_text_color(*BLACK)
        pdf.ln(2)

        # Question text
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*BRAND)
        pdf.multi_cell(180, 6, _pdf_safe(r["question"]))
        pdf.set_text_color(*BLACK)
        pdf.ln(3)

        # Candidate answer
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, "Candidate Answer:", new_x=NX, new_y=NY)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_fill_color(*LIGHT)
        answer_text = _pdf_safe(r["answer"]) if r["answer"] else "(no answer recorded)"
        pdf.multi_cell(180, 5, answer_text, fill=True)
        pdf.ln(3)

        # Score grid - header row
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(*BRAND)
        pdf.set_text_color(*WHITE)
        for label, w in [("Relevance", 45), ("Depth", 45),
                         ("Clarity",   45), ("Overall", 45)]:
            pdf.cell(w, 7, label, fill=True, align="C")
        pdf.ln()

        # Score grid - value row
        pdf.set_text_color(*BLACK)
        pdf.set_fill_color(*LIGHT)
        pdf.set_font("Helvetica", "B", 10)
        for val in [r["relevance_score"], r["depth_score"],
                    r["clarity_score"],   r["overall_score"]]:
            pdf.cell(45, 8, f"{val:.1f} / 10", fill=True, align="C")
        pdf.ln(3)

        # AI probability + verdict
        ai_pct = r["ai_generated_prob"] * 100
        v      = r.get("verdict", "")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(90, 6, f"AI-Generated Probability:  {ai_pct:.0f}%")
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*(ACCENT if v in ("SUSPICIOUS", "POOR") else BRAND))
        pdf.cell(0, 6, f"Verdict:  {verdict_label.get(v, v)}",
                 new_x=NX, new_y=NY)
        pdf.set_text_color(*BLACK)
        pdf.ln(3)

        # Strengths
        if r.get("strengths"):
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, "Strengths:", new_x=NX, new_y=NY)
            pdf.set_font("Helvetica", "", 9)
            for s in r["strengths"]:
                pdf.multi_cell(180, 5, "   " + _pdf_safe(f"+ {s}"))

        # Weaknesses
        if r.get("weaknesses"):
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, "Weaknesses:", new_x=NX, new_y=NY)
            pdf.set_font("Helvetica", "", 9)
            for w in r["weaknesses"]:
                pdf.multi_cell(180, 5, "   " + _pdf_safe(f"- {w}"))

        # Suggestion
        if r.get("suggestion"):
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, "Suggestion:", new_x=NX, new_y=NY)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_fill_color(*LIGHT)
            pdf.multi_cell(180, 5, _pdf_safe(r["suggestion"]), fill=True)

    # ════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY PAGE
    # ════════════════════════════════════════════════════════════════════
    pdf.add_page()

    pdf.set_fill_color(*BRAND)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "  Final Summary", fill=True, new_x=NX, new_y=NY)
    pdf.set_text_color(*BLACK)
    pdf.ln(4)

    # Overall score
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*BRAND)
    pdf.cell(0, 7, f"Overall Interview Score:  {avg:.1f} / 10",
             new_x=NX, new_y=NY)
    pdf.set_text_color(*BLACK)
    pdf.ln(3)

    # Aggregate lists
    all_strengths   = [s for r in results for s in r.get("strengths",  [])]
    all_weaknesses  = [w for r in results for w in r.get("weaknesses", [])]
    all_suggestions = [r["suggestion"] for r in results if r.get("suggestion")]

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Candidate Strengths:", new_x=NX, new_y=NY)
    pdf.set_font("Helvetica", "", 9)
    for s in all_strengths[:6]:
        pdf.multi_cell(180, 5, "   " + _pdf_safe(f"+ {s}"))

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Candidate Weaknesses:", new_x=NX, new_y=NY)
    pdf.set_font("Helvetica", "", 9)
    for w in all_weaknesses[:6]:
        pdf.multi_cell(180, 5, "   " + _pdf_safe(f"- {w}"))

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Improvement Suggestions:", new_x=NX, new_y=NY)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_fill_color(*LIGHT)
    for suggestion in all_suggestions[:3]:
        pdf.multi_cell(180, 5, "   " + _pdf_safe(f"* {suggestion}"), fill=True)

    # Cheating signals banner
    pdf.ln(4)
    suspicious_count = sum(1 for r in results if r.get("verdict") == "SUSPICIOUS")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(*BRAND)
    pdf.set_text_color(*WHITE)
    pdf.cell(
        0, 8,
        f"  Cheating Signals Detected: {suspicious_count} / {len(results)} answers flagged",
        fill=True, new_x=NX, new_y=NY,
    )
    pdf.set_text_color(*BLACK)

    pdf.output(output_path)
    print(f"\n[Report] PDF saved -> {output_path}")
    return output_path


def run_interview_session(
    questions   : list[str],
    role        : str = "Software Engineer",
    candidate   : str = "Candidate",
    pdf_path    : str = "interview_report.pdf",
) -> list[dict]:
    """
    Run a complete multi-question interview session.

    Behaviour:
        • Ask each question one at a time.
        • Record and transcribe each answer silently.
        • Store all analyses internally — no scoring output during interview.
        • After ALL questions are answered, print the full analysis for each.
        • Generate a PDF report with per-question detail + final summary.
        • Return all cheating signals for Member 4's engine.

    Args:
        questions : List of interview question strings.
        role      : Candidate's target role (used for Llama context).
        candidate : Candidate name (used in PDF cover).
        pdf_path  : Output path for the generated PDF report.

    Returns:
        List of cheating signal dicts (one per question), ready for
        the central Cheating Engine (get_cheating_signal interface).
    """
    if not questions:
        raise ValueError("Question list cannot be empty.")

    total     = len(questions)
    results   : list[AnswerAnalysis] = []
    print(f"\n{'═'*55}")
    print(f"  AI MOCK INTERVIEW SESSION")
    print(f"  Role: {role}  |  Questions: {total}")
    print(f"{'═'*55}")
    print("  The full analysis will be shown AFTER the interview.\n")

    # ════════════════════════════════════════════════════════════════════
    # INTERVIEW LOOP — record + transcribe + analyse (silent)
    # ════════════════════════════════════════════════════════════════════
    for idx, question in enumerate(questions, 1):

        print(f"\n{'─'*55}")
        print(f"  QUESTION {idx} of {total}")
        print(f"{'─'*55}")
        print(f"  {question}")
        print(f"{'─'*55}")
        input(f"\n  Press ENTER when ready to answer question {idx} …\n")

        # ── Record ───────────────────────────────────────────────────
        audio_file = f"answer_q{idx}.wav"
        audio_path = record_audio(filename=audio_file)

        # ── Transcribe ───────────────────────────────────────────────
        transcript = speech_to_text(audio_path)

        print(f"\n  ✅  Answer {idx} recorded. Moving to next question …\n")

        # ── Analyse (silently — no print yet) ────────────────────────
        if transcript.strip():
            analysis = analyse_answer(question, transcript, role=role)
        else:
            analysis = AnswerAnalysis(
                question   = question,
                answer     = "",
                verdict    = "POOR",
                weaknesses = ["No speech detected."],
            )

        results.append(analysis)

    # ════════════════════════════════════════════════════════════════════
    # POST-INTERVIEW — print all analyses now
    # ════════════════════════════════════════════════════════════════════
    print(f"\n\n{'═'*55}")
    print(f"  INTERVIEW COMPLETE — FULL ANALYSIS")
    print(f"{'═'*55}\n")

    for idx, analysis in enumerate(results, 1):
        print(f"\n  ── Question {idx}: {analysis.question[:60]}{'…' if len(analysis.question)>60 else ''}")
        print(f"  ── Transcribed Answer: \"{analysis.answer[:120]}{'…' if len(analysis.answer)>120 else ''}\"")
        print_analysis(analysis)

        followups = generate_followup_questions(analysis)
        print("  Suggested Follow-up Questions:")
        for i, fq in enumerate(followups, 1):
            print(f"    {i}. {fq}")
        print()

    # ════════════════════════════════════════════════════════════════════
    # OVERALL SCORE SUMMARY
    # ════════════════════════════════════════════════════════════════════
    scores = [a.overall_score for a in results]
    avg    = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'═'*55}")
    print(f"  FINAL OVERALL SCORE:  {avg:.1f} / 10")
    print(f"  Questions answered  :  {len(results)}")
    print(f"  Suspicious answers  :  {sum(1 for a in results if a.suspicious)}")
    print(f"{'═'*55}\n")

    # ════════════════════════════════════════════════════════════════════
    # GENERATE PDF REPORT
    # ════════════════════════════════════════════════════════════════════
    report_data = [a.to_dict() for a in results]
    generate_pdf_report(
        results     = report_data,
        output_path = pdf_path,
        candidate   = candidate,
        role        = role,
    )

    # ════════════════════════════════════════════════════════════════════
    # RETURN CHEATING SIGNALS for Member 4's engine
    # ════════════════════════════════════════════════════════════════════
    cheating_signals = [get_cheating_signal(a) for a in results]
    print(f"[Cheating Engine] Signals → {cheating_signals}")
    return cheating_signals


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not check_ollama_health():
        print("❌  Ollama not running. Start it with:  ollama serve")
        exit(1)

    questions = [
        "Explain the difference between a process and a thread.",
        "What is a deadlock in operating systems?",
        "Explain how memory management works in an OS.",
    ]

    cheating_signals = run_interview_session(
        questions = questions,
        role      = "Backend Engineer",
        candidate = "Candidate",
        pdf_path  = "interview_report.pdf",
    )