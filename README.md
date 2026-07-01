# Provenance Guard

A backend service that classifies text content as human-written or AI-generated, surfaces a transparency label to readers, and provides creators with an appeals workflow when they believe they've been misclassified.

---

## Architecture

```
POST /submit
    │
    ├─► Signal 1: LLM assessment (Groq / llama-3.1-8b-instant)
    │       Evaluates semantic + stylistic coherence holistically
    │       Output: { score: 0-1, reasoning: str, ok: bool }
    │
    ├─► Signal 2: Stylometric heuristics (pure Python)
    │       Measures sentence-length CV, type-token ratio, punctuation entropy
    │       Output: { score: 0-1, metrics: {...}, ok: bool }
    │
    ├─► Confidence scoring (scoring.py)
    │       Weighted combine → ai_probability
    │       Classify ai_probability → attribution + confidence + label
    │
    ├─► Audit log (storage.py, append-only JSONL)
    │       Writes one structured entry per event
    │
    └─► JSON response { content_id, attribution, ai_probability,
                        confidence, label, signals, status }

POST /appeal
    │
    ├─► Lookup content_id in status index
    ├─► Append appeal event to audit log
    └─► JSON response { content_id, status: "under_review", ... }

GET /log  → returns recent audit entries as JSON
GET /content/<id> → returns current status for one submission
GET /health → liveness check
```

**Submission flow:** text arrives at `POST /submit`, runs through both signals in parallel (conceptually), the scoring layer combines their outputs into a single `ai_probability`, maps it to one of three transparency-label variants, and writes the whole result to the append-only audit log before returning a response that includes the `content_id` and human-readable label.

**Appeal flow:** a creator sends `POST /appeal` with their `content_id` and reasoning; the system looks up the original decision, marks the content `under_review`, and appends an appeal event to the audit log so a human reviewer can see both the original classification and the creator's rebuttal side-by-side.

---

## Detection Signals

### Signal 1 — LLM Assessment (Groq)

**What it measures:** whether a passage reads as AI-generated based on semantic and stylistic coherence. The model looks for hedging language, generic transitions, list-y structure, over-polished prose, and lack of personal voice (AI signals), versus personal anecdote, irregular rhythm, idiosyncratic word choice, and lived-in detail (human signals).

**Why it differs:** LLMs produce text by predicting the most probable next token at every step. This leads to statistically smooth prose — coherent, well-structured, but often without the jagged individuality of lived human experience. A fine-tuned assessor LLM can read for these patterns holistically in a way that rule-based statistics cannot.

**Blind spots:** the LLM assessor can be misled by lightly-edited AI text (where a human has broken up the uniformity), by non-native-English writers whose formal register resembles AI prose, and by highly polished human writing (essays, legal briefs) that the model reads as "too good to be human."

**Output:** `{ score: float [0,1], reasoning: str, ok: bool, model: str }` — `score=1` means "almost certainly AI."

### Signal 2 — Stylometric Heuristics

**What it measures:** three structural statistics computed in pure Python with no external API:

| Metric | What it captures | AI direction |
|--------|-----------------|--------------|
| **Sentence-length CV** | Coefficient of variation of word counts per sentence | Low CV → uniform → AI-leaning |
| **Type-token ratio** | Lexical diversity over a 100-word window | High TTR → diverse vocabulary → mildly AI-leaning |
| **Punctuation entropy** | Shannon entropy of punctuation character distribution | Low entropy (mostly comma+period) → AI-leaning |

**Why it differs:** LLM output is generated with a softmax over a fixed vocabulary at each step. This produces statistically uniform sentence lengths and heavy reliance on the two highest-frequency punctuation marks (comma and period). Human writing has more variance in all three dimensions.

**Blind spots:** short passages (< 2 sentences) produce undefined variance; academic or legal human writing has naturally high regularity; and heavy AI text that a human has edited to add sentence variety can score as human.

**Output:** `{ score: float [0,1], metrics: {...}, components: {...}, ok: bool }` — contributions are weighted CV×0.50 + TTR×0.20 + punct×0.30, normalized to active metrics only.

---

## Confidence Scoring

### Combination formula

```
ai_probability = 0.65 × llm_score + 0.35 × stylometric_score
```

The LLM signal gets a higher weight because it captures semantic holism — patterns a structural scanner can't see. The stylometric signal provides an independent structural check that is immune to LLM API outages.

**Degradation:** when a signal is unavailable (`ok=False`), its contribution is pulled toward 0.5 rather than trusted. If both signals are down the pipeline returns a neutral 0.5 with both marked `ok=False`.

### Thresholds

| Range | Attribution | Rationale |
|-------|-------------|-----------|
| `ai_probability ≥ 0.75` | `likely_ai` | Strong structural and semantic AI signal; high-confidence verdict |
| `0.30 < ai_probability < 0.75` | `uncertain` | Wide band: system defaults to "we don't know" to avoid mislabelling human writers |
| `ai_probability ≤ 0.30` | `likely_human` | Both signals consistently read human-like structure and style |

The uncertain band is intentionally wide because **a false positive (labelling a human writer as AI) is worse than a false negative** on a writing platform. The cost asymmetry is baked into the thresholds.

### Confidence value

The `confidence` field in the response measures the strength of the specific verdict (not AI-probability):
- `likely_ai`: `confidence = ai_probability`
- `likely_human`: `confidence = 1 − ai_probability`
- `uncertain`: `confidence = 1 − 2|ai_probability − 0.5|` — highest at 0.5 (deeply uncertain), lower toward either threshold

### Example submissions

**High-confidence AI (ai_probability ≈ 0.83, attribution: likely_ai)**
```
Input: "Artificial intelligence represents a transformative paradigm shift in
modern society. It is important to note that while the benefits of AI are
numerous, it is equally essential to consider the ethical implications.
Furthermore, stakeholders across various sectors must collaborate to ensure
responsible deployment."

llm_score: 0.92   stylometric_score: 0.64   ai_probability: 0.822
confidence: 0.822   attribution: likely_ai
```

**Low-confidence, human-leaning (ai_probability ≈ 0.18, attribution: likely_human)**
```
Input: "ok so i finally tried that new ramen place downtown and honestly?
underwhelming. the broth was fine but they put WAY too much sodium in it and
i was thirsty for like three hours after. my friend got the spicy version and
said it was better. probably won't go back unless someone drags me there"

llm_score: 0.12   stylometric_score: 0.29   ai_probability: 0.180
confidence: 0.820   attribution: likely_human
```

---

## Transparency Labels

All three label variants are shown below exactly as they appear in the API response and would be displayed to a reader.

### High-confidence AI

> **AI-generated (high confidence)**
>
> Our automated review estimates an 82% likelihood that this content was generated by AI, based on a language-model assessment and stylometric analysis. The creator can appeal this classification if they believe it is incorrect.

### High-confidence Human

> **Human-written (high confidence)**
>
> Our automated review estimates an 82% likelihood that this content was written by a human, based on a language-model assessment and stylometric analysis. Automated attribution is imperfect — appeals are welcome.

### Uncertain

> **Attribution uncertain**
>
> Our automated review could not confidently determine whether this content was written by a human or generated by AI (AI-likelihood estimate: 53%). This label reflects genuine uncertainty, not a verdict. Please read on your own judgment.

---

## Rate Limiting

Rate limits are applied by Flask-Limiter using in-memory storage (per-process, resets on restart; a real deployment would use Redis).

| Endpoint | Limit | Reasoning |
|----------|-------|-----------|
| `POST /submit` | **10 per minute, 100 per day** | A genuine creator submits a few pieces of work per session, not dozens per minute. 10/min accommodates batch uploads; 100/day prevents a script from exhausting the system. |
| `POST /appeal` | **5 per minute, 20 per day** | Appeals are deliberate, one-at-a-time actions. 5/min prevents appeal-flooding; 20/day reflects realistic usage (a creator with multiple contested pieces). |

### Rate-limit evidence

Running 12 rapid submissions against the 10/minute limit:

```
200
200
200
200
200
200
200
200
200
200
429
429
```

The first 10 requests succeed; requests 11 and 12 receive `429 Too Many Requests` with:
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. See README for configured limits."
}
```

---

## Audit Log

Every submission and appeal writes a structured JSON line to `data/audit.jsonl`. The log is append-only; on restart, the process replays it to rebuild the in-memory status index.

Retrieve entries with `GET /log` (returns JSON; `?limit=N` to cap results).

### Sample entries

**Submission entry:**
```json
{
  "event": "submission",
  "content_id": "3f7a2b1e-9c4d-4a1b-8e5f-123456789abc",
  "creator_id": "test-user-1",
  "timestamp": "2026-06-30T14:32:10.123Z",
  "text_length": 247,
  "attribution": "likely_ai",
  "ai_probability": 0.822,
  "confidence": 0.822,
  "label_variant": "high_confidence_ai",
  "signals": {
    "llm": { "score": 0.92, "ok": true, "model": "llama-3.1-8b-instant", "reasoning": "The text uses generic corporate hedging and lacks personal voice." },
    "stylometric": { "score": 0.64, "ok": true, "metrics": { "sentence_length_cv": 0.31, "type_token_ratio": 0.74, "punctuation_entropy_bits": 1.1, "sentence_count": 4 }, "components": { "sentence_length_cv_ai": 0.986, "type_token_ratio_ai": 0.637, "punctuation_entropy_ai": 0.507 } }
  },
  "status": "classified"
}
```

**Appeal entry:**
```json
{
  "event": "appeal",
  "content_id": "3f7a2b1e-9c4d-4a1b-8e5f-123456789abc",
  "creator_id": "test-user-1",
  "timestamp": "2026-06-30T14:35:22.456Z",
  "original_attribution": "likely_ai",
  "original_ai_probability": 0.822,
  "original_confidence": 0.822,
  "appeal_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical.",
  "status": "under_review"
}
```

**Uncertain-result submission entry:**
```json
{
  "event": "submission",
  "content_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "creator_id": "writer-42",
  "timestamp": "2026-06-30T14:40:05.789Z",
  "text_length": 312,
  "attribution": "uncertain",
  "ai_probability": 0.521,
  "confidence": 0.042,
  "label_variant": "uncertain",
  "signals": {
    "llm": { "score": 0.55, "ok": true, "model": "llama-3.1-8b-instant", "reasoning": "The text mixes formal prose with casual phrasing making attribution uncertain." },
    "stylometric": { "score": 0.47, "ok": true, "metrics": { "sentence_length_cv": 0.58, "type_token_ratio": 0.71, "punctuation_entropy_bits": 1.4, "sentence_count": 7 }, "components": { "sentence_length_cv_ai": 0.600, "type_token_ratio_ai": 0.521, "punctuation_entropy_ai": 0.321 } }
  },
  "status": "classified"
}
```

---

## Appeals Workflow

**Who can submit:** any creator who has a `content_id` from a prior `/submit` response.

**What they provide:** `content_id` (required) and `creator_reasoning` (required, ≥ 10 characters) explaining why they believe the classification is incorrect.

**What the system does:**
1. Looks up the `content_id` in the status index; returns 404 if not found.
2. Appends an `appeal` event to the audit log including the original classification and the creator's reasoning.
3. Updates the in-memory status to `under_review`.
4. Returns a confirmation with the appeal timestamp and original attribution.

**What a human reviewer sees:** `GET /log` returns both the original `submission` event and the subsequent `appeal` event for the same `content_id`, so the reviewer can compare the automated decision and confidence score against the creator's stated reasoning.

**Automated re-classification is not implemented** — the intent is to route uncertain or contested cases to human judgment rather than create a loop where creators submit appeals until a model changes its mind.

---

## Known Limitations

1. **Non-native English writers:** formal, careful prose from a non-native English speaker produces low sentence-length variance and minimal casual punctuation — the same structural fingerprint as AI-generated text. The LLM signal partially compensates by looking for personal anecdote and lived detail, but the stylometric signal will over-index on the formal register. This is the most likely source of false positives.

2. **Short passages (< 100 words):** the type-token ratio signal is suppressed below 80 words (the measure is meaningless on short windows), and sentence-length CV is undefined for single-sentence inputs. The pipeline degrades gracefully — missing signals are excluded from the weighted average — but confidence naturally falls toward the uncertain band. A haiku or a one-line caption will almost always land as "uncertain."

3. **Heavily edited AI text:** if a human edits AI-generated content to add sentence variety, casual punctuation, and a first-person anecdote, both signals will read it as human-leaning. The system is not adversarially hardened.

---

## Spec Reflection

**Where the spec guided implementation well:** the explicit false-positive asymmetry note ("labelling a human writer as AI is worse than a false negative") directly shaped the threshold design. Setting `AI_THRESHOLD=0.75` and `HUMAN_THRESHOLD=0.30` rather than symmetric 0.6/0.4 values means the system defaults to "uncertain" across a wide middle band, which is exactly right for a writing platform where misclassification has reputational consequences for creators.

**Where the implementation diverged:** the spec proposed combining two signals into a single score. During implementation, the signal outputs diverge more often than expected — the LLM reads a passage as likely-human while stylometrics reads it as AI-leaning, or vice versa. Rather than trying to reconcile disagreement in the score, the response now includes the per-signal scores and an `ok` flag for each in the `signals` field. This lets a human reviewer (or a future UI) see where the disagreement is, which is more informative than a single blended number.

---

## Rate Limiting — Quick Start

```bash
# Install dependencies (create a venv first if you like)
pip install -r requirements.txt

# Set up environment
cp env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the server
python -m provenance_guard.app

# Test submission
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications.", "creator_id": "test-user-1"}' | python -m json.tool

# Test appeal (replace CONTENT_ID with value from submit response)
curl -s -X POST http://localhost:5000/appeal \
  -H "Content-Type: application/json" \
  -d '{"content_id": "CONTENT_ID", "creator_reasoning": "I wrote this myself from personal experience."}' | python -m json.tool

# View audit log
curl -s http://localhost:5000/log | python -m json.tool
```

---

## AI Usage

1. **Architecture scaffold:** I directed the AI to generate a Flask app skeleton with the `/submit`, `/appeal`, and `/log` routes based on the architecture diagram in `planning.md`. The generated structure was clean but used a global list for storage. I replaced it with the `AuditStore` class that uses an append-only JSONL file and rebuilds the in-memory index on startup — a meaningful change for restartability.

2. **Stylometric scoring functions:** I directed the AI to implement `_ai_from_cv`, `_ai_from_ttr`, and `_ai_from_punct_entropy` given the anchor-point pairs I specified (e.g., "CV=0.30 → 1.0, CV=1.00 → 0.0"). The generated linear interpolations were correct but the AI chose symmetric anchor points for all three. I overrode the punctuation-entropy anchors to reflect that humans use a wider punctuation range than the AI initially assumed, narrowing the high-entropy end to better separate casual human prose from academic prose.
