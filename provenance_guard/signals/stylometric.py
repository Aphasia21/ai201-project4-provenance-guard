"""Signal 2: stylometric heuristics.

Pure-Python statistics that empirically differ between AI and human prose:

  1. Sentence-length coefficient of variation
     AI text tends toward uniform medium-length sentences (CV ~0.3-0.5).
     Human prose varies more (CV ~0.6-1.4). Low CV → AI-leaning.

  2. Type-token ratio (lexical diversity)
     AI rarely repeats words; humans repeat naturally. High TTR after
     length-normalisation is mildly AI-leaning. Weak signal alone — kept
     light in the weighting.

  3. Punctuation diversity (Shannon entropy over punctuation marks)
     Humans mix dashes, ellipses, semicolons, exclamation marks. AI prose
     leans heavily on the comma+period pair. Low entropy → AI-leaning.

Blind spots (documented in README): short passages have unstable statistics,
and any heavily-edited piece (human or AI) drifts toward the middle.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")
# Includes period and comma: AI text tends to use almost ONLY these two,
# producing low-entropy punctuation distributions. Humans mix more freely.
_PUNCT_CHARS = ".,;:—–…!?"


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p for p in (s.strip() for s in parts) if p]


def _sentence_length_cv(sentences: list[str]) -> float | None:
    """Coefficient of variation of word counts per sentence.

    Returns None when fewer than 2 sentences exist (variance undefined).
    """
    if len(sentences) < 2:
        return None
    lengths = [len(_WORD.findall(s)) for s in sentences]
    lengths = [n for n in lengths if n > 0]
    if len(lengths) < 2:
        return None
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return None
    var = sum((n - mean) ** 2 for n in lengths) / len(lengths)
    return math.sqrt(var) / mean


def _type_token_ratio(text: str) -> float | None:
    """Type-token ratio over a fixed 100-word window.

    Returns None below 80 words because raw TTR scales with length (every
    word in a short passage is new) and the signal becomes meaningless.
    Truncating to a fixed window stabilises the measure across lengths.
    """
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < 80:
        return None
    window = words[:100]
    return len(set(window)) / len(window)


def _punctuation_entropy(text: str) -> float | None:
    """Shannon entropy of punctuation usage in bits.

    Returns None when fewer than 5 punctuation marks appear — not enough to
    estimate a distribution.
    """
    counts = Counter(ch for ch in text if ch in _PUNCT_CHARS)
    total = sum(counts.values())
    if total < 3:
        return None
    entropy = 0.0
    for n in counts.values():
        p = n / total
        entropy -= p * math.log2(p)
    return entropy


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _ai_from_cv(cv: float) -> float:
    """Map sentence-length CV → AI-likeness in [0, 1].

    Anchor points: CV=0.30 → 1.00 (very uniform), CV=1.00 → 0.00 (very varied).
    Linear interpolation between, clamped at the ends.
    """
    return _clamp(1.0 - (cv - 0.30) / 0.70)


def _ai_from_ttr(ttr: float) -> float:
    """Map windowed TTR → AI-likeness in [0, 1].

    Anchor points: TTR=0.55 → 0.20 (repetition, human-casual), TTR=0.78 →
    0.75 (high diversity in a 100-word window, AI-leaning). Linear between.
    """
    return _clamp(0.20 + (ttr - 0.55) * (0.55 / 0.23))


def _ai_from_punct_entropy(entropy: float) -> float:
    """Map punctuation entropy → AI-likeness in [0, 1].

    Anchor points: entropy=0.8 bits → 0.80 (only periods+commas, the AI
    default), entropy=2.2 bits → 0.15 (well-mixed punctuation).
    """
    return _clamp(0.80 - (entropy - 0.80) * (0.65 / 1.40))


def score_with_stylometrics(text: str) -> dict[str, Any]:
    """Compute stylometric AI-likeness for a passage.

    Returns a dict with:
      - score: weighted combination in [0, 1]; 1 means AI-leaning structure
      - metrics: dict of the raw metric values used, for the audit log
      - components: per-metric AI-likeness sub-scores
      - ok: True if enough text to compute at least one component
    """
    sentences = _split_sentences(text)
    cv = _sentence_length_cv(sentences)
    ttr = _type_token_ratio(text)
    entropy = _punctuation_entropy(text)

    contributions: list[tuple[str, float, float]] = []  # (name, ai_score, weight)
    components: dict[str, float | None] = {
        "sentence_length_cv_ai": None,
        "type_token_ratio_ai": None,
        "punctuation_entropy_ai": None,
    }

    if cv is not None:
        v = _ai_from_cv(cv)
        components["sentence_length_cv_ai"] = round(v, 3)
        contributions.append(("sentence_length_cv_ai", v, 0.50))

    if ttr is not None:
        v = _ai_from_ttr(ttr)
        components["type_token_ratio_ai"] = round(v, 3)
        contributions.append(("type_token_ratio_ai", v, 0.20))

    if entropy is not None:
        v = _ai_from_punct_entropy(entropy)
        components["punctuation_entropy_ai"] = round(v, 3)
        contributions.append(("punctuation_entropy_ai", v, 0.30))

    if not contributions:
        return {
            "score": 0.5,
            "metrics": {
                "sentence_length_cv": cv,
                "type_token_ratio": ttr,
                "punctuation_entropy_bits": entropy,
                "sentence_count": len(sentences),
            },
            "components": components,
            "ok": False,
        }

    weight_total = sum(w for _, _, w in contributions)
    combined = sum(v * w for _, v, w in contributions) / weight_total

    return {
        "score": round(combined, 3),
        "metrics": {
            "sentence_length_cv": round(cv, 3) if cv is not None else None,
            "type_token_ratio": round(ttr, 3) if ttr is not None else None,
            "punctuation_entropy_bits": round(entropy, 3) if entropy is not None else None,
            "sentence_count": len(sentences),
        },
        "components": components,
        "ok": True,
    }
