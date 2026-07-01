"""Tests for scoring.py — signal combination and label generation."""

import pytest
from provenance_guard.scoring import score_and_label, _combine, _classify, _label_for

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(score, ok=True):
    return {"score": score, "ok": ok, "model": "test", "reasoning": "test"}

def _style(score, ok=True):
    return {"score": score, "ok": ok, "metrics": {}, "components": {}}


# ---------------------------------------------------------------------------
# _combine
# ---------------------------------------------------------------------------

class TestCombine:
    def test_both_ok_weighted(self):
        result = _combine(_llm(1.0), _style(0.0))
        assert abs(result - 0.65) < 1e-9

    def test_both_ok_symmetric(self):
        result = _combine(_llm(0.0), _style(1.0))
        assert abs(result - 0.35) < 1e-9

    def test_only_llm_ok(self):
        result = _combine(_llm(1.0), _style(0.0, ok=False))
        # 0.7 * 1.0 + 0.3 * 0.5 = 0.85
        assert abs(result - 0.85) < 1e-9

    def test_only_style_ok(self):
        result = _combine(_llm(1.0, ok=False), _style(1.0))
        # 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        assert abs(result - 0.75) < 1e-9

    def test_both_down_neutral(self):
        result = _combine(_llm(1.0, ok=False), _style(1.0, ok=False))
        assert result == 0.5


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_high_ai(self):
        attr, conf = _classify(0.80)
        assert attr == "likely_ai"
        assert conf == pytest.approx(0.80)

    def test_high_human(self):
        attr, conf = _classify(0.20)
        assert attr == "likely_human"
        assert conf == pytest.approx(0.80)

    def test_uncertain_midpoint(self):
        attr, conf = _classify(0.50)
        assert attr == "uncertain"
        assert conf == pytest.approx(1.0)

    def test_uncertain_near_ai_threshold(self):
        attr, conf = _classify(0.74)
        assert attr == "uncertain"
        # 1 - 2*|0.74 - 0.5| = 1 - 0.48 = 0.52; near threshold so <1.0
        assert conf < 1.0

    def test_boundary_ai(self):
        attr, _ = _classify(0.75)
        assert attr == "likely_ai"

    def test_boundary_human(self):
        attr, _ = _classify(0.30)
        assert attr == "likely_human"


# ---------------------------------------------------------------------------
# _label_for
# ---------------------------------------------------------------------------

class TestLabelFor:
    def test_ai_label(self):
        label = _label_for("likely_ai", 0.82)
        assert label["variant"] == "high_confidence_ai"
        assert "82%" in label["body"]
        assert "appeal" in label["body"].lower()

    def test_human_label(self):
        label = _label_for("likely_human", 0.18)
        assert label["variant"] == "high_confidence_human"
        # human_pct = 100 - 18 = 82
        assert "82%" in label["body"]

    def test_uncertain_label(self):
        label = _label_for("uncertain", 0.53)
        assert label["variant"] == "uncertain"
        assert "53%" in label["body"]
        assert "uncertainty" in label["body"].lower()


# ---------------------------------------------------------------------------
# score_and_label integration
# ---------------------------------------------------------------------------

class TestScoreAndLabel:
    def test_returns_required_keys(self):
        result = score_and_label(_llm(0.9), _style(0.7))
        for key in ("attribution", "ai_probability", "confidence", "label", "signals", "thresholds"):
            assert key in result

    def test_high_ai_signals_produce_ai_attribution(self):
        result = score_and_label(_llm(0.95), _style(0.90))
        assert result["attribution"] == "likely_ai"
        assert result["ai_probability"] > 0.75

    def test_low_signals_produce_human_attribution(self):
        result = score_and_label(_llm(0.05), _style(0.10))
        assert result["attribution"] == "likely_human"
        assert result["ai_probability"] < 0.30

    def test_mid_signals_produce_uncertain(self):
        result = score_and_label(_llm(0.50), _style(0.50))
        assert result["attribution"] == "uncertain"

    def test_probabilities_rounded(self):
        result = score_and_label(_llm(0.123456), _style(0.654321))
        assert result["ai_probability"] == round(result["ai_probability"], 3)

    def test_signals_block_shape(self):
        result = score_and_label(_llm(0.8), _style(0.6))
        assert "llm" in result["signals"]
        assert "stylometric" in result["signals"]
        assert result["signals"]["llm"]["ok"] is True
