"""Tests for the stylometric signal."""

from provenance_guard.signals.stylometric import (
    score_with_stylometrics,
    _sentence_length_cv,
    _type_token_ratio,
    _punctuation_entropy,
    _split_sentences,
)


AI_TEXT = (
    "Artificial intelligence represents a transformative paradigm shift in modern society. "
    "It is important to note that while the benefits of AI are numerous, it is equally "
    "essential to consider the ethical implications. Furthermore, stakeholders across "
    "various sectors must collaborate to ensure responsible deployment of these systems."
)

HUMAN_TEXT = (
    "ok so i finally tried that new ramen place downtown and honestly? "
    "underwhelming. the broth was fine but they put WAY too much sodium in it and "
    "i was thirsty for like three hours after. my friend got the spicy version and "
    "said it was better. probably won't go back unless someone drags me there"
)


class TestSplitSentences:
    def test_basic(self):
        parts = _split_sentences("Hello world. How are you? Fine!")
        assert len(parts) == 3

    def test_empty(self):
        assert _split_sentences("") == []

    def test_single(self):
        assert len(_split_sentences("Just one sentence.")) == 1


class TestSentenceLengthCV:
    def test_uniform_returns_low_cv(self):
        sentences = ["one two three", "four five six", "seven eight nine"]
        cv = _sentence_length_cv(sentences)
        assert cv is not None
        assert cv < 0.1

    def test_varied_returns_high_cv(self):
        sentences = ["Hi.", "This is a much longer sentence with many words in it.", "Bye."]
        cv = _sentence_length_cv(sentences)
        assert cv is not None
        assert cv > 0.5

    def test_single_sentence_returns_none(self):
        assert _sentence_length_cv(["only one sentence here"]) is None


class TestTypeTokenRatio:
    def test_short_text_returns_none(self):
        assert _type_token_ratio("short") is None

    def test_long_repetitive_lower_than_diverse(self):
        # 120 words, only 3 unique → very low TTR
        repetitive = ("the cat sat " * 40).strip()
        # 120 unique alphabetic words
        alphabet_words = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
            "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
            "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
            "victor", "whiskey", "xray", "yankee", "zulu", "apple", "banana",
            "cherry", "dragon", "ember", "falcon", "garden", "harbor", "island",
            "jungle", "kitten", "lemon", "mango", "nectar", "orange", "plum",
            "quartz", "raven", "silver", "tiger", "umbrella", "violet", "walnut",
            "xenon", "yellow", "zebra", "amber", "bronze", "copper", "dusk",
            "elbow", "flame", "gravel", "hollow", "ivory", "jewel", "koala",
            "lantern", "marble", "needle", "opaque", "pebble", "riddle", "stone",
            "timber", "umber", "velvet", "winter", "xylem", "yeoman", "zenith",
            "acorn", "birch", "cedar", "denim", "ether", "fern", "granite",
            "hazel", "imply", "jasper", "kestrel", "laurel", "maple", "nettle",
            "onyx", "pine", "quail", "reed", "sage", "thyme", "ursine",
            "valor", "wheat", "xeric", "yarrow", "zinc", "alder", "beech",
            "clover", "dock", "elder", "furze", "gorse", "heather", "iris",
            "juniper", "kelp", "larch", "moss", "nethermost", "oaken",
        ]
        diverse = " ".join(alphabet_words[:120])
        r = _type_token_ratio(repetitive)
        d = _type_token_ratio(diverse)
        assert r is not None, "repetitive text is long enough"
        assert d is not None, "diverse text is long enough"
        assert r < d, f"expected repetitive({r}) < diverse({d})"


class TestPunctuationEntropy:
    def test_only_periods_low_entropy(self):
        text = "Hello. Goodbye. Thanks. Fine. Sure."
        entropy = _punctuation_entropy(text)
        assert entropy is not None
        assert entropy < 0.5

    def test_mixed_punct_higher_entropy(self):
        text = "Wow! Really? Yes — definitely... maybe; or not."
        entropy = _punctuation_entropy(text)
        assert entropy is not None
        assert entropy > 1.5

    def test_no_punct_returns_none(self):
        assert _punctuation_entropy("no punctuation here at all") is None


class TestScoreWithStylemetrics:
    def test_ok_true_on_sufficient_text(self):
        result = score_with_stylometrics(AI_TEXT)
        assert result["ok"] is True

    def test_ok_false_on_very_short_text(self):
        result = score_with_stylometrics("Hi.")
        assert result["ok"] is False
        assert result["score"] == 0.5

    def test_ai_text_scores_higher_than_human(self):
        ai_score = score_with_stylometrics(AI_TEXT)["score"]
        human_score = score_with_stylometrics(HUMAN_TEXT)["score"]
        assert ai_score > human_score

    def test_result_keys(self):
        result = score_with_stylometrics(AI_TEXT)
        assert "score" in result
        assert "metrics" in result
        assert "components" in result
        assert "ok" in result

    def test_score_in_range(self):
        result = score_with_stylometrics(AI_TEXT)
        assert 0.0 <= result["score"] <= 1.0
