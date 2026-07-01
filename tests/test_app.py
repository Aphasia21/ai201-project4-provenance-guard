"""Integration tests for the Flask API endpoints."""

import json
import pytest

from provenance_guard.app import create_app

LONG_TEXT = (
    "The relationship between monetary policy and asset price inflation has been "
    "extensively studied in the literature. Central banks face a fundamental tension "
    "between their mandate for price stability and the unintended consequences of "
    "prolonged low interest rates on equity and real estate valuations."
)


def _llm_stub(score):
    def fn(text):
        return {"score": score, "ok": True, "model": "stub", "reasoning": "stub"}
    return fn


def _style_stub(score):
    def fn(text):
        return {"score": score, "ok": True, "metrics": {}, "components": {}}
    return fn


@pytest.fixture
def client(tmp_path):
    app = create_app(
        data_dir=str(tmp_path),
        llm_fn=_llm_stub(0.9),
        style_fn=_style_stub(0.8),
        enable_rate_limits=False,
    )
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def human_client(tmp_path):
    """Client wired for human-leaning classifications."""
    app = create_app(
        data_dir=str(tmp_path),
        llm_fn=_llm_stub(0.1),
        style_fn=_style_stub(0.1),
        enable_rate_limits=False,
    )
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True


# ---------------------------------------------------------------------------
# POST /submit — happy path
# ---------------------------------------------------------------------------

def test_submit_returns_required_fields(client):
    resp = client.post(
        "/submit",
        json={"text": LONG_TEXT, "creator_id": "user-1"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    for field in ("content_id", "attribution", "ai_probability", "confidence", "label", "signals", "status"):
        assert field in data, f"missing field: {field}"


def test_submit_high_ai_signals(client):
    resp = client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-1"})
    data = resp.get_json()
    assert data["attribution"] == "likely_ai"
    assert data["ai_probability"] >= 0.75


def test_submit_human_signals(human_client):
    resp = human_client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-1"})
    data = resp.get_json()
    assert data["attribution"] == "likely_human"
    assert data["ai_probability"] <= 0.30


def test_submit_label_variant_matches_attribution(client):
    resp = client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-1"})
    data = resp.get_json()
    assert data["label"]["variant"] == "high_confidence_ai"


# ---------------------------------------------------------------------------
# POST /submit — validation errors
# ---------------------------------------------------------------------------

def test_submit_missing_text(client):
    resp = client.post("/submit", json={"creator_id": "user-1"})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_text"


def test_submit_short_text(client):
    resp = client.post("/submit", json={"text": "hi", "creator_id": "user-1"})
    assert resp.status_code == 400


def test_submit_missing_creator(client):
    resp = client.post("/submit", json={"text": LONG_TEXT})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_creator_id"


def test_submit_empty_creator(client):
    resp = client.post("/submit", json={"text": LONG_TEXT, "creator_id": "  "})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /appeal — happy path
# ---------------------------------------------------------------------------

def test_appeal_full_flow(client):
    # Submit first
    sub = client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-1"}).get_json()
    content_id = sub["content_id"]

    # Appeal
    resp = client.post(
        "/appeal",
        json={"content_id": content_id, "creator_reasoning": "I wrote this myself."},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "under_review"
    assert data["content_id"] == content_id
    assert "appeal_logged_at" in data


def test_appeal_updates_content_status(client):
    sub = client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-1"}).get_json()
    cid = sub["content_id"]
    client.post("/appeal", json={"content_id": cid, "creator_reasoning": "I wrote this."})

    content = client.get(f"/content/{cid}").get_json()
    assert content["status"] == "under_review"


# ---------------------------------------------------------------------------
# POST /appeal — validation errors
# ---------------------------------------------------------------------------

def test_appeal_unknown_content_id(client):
    resp = client.post(
        "/appeal",
        json={"content_id": "nonexistent-id", "creator_reasoning": "I wrote this."},
    )
    assert resp.status_code == 404


def test_appeal_missing_reasoning(client):
    sub = client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-1"}).get_json()
    resp = client.post("/appeal", json={"content_id": sub["content_id"]})
    assert resp.status_code == 400


def test_appeal_short_reasoning(client):
    sub = client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-1"}).get_json()
    resp = client.post(
        "/appeal",
        json={"content_id": sub["content_id"], "creator_reasoning": "no"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /log
# ---------------------------------------------------------------------------

def test_log_captures_submissions(client):
    client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-1"})
    client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-2"})
    data = client.get("/log").get_json()
    assert data["count"] >= 2
    events = [e["event"] for e in data["entries"]]
    assert "submission" in events


def test_log_captures_appeal(client):
    sub = client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-1"}).get_json()
    client.post("/appeal", json={"content_id": sub["content_id"], "creator_reasoning": "I wrote this."})
    data = client.get("/log").get_json()
    events = [e["event"] for e in data["entries"]]
    assert "appeal" in events


def test_log_entries_have_required_fields(client):
    client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-1"})
    entries = client.get("/log").get_json()["entries"]
    entry = next(e for e in entries if e["event"] == "submission")
    for field in ("content_id", "creator_id", "timestamp", "attribution", "ai_probability", "confidence", "signals"):
        assert field in entry, f"missing audit field: {field}"


def test_log_limit_param(client):
    for i in range(5):
        client.post("/submit", json={"text": LONG_TEXT, "creator_id": f"user-{i}"})
    data = client.get("/log?limit=2").get_json()
    assert data["count"] == 2


# ---------------------------------------------------------------------------
# GET /content/<id>
# ---------------------------------------------------------------------------

def test_get_content_found(client):
    sub = client.post("/submit", json={"text": LONG_TEXT, "creator_id": "user-1"}).get_json()
    resp = client.get(f"/content/{sub['content_id']}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "classified"


def test_get_content_not_found(client):
    resp = client.get("/content/nonexistent")
    assert resp.status_code == 404
