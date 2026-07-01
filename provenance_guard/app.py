"""Flask application factory for Provenance Guard.

Endpoints:
  POST /submit    — attribute a piece of text
  POST /appeal    — file an appeal against a prior classification
  GET  /log       — read recent audit-log entries
  GET  /content/<id> — look up current status for one submission
  GET  /health    — trivial liveness check

Rate limits are applied via Flask-Limiter with in-memory storage. Chosen
values and their reasoning are documented in the README.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from provenance_guard.scoring import score_and_label
from provenance_guard.signals import score_with_llm, score_with_stylometrics
from provenance_guard.storage import AuditStore, new_content_id

MIN_TEXT_LENGTH = 20
MIN_APPEAL_REASONING = 10

SubmitLimits = "10 per minute;100 per day"
AppealLimits = "5 per minute;20 per day"


def create_app(
    *,
    data_dir: str | None = None,
    llm_fn: Callable[[str], dict[str, Any]] | None = None,
    style_fn: Callable[[str], dict[str, Any]] | None = None,
    enable_rate_limits: bool = True,
) -> Flask:
    """Build a Flask app.

    Test hook: pass `llm_fn` / `style_fn` to swap in deterministic stubs
    that don't call the real Groq API. Set `enable_rate_limits=False` to
    disable limiter decorators for tests that need to fire many requests.
    """
    app = Flask(__name__)
    store = AuditStore(data_dir=data_dir or os.environ.get("PG_DATA_DIR", "data"))
    app.config["STORE"] = store

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[],
        storage_uri="memory://",
        enabled=enable_rate_limits,
    )
    app.extensions["limiter"] = limiter  # prevent GC under test

    score_llm = llm_fn or score_with_llm
    score_style = style_fn or score_with_stylometrics

    @app.errorhandler(429)
    def _rate_limited(err):  # noqa: ARG001
        return (
            jsonify(
                {
                    "error": "rate_limit_exceeded",
                    "message": (
                        "Too many requests. See README for configured limits."
                    ),
                }
            ),
            429,
        )

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "provenance-guard"})

    @app.post("/submit")
    @limiter.limit(SubmitLimits)
    def submit():
        payload = request.get_json(silent=True) or {}
        text = payload.get("text")
        creator_id = payload.get("creator_id")

        if not isinstance(text, str) or len(text.strip()) < MIN_TEXT_LENGTH:
            return (
                jsonify(
                    {
                        "error": "invalid_text",
                        "message": (
                            f"'text' must be a string with at least "
                            f"{MIN_TEXT_LENGTH} characters."
                        ),
                    }
                ),
                400,
            )
        if not isinstance(creator_id, str) or not creator_id.strip():
            return (
                jsonify(
                    {
                        "error": "invalid_creator_id",
                        "message": "'creator_id' must be a non-empty string.",
                    }
                ),
                400,
            )

        text = text.strip()
        creator_id = creator_id.strip()

        llm_result = score_llm(text)
        style_result = score_style(text)
        result = score_and_label(llm_result, style_result)

        content_id = new_content_id()
        store.record_submission(
            content_id=content_id,
            creator_id=creator_id,
            text_length=len(text),
            result=result,
        )

        return jsonify(
            {
                "content_id": content_id,
                "creator_id": creator_id,
                "attribution": result["attribution"],
                "ai_probability": result["ai_probability"],
                "confidence": result["confidence"],
                "label": result["label"],
                "signals": result["signals"],
                "status": "classified",
            }
        )

    @app.post("/appeal")
    @limiter.limit(AppealLimits)
    def appeal():
        payload = request.get_json(silent=True) or {}
        content_id = payload.get("content_id")
        reasoning = payload.get("creator_reasoning")

        if not isinstance(content_id, str) or not content_id.strip():
            return (
                jsonify(
                    {
                        "error": "invalid_content_id",
                        "message": "'content_id' is required.",
                    }
                ),
                400,
            )
        if not isinstance(reasoning, str) or len(reasoning.strip()) < MIN_APPEAL_REASONING:
            return (
                jsonify(
                    {
                        "error": "invalid_reasoning",
                        "message": (
                            f"'creator_reasoning' must be a string of at "
                            f"least {MIN_APPEAL_REASONING} characters."
                        ),
                    }
                ),
                400,
            )

        entry = store.record_appeal(content_id.strip(), reasoning.strip())
        if entry is None:
            return (
                jsonify(
                    {
                        "error": "not_found",
                        "message": f"No submission found for content_id: {content_id}",
                    }
                ),
                404,
            )

        return jsonify(
            {
                "content_id": content_id,
                "status": "under_review",
                "message": (
                    "Appeal received. A human reviewer will re-examine "
                    "this classification."
                ),
                "appeal_logged_at": entry["timestamp"],
                "original_attribution": entry["original_attribution"],
            }
        )

    @app.get("/log")
    def get_log():
        limit_param = request.args.get("limit", "50")
        try:
            limit = int(limit_param)
        except ValueError:
            return (
                jsonify(
                    {
                        "error": "invalid_limit",
                        "message": "'limit' must be an integer.",
                    }
                ),
                400,
            )
        entries = store.get_entries(limit=limit)
        return jsonify({"count": len(entries), "entries": entries})

    @app.get("/content/<content_id>")
    def get_content(content_id: str):
        record = store.get_content(content_id)
        if record is None:
            return (
                jsonify(
                    {
                        "error": "not_found",
                        "message": f"No submission found for content_id: {content_id}",
                    }
                ),
                404,
            )
        return jsonify({"content_id": content_id, **record})

    return app


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
