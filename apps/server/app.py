"""Minimal Flask application for Sequence web migration.

This app currently exposes a single health-check endpoint and provides a
placeholder structure for future API development.  Firebase authentication,
match management and AI endpoints will be added in subsequent iterations.
"""

from __future__ import annotations

from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/api/v1/health")
def health() -> tuple[dict[str, str], int]:
    """Simple health-check endpoint."""
    return jsonify(status="ok"), 200


if __name__ == "__main__":
    app.run(debug=True)
