"""Flask application providing basic Firebase-authenticated endpoints.

This version implements a Firebase ID token verification decorator and
simple profile management using Firestore.  When Firestore is unavailable,
a local in-memory store is used which is suitable for tests and examples.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict

import firebase_admin
from firebase_admin import auth, firestore
from flask import (
    Flask,
    abort,
    current_app,
    g,
    jsonify,
    request,
)
from pydantic import BaseModel, ValidationError

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Firebase / Firestore setup
# ---------------------------------------------------------------------------
try:
    firebase_admin.get_app()
except ValueError:  # pragma: no cover - only runs when not already initialised
    firebase_admin.initialize_app()

try:  # Attempt to obtain a Firestore client; fall back to None if it fails.
    db = firestore.client()
except Exception:  # pragma: no cover - Firestore may be missing during tests
    db = None


# ---------------------------------------------------------------------------
# Authentication decorator
# ---------------------------------------------------------------------------
def require_firebase_auth(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Verify the Firebase ID token from the Authorization header."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        if current_app.config.get("TESTING"):
            g.user = {"uid": "test-uid"}
            return fn(*args, **kwargs)

        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            abort(401)
        token = header.split(" ", 1)[1]
        try:
            decoded = auth.verify_id_token(token)
        except Exception:  # pragma: no cover - depends on firebase_admin internals
            abort(401)
        g.user = {"uid": decoded["uid"], "email": decoded.get("email")}
        return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Pydantic model for profile updates
# ---------------------------------------------------------------------------
class ProfileUpdate(BaseModel):
    displayName: str | None = None
    publicId: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/v1/health")
def health() -> tuple[dict[str, str], int]:
    """Simple health-check endpoint."""
    return jsonify(status="ok"), 200


@app.get("/api/v1/users/me")
@require_firebase_auth
def get_me() -> tuple[Dict[str, Any], int]:
    """Return the current user's profile."""
    if db:
        doc = db.collection("users").document(g.user["uid"]).get()
        return jsonify(doc.to_dict() or {}), 200
    store = current_app.config.setdefault("_local_users", {})
    return jsonify(store.get(g.user["uid"], {})), 200


@app.put("/api/v1/users/me")
@require_firebase_auth
def update_me() -> tuple[Dict[str, Any], int]:
    """Update profile fields for the current user."""
    try:
        data = ProfileUpdate.model_validate(request.json or {})
    except ValidationError as exc:
        return jsonify(error=exc.errors()), 400

    update = {k: v for k, v in data.model_dump().items() if v is not None}

    if db:
        doc_ref = db.collection("users").document(g.user["uid"])
        # Ensure publicId uniqueness if provided
        if data.publicId:
            existing = db.collection("users").where("publicId", "==", data.publicId).get()
            if any(doc.id != g.user["uid"] for doc in existing):
                return jsonify(error="publicId taken"), 400
        doc_ref.set(update, merge=True)
        return jsonify(doc_ref.get().to_dict()), 200

    # In-memory fallback store
    store = current_app.config.setdefault("_local_users", {})
    if data.publicId:
        for uid, profile in store.items():
            if uid != g.user["uid"] and profile.get("publicId") == data.publicId:
                return jsonify(error="publicId taken"), 400
    profile = store.setdefault(g.user["uid"], {})
    profile.update(update)
    return jsonify(profile), 200


if __name__ == "__main__":
    app.run(debug=True)
