# functions-python/main.py
"""
Firebase Cloud Functions (Gen2, Python) — Callable endpoints (no CORS)
para Sequence Online.

Todas las funciones expuestas aquí son https callable; el cliente debe
invocarlas con httpsCallable(functions, 'nombre').
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import firebase_admin
from firebase_admin import firestore as admin_fs
from firebase_functions.https_fn import FunctionsErrorCode, HttpsError
from google.cloud.firestore import Client, Transaction
from firebase_functions import https_fn, options

from game_logic.errors import EngineError, ErrorCode

# -------------------------
# Región y timeout (<= 3600 para Gen2 HTTP/callable)
# -------------------------
options.set_global_options(region="us-central1", timeout_sec=3000)

# -------------------------
# Helper de depuración (PyCharm/DataSpell Remote Debug - pydevd)
# -------------------------
import os

def ide_attach(port: int = 5678, suspend: bool = True, host: str = "127.0.0.1") -> None:
    """
    Attach to PyCharm/DataSpell Debug Server using pydevd_pycharm.
    Uses the snake_case kwargs that your installed pydevd expects.
    Safe: prints diagnostics and won't raise if attach fails.
    """
    if os.getenv("PYCHARM_DEBUG") != "1":
        return

    try:
        import pydevd_pycharm
    except Exception as e:
        print("[pydevd] import failed:", e)
        return

    try:
        # call using the signature your installed module exposes
        pydevd_pycharm.settrace(
            host=host,
            stdout_to_server=True,
            stderr_to_server=True,
            port=port,
            suspend=suspend,
        )
        print(f"[pydevd] connected to IDE at {host}:{port}")
        return
    except TypeError as e:
        # Should not happen for your reported version, but keep a fallback
        print("[pydevd] settrace TypeError:", e)

    # Fallback attempts (positional / minimal) — extra safety for other builds
    try:
        pydevd_pycharm.settrace(host, True, True, port, suspend)
        print(f"[pydevd] connected (positional) to {host}:{port}")
        return
    except Exception as e:
        print("[pydevd] positional fallback failed:", e)

    try:
        pydevd_pycharm.settrace(host=host, port=port, suspend=suspend)
        print(f"[pydevd] connected (minimal) to {host}:{port}")
        return
    except Exception as e:
        print("[pydevd] minimal fallback failed:", e)

    print("[pydevd] all attach attempts failed")


# -------------------------
# Admin SDK
# -------------------------
if not firebase_admin._apps:
    # Asegúrate de tener configuradas las variables de emulador si depuras local:
    #   FIRESTORE_EMULATOR_HOST=127.0.0.1:8080
    #   FIREBASE_AUTH_EMULATOR_HOST=127.0.0.1:9099
    firebase_admin.initialize_app()

db: Client = admin_fs.client()

# --- Motor del juego (tuyo) ---
from game_logic.engine import (
    create_full_deck,
    shuffle_deck,
    apply_move_to_state,
)

# -------------------------
# Utilidades
# -------------------------

def _uid_or_none(req: https_fn.CallableRequest) -> Optional[str]:
    if req.auth and getattr(req.auth, "uid", None):
        return req.auth.uid
    return None

def _require_uid(req: https_fn.CallableRequest) -> str:
    uid = _uid_or_none(req)
    if not uid:
        # Unify unauthenticated across the app
        raise HttpsError(
            FunctionsErrorCode.UNAUTHENTICATED,
            ErrorCode.ERR_UNAUTHENTICATED.value,
            details={"error": ErrorCode.ERR_UNAUTHENTICATED.value},
        )
    return uid

def _read_board_cells() -> List[List[str]]:
    # Lee el JSON del tablero si existe; si no, fallback 10x10 vacío.
    board_file = os.path.join(
        os.path.dirname(__file__),
         "assets", "boards", "standard_10x10.json",
    )
    try:
        with open(board_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            cells = data.get("cells", [])
            if isinstance(cells, list) and cells:
                return cells
    except Exception:
        pass
    return [["" for _ in range(10)] for _ in range(10)]

def _cards_per_player(total_seats: int) -> int:
    if total_seats == 2:
        return 7
    if total_seats in (3, 4):
        return 6
    if total_seats in (5, 6):
        return 5
    if total_seats == 8:
        return 4
    if total_seats >= 10:
        return 3
    return 4

# -------------------------
# Helpers transaccionales
# -------------------------

@admin_fs.transactional
def _txn_create_match(
        transaction: Transaction,
        match_ref,
        match_data: Dict[str, Any],
        seats: Dict[str, Dict[str, Any]],
) -> None:
    # 1) Crear match y asientos
    transaction.set(match_ref, match_data)
    for seat_id, seat_val in seats.items():
        seat_ref = match_ref.collection("players").document(seat_id)
        transaction.set(seat_ref, seat_val)

    # 2) Tablero inicial SIN arrays-anidados: usamos boardRows = [{cells: [...]}, ...]
    cells = _read_board_cells()
    board_rows: List[Dict[str, Any]] = []
    for r in range(len(cells)):
        row_cells: List[Dict[str, Any]] = []
        for c in range(len(cells[r])):
            row_cells.append({"card": cells[r][c], "chip": None})
        board_rows.append({"cells": row_cells})

    state_doc = {
        "turnIndex": 0,
        "currentSeatId": None,
        "phase": "lobby",
        "deck": {"cards": [], "discardPile": [], "burnedCards": []},
        "hands": {},
        "boardRows": board_rows,  # <-- no nested arrays
        "sequences": {str(i): 0 for i in range(match_data["config"]["teams"])},
        "sequencesMeta": [],          # <-- NEW
        "sequenceCells": [],          # <-- NEW
        "winners": [],
        "lastMove": None,
        "roundCount": 0,
        "discardPileCount": 0,
        "burnedCount": 0,
    }
    state_ref = match_ref.collection("state").document("state")
    transaction.set(state_ref, state_doc)

@admin_fs.transactional
def _txn_claim_seat(
    transaction: Transaction,
    match_ref,
    seat_code: str,
    uid: Optional[str],
    display_name: Optional[str],
) -> str:
    players = list(match_ref.collection("players").stream(transaction=transaction))
    claimed_seat_id: Optional[str] = None

    for p in players:
        pd = p.to_dict()
        if pd.get("seatCode") == seat_code:
            if pd.get("uid"):
                raise HttpsError(
                    FunctionsErrorCode.ALREADY_EXISTS,
                    ErrorCode.ERR_SEAT_TAKEN.value,
                    details={"error": ErrorCode.ERR_SEAT_TAKEN.value},
                )
            claimed_seat_id = p.id
            seat_ref = match_ref.collection("players").document(claimed_seat_id)
            transaction.update(
                seat_ref,
                {
                    "uid": uid,
                    "displayName": display_name,
                    "connected": True,
                    "lastPingAt": admin_fs.SERVER_TIMESTAMP,
                    "isReady": True,
                },
            )
            break

    if claimed_seat_id is None:
        raise HttpsError(
            FunctionsErrorCode.INVALID_ARGUMENT,
            ErrorCode.ERR_INVALID_SEAT_CODE.value,
            details={"error": ErrorCode.ERR_INVALID_SEAT_CODE.value},
        )
    return claimed_seat_id


@admin_fs.transactional
def _txn_start_game(transaction: Transaction, match_ref, rng_seed: int) -> None:
    m_snap = match_ref.get(transaction=transaction)
    if not m_snap.exists:
        raise https_fn.HttpsError("not-found", "Match not found.")
    m_data = m_snap.to_dict()

    if m_data.get("status") != "lobby":
        return  # nada que hacer

    players_docs = list(match_ref.collection("players").stream(transaction=transaction))
    for seat in players_docs:
        sd = seat.to_dict()
        if not sd.get("isAgent") and not sd.get("uid"):
            return  # aún no listo

    full_deck = create_full_deck() + create_full_deck()
    shuffle_deck(full_deck, seed=rng_seed)

    teams = int(m_data["config"]["teams"])
    players_per_team = int(m_data["config"]["playersPerTeam"])
    total_seats = teams * players_per_team

    hand_size = _cards_per_player(total_seats)
    players_sorted = sorted(players_docs, key=lambda d: d.to_dict()["seatIndex"])

    deck_cards = full_deck[:]
    hands: Dict[str, List[str]] = {}
    for seat_doc in players_sorted:
        seat_id = seat_doc.id
        hand_cards = [deck_cards.pop() for _ in range(hand_size)]
        hands[seat_id] = hand_cards

    current_seat = players_sorted[0].id  # podrías randomizar

    state_ref = match_ref.collection("state").document("state")
    transaction.update(state_ref, {
        "turnIndex": 0,
        "currentSeatId": current_seat,
        "phase": "play",
        "deck": {"cards": deck_cards, "discardPile": [], "burnedCards": []},
        "hands": hands,
    })
    transaction.update(match_ref, {"status": "active"})

@admin_fs.transactional
def _txn_apply_move(
    transaction: Transaction,
    match_ref,
    seat_id: str,
    move_type: str,
    card: Optional[str],
    target: Optional[Dict[str, int]],
    removed: Optional[Dict[str, int]],
) -> None:
    m_snap = match_ref.get(transaction=transaction)
    if not m_snap.exists:
        raise HttpsError(
            FunctionsErrorCode.NOT_FOUND,
            ErrorCode.ERR_MATCH_NOT_FOUND.value,
            details={"error": ErrorCode.ERR_MATCH_NOT_FOUND.value},
        )
    m_data = m_snap.to_dict()
    if m_data.get("status") != "active":
        raise HttpsError(
            FunctionsErrorCode.FAILED_PRECONDITION,
            ErrorCode.ERR_MATCH_NOT_ACTIVE.value,
            details={"error": ErrorCode.ERR_MATCH_NOT_ACTIVE.value},
        )

    state_ref = match_ref.collection("state").document("state")
    st_snap = state_ref.get(transaction=transaction)
    st_data = st_snap.to_dict() or {}

    if st_data.get("currentSeatId") != seat_id:
        raise HttpsError(
            FunctionsErrorCode.PERMISSION_DENIED,
            ErrorCode.ERR_NOT_YOUR_TURN.value,
            details={"error": ErrorCode.ERR_NOT_YOUR_TURN.value},
        )

    if move_type not in ("timeout-skip", "burn"):
        hands = st_data.get("hands", {})
        hand = list(hands.get(seat_id, []))
        if card not in hand:
            raise HttpsError(
                FunctionsErrorCode.INVALID_ARGUMENT,
                ErrorCode.ERR_CARD_NOT_IN_HAND.value,
                details={"error": ErrorCode.ERR_CARD_NOT_IN_HAND.value},
            )

    players = list(match_ref.collection("players").stream(transaction=transaction))
    team_index = None
    for p in players:
        if p.id == seat_id:
            team_index = p.to_dict().get("teamIndex")
            break
    if team_index is None:
        raise HttpsError(
            FunctionsErrorCode.INVALID_ARGUMENT,
            ErrorCode.ERR_SEAT_NOT_FOUND.value,
            details={"error": ErrorCode.ERR_SEAT_NOT_FOUND.value},
        )

    try:
        new_state, move_record = apply_move_to_state(
            st_data, seat_id, team_index, move_type, card, target, removed, m_data["config"]
        )
    except EngineError as ee:
        print("EngineError:", ee.code.value, ee.details)
        raise HttpsError(
            FunctionsErrorCode.INVALID_ARGUMENT,
            ee.code.value,
            details={"error": ee.code.value, "meta": ee.details},
        )
    except Exception as e:
        print("Unexpected error in apply_move_to_state:", e)
        raise HttpsError(
            FunctionsErrorCode.INTERNAL,
            ErrorCode.ERR_INTERNAL.value,
            details={"error": ErrorCode.ERR_INTERNAL.value},
        )

    transaction.update(state_ref, new_state)

    mv_ref = match_ref.collection("moves").document()
    transaction.set(mv_ref, move_record)

    players_sorted = sorted(players, key=lambda d: d.to_dict()["seatIndex"])
    cur_idx = next((i for i, d in enumerate(players_sorted) if d.id == seat_id), 0)
    nxt_idx = (cur_idx + 1) % len(players_sorted)
    next_seat_id = players_sorted[nxt_idx].id
    transaction.update(
        state_ref,
        {
            "currentSeatId": next_seat_id,
            "turnIndex": (st_data.get("turnIndex", 0) + 1),
        },
    )

    if new_state.get("winners"):
        transaction.update(match_ref, {"status": "finished"})


@admin_fs.transactional
def _txn_finalize_stats(transaction: Transaction, match_ref) -> None:
    m_snap = match_ref.get(transaction=transaction)
    if not m_snap.exists:
        raise https_fn.HttpsError("not-found", "Match not found.")
    m_data = m_snap.to_dict()
    if m_data.get("status") != "finished":
        return

    st_snap = match_ref.collection("state").document("state").get(transaction=transaction)
    winners = (st_snap.to_dict() or {}).get("winners", [])
    teams = int(m_data["config"]["teams"])

    players = list(match_ref.collection("players").stream(transaction=transaction))
    for p in players:
        pd = p.to_dict()
        u = pd.get("uid")
        if not u:
            continue
        user_ref = db.collection("users").document(u)
        user_snapshot = user_ref.get(transaction=transaction)
        user_data = user_snapshot.to_dict() if user_snapshot.exists else {}
        stats = user_data.get(
            "stats",
            {
                "wins2": 0, "losses2": 0, "draws2": 0,
                "wins3": 0, "losses3": 0, "draws3": 0,
                "sequencesMade": 0,
            },
        )
        team_idx = pd.get("teamIndex")
        won = team_idx in winners
        mode = f"{teams}"
        if won:
            stats[f"wins{mode}"] = stats.get(f"wins{mode}", 0) + 1
        else:
            stats[f"losses{mode}"] = stats.get(f"losses{mode}", 0) + 1
        transaction.set(user_ref, {"stats": stats}, merge=True)

@admin_fs.transactional
def _txn_respond_friend_request(
        transaction: Transaction,
        fr_ref,
        uid: str,
        action: str,
) -> None:
    fr_snap = fr_ref.get(transaction=transaction)
    if not fr_snap.exists:
        raise https_fn.HttpsError("not-found", "Request not found.")
    fr = fr_snap.to_dict()
    if fr.get("toUid") != uid:
        raise https_fn.HttpsError("permission-denied", "Not authorized.")

    transaction.update(fr_ref, {"status": action})
    if action == "accept":
        from_uid = fr.get("fromUid")
        to_uid = fr.get("toUid")
        from_ref = db.collection("users").document(from_uid)
        to_ref = db.collection("users").document(to_uid)

        from_user = from_ref.get(transaction=transaction).to_dict() or {}
        to_user = to_ref.get(transaction=transaction).to_dict() or {}

        from_friends = set(from_user.get("friends", []))
        to_friends = set(to_user.get("friends", []))
        from_friends.add(to_uid)
        to_friends.add(from_uid)

        transaction.set(from_ref, {"friends": list(from_friends)}, merge=True)
        transaction.set(to_ref, {"friends": list(to_friends)}, merge=True)

# -------------------------
# Endpoints callable
# -------------------------

@https_fn.on_call()
def create_match(req: https_fn.CallableRequest) -> Dict[str, Any]:
    # ¡PON TU BREAKPOINT AQUÍ! El IDE se conectará cuando PYCHARM_DEBUG=1.
    uid = _require_uid(req)
    data = req.data or {}
    config = data.get("config", {}) or {}
    public = bool(data.get("assets", False))
    require_seat_codes = not public

    teams = int(config.get("teams", 2))
    players_per_team = int(config.get("playersPerTeam", 2))

    match_ref = db.collection("matches").document()
    match_id = match_ref.id
    join_code = "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))

    total_seats = teams * players_per_team
    seats: Dict[str, Dict[str, Any]] = {}
    for seat_index in range(total_seats):
        code = "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        seats[str(seat_index)] = {
            "seatIndex": seat_index,
            "teamIndex": seat_index % teams,
            "uid": None,
            "displayName": None,
            "photoURL": None,
            "isAgent": False,
            "seatCode": code,
            "connected": False,
            "lastPingAt": admin_fs.SERVER_TIMESTAMP,
            "isReady": False,
        }

    match_data = {
        "ownerUid": uid,
        "createdAt": admin_fs.SERVER_TIMESTAMP,
        "status": "lobby",
        "config": {
            "teams": teams,
            "playersPerTeam": players_per_team,
            "allowAdvancedJack": bool(config.get("allowAdvancedJack", False)),
            "allowDraws": bool(config.get("allowDraws", False)),
            "turnSeconds": int(config.get("turnSeconds", 60)),
            "totalMinutes": int(config.get("totalMinutes", 30)),
            "requireSeatCodes": require_seat_codes,
            "assets": public,
        },
        "security": {
            "joinCode": join_code,
            "requireAuth": not public,
        },
        "boardRef": "boards/standard_10x10.json",
        "agentWeightsURL": data.get("agentWeightsURL"),
        "storageWeightsPath": None,
        "rngSeed": int(time.time() * 1000),
    }

    transaction = db.transaction()
    _txn_create_match(transaction, match_ref, match_data, seats)

    return {
        "matchId": match_id,
        "joinLink": f"/m/{match_id}",
        "joinCode": join_code,
        "seatCodes": {i: seats[i]["seatCode"] for i in seats},
    }

@https_fn.on_call()
def join_match(req: https_fn.CallableRequest) -> Dict[str, Any]:
    uid = _uid_or_none(req)  # spectator allowed
    data = req.data or {}
    match_id = data.get("matchId")
    join_code = data.get("joinCode")
    seat_code = data.get("seatCode")
    display_name = (data.get("displayName") or "").strip() or ("Anonymous" if uid is None else None)

    if not match_id:
        raise HttpsError(
            FunctionsErrorCode.INVALID_ARGUMENT,
            ErrorCode.ERR_INPUT_MISSING.value,
            details={"error": ErrorCode.ERR_INPUT_MISSING.value},
        )

    match_ref = db.collection("matches").document(match_id)
    snap = match_ref.get()
    if not snap.exists:
        raise HttpsError(
            FunctionsErrorCode.NOT_FOUND,
            ErrorCode.ERR_MATCH_NOT_FOUND.value,
            details={"error": ErrorCode.ERR_MATCH_NOT_FOUND.value},
        )
    match_data = snap.to_dict()

    is_public = bool(match_data.get("config", {}).get("assets", False))
    if not is_public:
        expected = match_data.get("security", {}).get("joinCode")
        if not join_code or join_code != expected:
            raise HttpsError(
                FunctionsErrorCode.PERMISSION_DENIED,
                ErrorCode.ERR_INVALID_JOIN_CODE.value,
                details={"error": ErrorCode.ERR_INVALID_JOIN_CODE.value},
            )

    claimed_seat_id: Optional[str] = None
    if seat_code:
        transaction = db.transaction()
        claimed_seat_id = _txn_claim_seat(transaction, match_ref, seat_code, uid, display_name)

    return {"ok": True, "seatId": claimed_seat_id}

@https_fn.on_call()
def start_if_ready(req: https_fn.CallableRequest) -> Dict[str, Any]:
    data = req.data or {}
    match_id = data.get("matchId")
    if not match_id:
        raise https_fn.HttpsError("invalid-argument", "matchId required.")

    match_ref = db.collection("matches").document(match_id)
    snap = match_ref.get()
    if not snap.exists:
        raise https_fn.HttpsError("not-found", "Match not found.")
    m = snap.to_dict()
    if m.get("status") != "lobby":
        return {"ok": False, "reason": "Already started"}

    transaction = db.transaction()
    _txn_start_game(transaction, match_ref, rng_seed=m.get("rngSeed", int(time.time() * 1000)))
    return {"ok": True}

@https_fn.on_call()
def submit_move(req: https_fn.CallableRequest) -> Dict[str, Any]:
    _ = _uid_or_none(req)  # optional auth for spectators
    data = req.data or {}
    ide_attach(port=5678, suspend=True)  # optional debugger


    match_id = data.get("matchId")
    seat_id = data.get("seatId")
    move_type = data.get("type")
    card = data.get("card")
    target = data.get("target")
    removed = data.get("removed")

    if not match_id or not seat_id or not move_type:
        raise HttpsError(
            FunctionsErrorCode.INVALID_ARGUMENT,
            ErrorCode.ERR_INPUT_MISSING.value,
            details={"error": ErrorCode.ERR_INPUT_MISSING.value},
        )

    match_ref = db.collection("matches").document(match_id)
    transaction = db.transaction()
    _txn_apply_move(transaction, match_ref, seat_id, move_type, card, target, removed)
    return {"ok": True}

@https_fn.on_call()
def get_public_state(req: https_fn.CallableRequest) -> Dict[str, Any]:
    data = req.data or {}
    match_id = data.get("matchId")
    seat_id = data.get("seatId")

    if not match_id:
        raise HttpsError(
            FunctionsErrorCode.INVALID_ARGUMENT,
            ErrorCode.ERR_INPUT_MISSING.value,
            details={"error": ErrorCode.ERR_INPUT_MISSING.value},
        )

    state_ref = db.collection("matches").document(match_id).collection("state").document("state")
    snap = state_ref.get()
    if not snap.exists:
        raise HttpsError(
            FunctionsErrorCode.NOT_FOUND,
            ErrorCode.ERR_STATE_NOT_FOUND.value,
            details={"error": ErrorCode.ERR_STATE_NOT_FOUND.value},
        )
    st = snap.to_dict() or {}

    pub = {k: v for k, v in st.items() if k not in ("hands", "boardRows")}
    if "boardRows" in st:
        pub["board"] = [row.get("cells", []) for row in st["boardRows"]]
    if seat_id:
        pub["hand"] = st.get("hands", {}).get(seat_id, [])
    return pub

@https_fn.on_call()
def post_message(req: https_fn.CallableRequest) -> Dict[str, Any]:
    uid = _uid_or_none(req)
    data = req.data or {}
    match_id = data.get("matchId")
    text = (data.get("text") or "").strip()
    team_only = bool(data.get("teamOnly", False))

    if not match_id or not text:
        raise HttpsError(
            FunctionsErrorCode.INVALID_ARGUMENT,
            ErrorCode.ERR_INPUT_MISSING.value,
            details={"error": ErrorCode.ERR_INPUT_MISSING.value},
        )

    match_ref = db.collection("matches").document(match_id)
    if not match_ref.get().exists:
        raise HttpsError(
            FunctionsErrorCode.NOT_FOUND,
            ErrorCode.ERR_MATCH_NOT_FOUND.value,
            details={"error": ErrorCode.ERR_MATCH_NOT_FOUND.value},
        )

    display = "Anonymous"
    team_index = None
    if uid:
        for p in match_ref.collection("players").stream():
            pd = p.to_dict()
            if pd.get("uid") == uid:
                display = pd.get("displayName") or "Player"
                team_index = pd.get("teamIndex")
                break

    chat_doc = {
        "uid": uid,
        "displayName": display,
        "text": text,
        "createdAt": admin_fs.SERVER_TIMESTAMP,
        "teamOnly": team_only,
        "teamIndex": team_index,
    }
    match_ref.collection("chat").add(chat_doc)
    return {"ok": True}

@https_fn.on_call()
def heartbeat(req: https_fn.CallableRequest) -> Dict[str, Any]:
    uid = _require_uid(req)
    data = req.data or {}
    match_id = data.get("matchId")
    seat_id = data.get("seatId")
    if not match_id or not seat_id:
        raise HttpsError(
            FunctionsErrorCode.INVALID_ARGUMENT,
            ErrorCode.ERR_INPUT_MISSING.value,
            details={"error": ErrorCode.ERR_INPUT_MISSING.value},
        )
    player_ref = db.collection("matches").document(match_id).collection("players").document(seat_id)
    player_ref.update({"connected": True, "lastPingAt": admin_fs.SERVER_TIMESTAMP})
    return {"ok": True}

@https_fn.on_call()
def finalize_match(req: https_fn.CallableRequest) -> Dict[str, Any]:
    data = req.data or {}
    match_id = data.get("matchId")
    if not match_id:
        raise HttpsError(
            FunctionsErrorCode.INVALID_ARGUMENT,
            ErrorCode.ERR_INPUT_MISSING.value,
            details={"error": ErrorCode.ERR_INPUT_MISSING.value},
        )

    match_ref = db.collection("matches").document(match_id)
    if not match_ref.get().exists:
        raise HttpsError(
            FunctionsErrorCode.NOT_FOUND,
            ErrorCode.ERR_MATCH_NOT_FOUND.value,
            details={"error": ErrorCode.ERR_MATCH_NOT_FOUND.value},
        )

    transaction = db.transaction()
    _txn_finalize_stats(transaction, match_ref)
    return {"ok": True}


@https_fn.on_call()
def send_friend_request(req: https_fn.CallableRequest) -> Dict[str, Any]:
    from_uid = _require_uid(req)
    data = req.data or {}
    to_uid = data.get("toUid")
    if not to_uid or to_uid == from_uid:
        raise HttpsError(
            FunctionsErrorCode.INVALID_ARGUMENT,
            ErrorCode.ERR_INVALID_TO_UID.value,
            details={"error": ErrorCode.ERR_INVALID_TO_UID.value},
        )
    fr_ref = db.collection("friendRequests").document()
    fr_ref.set(
        {
            "fromUid": from_uid,
            "toUid": to_uid,
            "status": "pending",
            "createdAt": admin_fs.SERVER_TIMESTAMP,
        }
    )
    return {"ok": True, "requestId": fr_ref.id}

@https_fn.on_call()
def respond_friend_request(req: https_fn.CallableRequest) -> Dict[str, Any]:
    uid = _require_uid(req)
    data = req.data or {}
    req_id = data.get("requestId")
    action = data.get("action")
    if action not in ("accept", "decline"):
        raise HttpsError(
            FunctionsErrorCode.INVALID_ARGUMENT,
            ErrorCode.ERR_INVALID_ACTION.value,
            details={"error": ErrorCode.ERR_INVALID_ACTION.value},
        )

    fr_ref = db.collection("friendRequests").document(req_id)
    transaction = db.transaction()
    try:
        _txn_respond_friend_request(transaction, fr_ref, uid, action)
    except HttpsError as he:
        # bubble up shaped errors (e.g., not found / permission)
        raise he
    except Exception:
        # If _txn_respond_friend_request raises generic exceptions, convert here
        raise HttpsError(
            FunctionsErrorCode.PERMISSION_DENIED,
            ErrorCode.ERR_NOT_AUTHORIZED.value,
            details={"error": ErrorCode.ERR_NOT_AUTHORIZED.value},
        )
    return {"ok": True}

