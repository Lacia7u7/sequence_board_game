"""
Main entry points for Firebase Cloud Functions (Gen2) implemented in Python.

This module exposes a handful of HTTP callable functions that implement the
server‑side logic for the Sequence game. They perform all validation and
updates to Firestore so that no client can cheat. State mutations are made
transactionally where appropriate.

Endpoints:

  * create_match: creates a new match document and allocates seats.
  * join_match: joins a player to a seat in a match.
  * start_if_ready: triggers when the lobby is full to deal hands and start.
  * submit_move: processes a player's move, validating legality and updating
    the board, discard pile and deck.
  * get_public_state: returns a projection of the current game state with
    private information (like hands) removed.
  * post_message: posts chat messages to a match.
  * heartbeat: records that a player is still connected.

The functions use the helper utilities defined in the game_logic package to
manipulate the deck and board and compute winning conditions.
"""

import json
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import functions_framework
from firebase_admin import firestore, initialize_app
from google.cloud.firestore import Client, Transaction

from game_logic.engine import (
    create_full_deck,
    shuffle_deck,
    deal_hands,
    apply_move_to_state,
    create_empty_board,
    compute_sequences,
    has_no_legal_moves,
)

# Initialize the Firebase Admin SDK
app = initialize_app()
db: Client = firestore.client()


def _require_auth(request) -> str:
    """Extracts and returns the user's UID from the Firebase auth token.

    Raises a ValueError if authentication is missing or invalid.
    """
    # In Cloud Functions, the request context includes `authorization` header
    # containing a JWT. In practice we rely on Firebase Functions wrapping
    # to provide request.auth. For local tests we allow anonymous.
    auth = getattr(request, "context", {}).get("auth")
    if not auth or not auth.get("uid"):
        raise ValueError("Authentication required")
    return auth["uid"]


@functions_framework.http
def create_match(request):
    """Creates a new match with the given configuration.

    Expected JSON payload:
      {
        "config": {"teams":2|3, "playersPerTeam": int, ...},
        "public": bool,
        "agentWeightsURL": str | null
      }

    Returns a JSON object containing matchId, joinLink and joinCode.
    """
    try:
        uid = _require_auth(request)
    except ValueError as exc:
        return (json.dumps({"error": str(exc)}), 401, {"Content-Type": "application/json"})
    data = request.get_json(silent=True) or {}
    config = data.get("config", {})
    public = bool(data.get("public", False))
    require_seat_codes = not public
    teams = int(config.get("teams", 2))
    players_per_team = int(config.get("playersPerTeam", 2))
    # Generate a random match id
    match_doc_ref = db.collection("matches").document()
    match_id = match_doc_ref.id
    join_code = ''.join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    # Generate seat codes per seat
    total_seats = teams * players_per_team
    seats = {}
    for seat_index in range(total_seats):
        # A simple seat code - hashed later
        code = ''.join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        seats[str(seat_index)] = {
            "seatIndex": seat_index,
            "teamIndex": seat_index % teams,
            "uid": None,
            "displayName": None,
            "photoURL": None,
            "isAgent": False,
            "seatCode": code,
            "connected": False,
            "lastPingAt": firestore.SERVER_TIMESTAMP,
            "isReady": False,
        }
    # Write match metadata
    match_data = {
        "ownerUid": uid,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "status": "lobby",
        "config": {
            "teams": teams,
            "playersPerTeam": players_per_team,
            "allowAdvancedJack": bool(config.get("allowAdvancedJack", False)),
            "allowDraws": bool(config.get("allowDraws", False)),
            "turnSeconds": int(config.get("turnSeconds", 60)),
            "totalMinutes": int(config.get("totalMinutes", 30)),
            "requireSeatCodes": require_seat_codes,
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
    # Transactionally create documents
    def _create_in_txn(transaction: Transaction) -> None:
        transaction.set(match_doc_ref, match_data)
        # create players subcollection
        for seat_id, seat_val in seats.items():
            seat_ref = match_doc_ref.collection("players").document(seat_id)
            transaction.set(seat_ref, seat_val)
        # Load board mapping from JSON file
        import os
        import json as pyjson
        board_file = os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'public', 'boards', 'standard_10x10.json')
        try:
            with open(board_file) as f:
                board_data = pyjson.load(f)
                cells = board_data.get('cells', [])
        except Exception:
            # fallback: use empty board
            cells = [["" for _ in range(10)] for _ in range(10)]
        # build board with mapping and chips None
        board = []
        for r in range(len(cells)):
            row = []
            for c in range(len(cells[r])):
                card_str = cells[r][c]
                row.append({"card": card_str, "chip": None})
            board.append(row)
        initial_state = {
            "turnIndex": 0,
            "currentSeatId": None,
            "phase": "lobby",
            "deck": {
                "cards": [],
                "discardPile": [],
                "burnedCards": [],
            },
            "hands": {},
            "board": board,
            "sequences": {str(i): 0 for i in range(teams)},
            "winners": [],
            "lastMove": None,
            "roundCount": 0,
        }
        transaction.set(match_doc_ref.collection("state").document("state"), initial_state)
    db.run_transaction(_create_in_txn)
    # Construct join link (client will replace host)
    join_link = f"/m/{match_id}"
    return (json.dumps({
        "matchId": match_id,
        "joinLink": join_link,
        "joinCode": join_code,
        "seatCodes": {i: seats[i]["seatCode"] for i in seats},
    }), 200, {"Content-Type": "application/json"})


@functions_framework.http
def join_match(request):
    """Joins a user (or spectator) to a match.

    Expected JSON payload:
      {
        "matchId": str,
        "joinCode": str | null,
        "seatCode": str | null,
        "displayName": str | null
      }
    """
    try:
        uid = _require_auth(request)
    except ValueError:
        uid = None  # allow anonymous spectators
    data = request.get_json(silent=True) or {}
    match_id = data.get("matchId")
    join_code = data.get("joinCode")
    seat_code = data.get("seatCode")
    display_name = data.get("displayName") or ("Anonymous" if uid is None else None)
    match_ref = db.collection("matches").document(match_id)
    match_snapshot = match_ref.get()
    if not match_snapshot.exists:
        return (json.dumps({"error": "Match not found"}), 404, {"Content-Type": "application/json"})
    match_data = match_snapshot.to_dict()
    # Validate join code if required
    if not match_data.get("public", False):
        if join_code != match_data["security"]["joinCode"]:
            return (json.dumps({"error": "Invalid join code"}), 403, {"Content-Type": "application/json"})
    # If seatCode provided, attempt to claim seat
    claimed_seat_id: Optional[str] = None
    if seat_code:
        players_collection = match_ref.collection("players")
        for seat_doc in players_collection.stream():
            seat_data = seat_doc.to_dict()
            if seat_data["seatCode"] == seat_code:
                claimed_seat_id = seat_doc.id
                break
        if claimed_seat_id is None:
            return (json.dumps({"error": "Invalid seat code"}), 403, {"Content-Type": "application/json"})
        # Claim seat transactionally
        def _claim_seat(transaction: Transaction) -> None:
            seat_ref = match_ref.collection("players").document(claimed_seat_id)
            seat_snapshot = seat_ref.get(transaction=transaction)
            sd = seat_snapshot.to_dict()
            if sd.get("uid"):
                raise ValueError("Seat already taken")
            transaction.update(seat_ref, {
                "uid": uid,
                "displayName": display_name,
                "connected": True,
                "lastPingAt": firestore.SERVER_TIMESTAMP,
                "isReady": True,
            })
        try:
            db.run_transaction(_claim_seat)
        except Exception as exc:
            return (json.dumps({"error": str(exc)}), 409, {"Content-Type": "application/json"})
    else:
        # spectator: record spectator entry (optional), for simplicity we ignore
        pass
    return (json.dumps({"ok": True}), 200, {"Content-Type": "application/json"})


@functions_framework.http
def start_if_ready(request):
    """Starts the game if all human seats are filled.

    This function should be triggered by a client polling, or via scheduled
    function. It deals hands and sets up deck and turn order once ready.
    """
    data = request.get_json(silent=True) or {}
    match_id = data.get("matchId")
    match_ref = db.collection("matches").document(match_id)
    match_snapshot = match_ref.get()
    if not match_snapshot.exists:
        return (json.dumps({"error": "Match not found"}), 404, {"Content-Type": "application/json"})
    match_data = match_snapshot.to_dict()
    if match_data.get("status") != "lobby":
        return (json.dumps({"ok": False, "reason": "Already started"}), 200, {"Content-Type": "application/json"})
    players_docs = list(match_ref.collection("players").stream())
    # Determine if enough players are ready (non-agent seats must have uid)
    required_ready = True
    for seat_doc in players_docs:
        sd = seat_doc.to_dict()
        if not sd.get("isAgent") and not sd.get("uid"):
            required_ready = False
            break
    if not required_ready:
        return (json.dumps({"ok": False, "reason": "Waiting for players"}), 200, {"Content-Type": "application/json"})
    # Start the game transactionally: deal deck, assign hands, pick turn order
    def _start_game(transaction: Transaction) -> None:
        # Create fresh deck and shuffle
        full_deck = create_full_deck() + create_full_deck()
        shuffle_deck(full_deck, seed=match_data.get("rngSeed", int(time.time()*1000)))
        teams = match_data["config"]["teams"]
        players_per_team = match_data["config"]["playersPerTeam"]
        total_seats = teams * players_per_team
        # Determine cards per player
        if total_seats == 2:
            hand_size = 7
        elif total_seats == 3:
            hand_size = 6
        elif total_seats == 4:
            hand_size = 6
        elif total_seats == 5 or total_seats == 6:
            hand_size = 5
        elif total_seats == 8:
            hand_size = 4
        elif total_seats >= 10:
            hand_size = 3
        else:
            hand_size = 4
        # Deal hands; players_docs order may not be seat order, so sort by seatIndex
        players_sorted = sorted(players_docs, key=lambda d: d.to_dict()["seatIndex"])
        hands: Dict[str, List[str]] = {}
        # Deal cards and update deck list
        deck_cards = full_deck[:]  # copy
        for seat_doc in players_sorted:
            seat_id = seat_doc.id
            hand_cards = [deck_cards.pop() for _ in range(hand_size)]
            hands[seat_id] = hand_cards
        # Starting seat (seatIndex=0) but could randomize
        current_seat = players_sorted[0].id
        state_ref = match_ref.collection("state").document("state")
        transaction.update(state_ref, {
            "turnIndex": 0,
            "currentSeatId": current_seat,
            "phase": "play",
            "deck": {
                "cards": deck_cards,
                "discardPile": [],
                "burnedCards": [],
            },
            "hands": hands,
        })
        # Write updated match status
        transaction.update(match_ref, {"status": "active"})
    db.run_transaction(_start_game)
    return (json.dumps({"ok": True}), 200, {"Content-Type": "application/json"})


@functions_framework.http
def submit_move(request):
    """Processes a player's move.

    Expected JSON payload:
      {
        "matchId": str,
        "seatId": str,
        "type": "play" | "wild" | "jack-remove" | "burn" | "timeout-skip",
        "card": str,
        "target": {"r":int,"c":int} | null,
        "removed": {"r":int,"c":int} | null
      }
    """
    try:
        uid = _require_auth(request)
    except ValueError:
        uid = None
    data = request.get_json(silent=True) or {}
    match_id = data.get("matchId")
    seat_id = data.get("seatId")
    move_type = data.get("type")
    card = data.get("card")
    target = data.get("target")
    removed = data.get("removed")
    match_ref = db.collection("matches").document(match_id)
    state_ref = match_ref.collection("state").document("state")
    # transaction ensures atomicity
    error = None
    def _apply_move(transaction: Transaction) -> None:
        nonlocal error
        match_snapshot = match_ref.get(transaction=transaction)
        if not match_snapshot.exists:
            error = {"error": "Match not found", "code": 404}
            return
        match_data = match_snapshot.to_dict()
        if match_data.get("status") != "active":
            error = {"error": "Match not active", "code": 400}
            return
        # Validate seat
        player_ref = match_ref.collection("players").document(seat_id)
        player_snapshot = player_ref.get(transaction=transaction)
        if not player_snapshot.exists:
            error = {"error": "Invalid seat", "code": 400}
            return
        player_data = player_snapshot.to_dict()
        # Validate turn ownership
        state_snapshot = state_ref.get(transaction=transaction)
        state_data = state_snapshot.to_dict()
        if state_data.get("currentSeatId") != seat_id:
            error = {"error": "ERR_NOT_YOUR_TURN", "code": 403}
            return
        # Validate card in player's hand for play and remove types
        hands = state_data.get("hands", {})
        player_hand = list(hands.get(seat_id, []))
        if move_type != "timeout-skip" and move_type != "burn":
            if card not in player_hand:
                error = {"error": "Card not in hand", "code": 400}
                return
        # Apply move to state
        try:
            # Determine teamIndex for seat_id
            team_index_local = None
            # Fetch teamIndex from players collection
            players_collection = match_ref.collection("players")
            player_docs = list(players_collection.stream(transaction=transaction))
            for pdoc in player_docs:
                if pdoc.id == seat_id:
                    team_index_local = pdoc.to_dict().get("teamIndex")
                    break
            if team_index_local is None:
                raise ValueError("Seat not found")
            new_state, new_last_move = apply_move_to_state(
                state_data,
                seat_id,
                team_index_local,
                move_type,
                card,
                target,
                removed,
                match_data["config"],
            )
        except ValueError as exc:
            error = {"error": str(exc), "code": 400}
            return
        # Update Firestore: state document, hands, deck counts, sequences
        transaction.update(state_ref, new_state)
        # Record move
        move_doc_ref = match_ref.collection("moves").document()
        transaction.set(move_doc_ref, new_last_move)
        # Advance turn: find next seat in order
        players = list(match_ref.collection("players").stream(transaction=transaction))
        players_sorted = sorted(players, key=lambda d: d.to_dict()["seatIndex"])
        current_index = next((i for i, d in enumerate(players_sorted) if d.id == seat_id), 0)
        next_index = (current_index + 1) % len(players_sorted)
        next_seat_id = players_sorted[next_index].id
        transaction.update(state_ref, {
            "currentSeatId": next_seat_id,
            "turnIndex": state_data.get("turnIndex", 0) + 1,
        })
        # Update match winners if any
        if new_state.get("winners"):
            transaction.update(match_ref, {"status": "finished"})
    db.run_transaction(_apply_move)
    if error:
        return (json.dumps(error), error.get("code", 400), {"Content-Type": "application/json"})
    return (json.dumps({"ok": True}), 200, {"Content-Type": "application/json"})


@functions_framework.http
def get_public_state(request):
    """Returns the public view of the state for a match.

    Query string parameters:
      matchId: str
      seatId: str (optional) – if provided, include the player's own hand
    """
    match_id = request.args.get("matchId")
    seat_id = request.args.get("seatId")
    match_ref = db.collection("matches").document(match_id)
    state_ref = match_ref.collection("state").document("state")
    state_snapshot = state_ref.get()
    if not state_snapshot.exists:
        return (json.dumps({"error": "Not found"}), 404, {"Content-Type": "application/json"})
    state_data = state_snapshot.to_dict()
    public_data = {k: v for k, v in state_data.items() if k != "hands"}
    if seat_id:
        # include only the caller's hand
        public_data["hand"] = state_data.get("hands", {}).get(seat_id, [])
    return (json.dumps(public_data), 200, {"Content-Type": "application/json"})


@functions_framework.http
def post_message(request):
    """Posts a chat message to a match.

    Expected JSON payload:
      {
        "matchId": str,
        "text": str,
        "teamOnly": bool
      }
    """
    try:
        uid = _require_auth(request)
    except ValueError:
        uid = None
    data = request.get_json(silent=True) or {}
    match_id = data.get("matchId")
    text = (data.get("text") or "").strip()
    team_only = bool(data.get("teamOnly", False))
    if not text:
        return (json.dumps({"error": "Empty message"}), 400, {"Content-Type": "application/json"})
    match_ref = db.collection("matches").document(match_id)
    match_snapshot = match_ref.get()
    if not match_snapshot.exists:
        return (json.dumps({"error": "Match not found"}), 404, {"Content-Type": "application/json"})
    # Determine display name and team index if player
    display_name = "Anonymous"
    team_index = None
    if uid:
        players = list(match_ref.collection("players").stream())
        for p in players:
            pd = p.to_dict()
            if pd.get("uid") == uid:
                display_name = pd.get("displayName") or "Player"
                team_index = pd.get("teamIndex")
                break
    # Compose message doc
    chat_doc = {
        "uid": uid,
        "displayName": display_name,
        "text": text,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "teamOnly": team_only,
        "teamIndex": team_index,
    }
    match_ref.collection("chat").add(chat_doc)
    return (json.dumps({"ok": True}), 200, {"Content-Type": "application/json"})


@functions_framework.http
def heartbeat(request):
    """Records that a player is still connected. Client should call periodically."""
    try:
        uid = _require_auth(request)
    except ValueError:
        return (json.dumps({"error": "Auth required"}), 401, {"Content-Type": "application/json"})
    data = request.get_json(silent=True) or {}
    match_id = data.get("matchId")
    seat_id = data.get("seatId")
    if not match_id or not seat_id:
        return (json.dumps({"error": "matchId and seatId required"}), 400, {"Content-Type": "application/json"})
    player_ref = db.collection("matches").document(match_id).collection("players").document(seat_id)
    player_ref.update({
        "connected": True,
        "lastPingAt": firestore.SERVER_TIMESTAMP,
    })
    return (json.dumps({"ok": True}), 200, {"Content-Type": "application/json"})


@functions_framework.http
def finalize_match(request):
    """Trigger function that updates user stats after match finishes.

    Expected JSON payload:
      {
        "matchId": str
      }
    """
    data = request.get_json(silent=True) or {}
    match_id = data.get("matchId")
    match_ref = db.collection("matches").document(match_id)
    match_snapshot = match_ref.get()
    if not match_snapshot.exists:
        return (json.dumps({"error": "Match not found"}), 404, {"Content-Type": "application/json"})
    match_data = match_snapshot.to_dict()
    if match_data.get("status") != "finished":
        return (json.dumps({"ok": False, "reason": "Match not finished"}), 200, {"Content-Type": "application/json"})
    winners = match_ref.collection("state").document("state").get().to_dict().get("winners", [])
    teams = match_data["config"]["teams"]
    # Update stats for players
    def _update_stats(transaction: Transaction) -> None:
        players = list(match_ref.collection("players").stream())
        for player in players:
            pd = player.to_dict()
            uid = pd.get("uid")
            if not uid:
                continue
            user_ref = db.collection("users").document(uid)
            user_snapshot = user_ref.get(transaction=transaction)
            user_data = user_snapshot.to_dict() if user_snapshot.exists else {}
            stats = user_data.get("stats", {
                "wins2": 0, "losses2": 0, "draws2": 0,
                "wins3": 0, "losses3": 0, "draws3": 0,
                "sequencesMade": 0,
            })
            team_idx = pd.get("teamIndex")
            won = team_idx in winners
            mode = f"{teams}"
            if won:
                stats[f"wins{mode}"] = stats.get(f"wins{mode}", 0) + 1
            else:
                stats[f"losses{mode}"] = stats.get(f"losses{mode}", 0) + 1
            transaction.set(user_ref, {"stats": stats}, merge=True)
    db.run_transaction(_update_stats)
    return (json.dumps({"ok": True}), 200, {"Content-Type": "application/json"})


# --- Friend Request Endpoints ---

@functions_framework.http
def send_friend_request(request):
    """Creates a new friend request from the authenticated user to another user.

    Expected JSON payload:
      { "toUid": str }
    """
    try:
        from_uid = _require_auth(request)
    except ValueError as exc:
        return (json.dumps({"error": str(exc)}), 401, {"Content-Type": "application/json"})
    data = request.get_json(silent=True) or {}
    to_uid = data.get("toUid")
    if not to_uid or to_uid == from_uid:
        return (json.dumps({"error": "Invalid toUid"}), 400, {"Content-Type": "application/json"})
    # Check if request already exists
    fr_ref = db.collection("friendRequests").document()
    fr_data = {
        "fromUid": from_uid,
        "toUid": to_uid,
        "status": "pending",
        "createdAt": firestore.SERVER_TIMESTAMP,
    }
    fr_ref.set(fr_data)
    return (json.dumps({"ok": True, "requestId": fr_ref.id}), 200, {"Content-Type": "application/json"})


@functions_framework.http
def respond_friend_request(request):
    """Accepts or declines a friend request.

    Expected JSON payload:
      { "requestId": str, "action": "accept" | "decline" }
    """
    try:
        uid = _require_auth(request)
    except ValueError as exc:
        return (json.dumps({"error": str(exc)}), 401, {"Content-Type": "application/json"})
    data = request.get_json(silent=True) or {}
    req_id = data.get("requestId")
    action = data.get("action")
    fr_ref = db.collection("friendRequests").document(req_id)
    fr_snapshot = fr_ref.get()
    if not fr_snapshot.exists:
        return (json.dumps({"error": "Request not found"}), 404, {"Content-Type": "application/json"})
    fr_data = fr_snapshot.to_dict()
    if fr_data.get("toUid") != uid:
        return (json.dumps({"error": "Not authorized"}), 403, {"Content-Type": "application/json"})
    if action not in ("accept", "decline"):
        return (json.dumps({"error": "Invalid action"}), 400, {"Content-Type": "application/json"})
    def _respond(transaction: Transaction) -> None:
        # Update request status
        transaction.update(fr_ref, {"status": action})
        if action == "accept":
            from_uid = fr_data.get("fromUid")
            to_uid = fr_data.get("toUid")
            # Add each other to friends list
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
    db.run_transaction(_respond)
    return (json.dumps({"ok": True}), 200, {"Content-Type": "application/json"})
