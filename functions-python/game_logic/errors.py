# Auto-generated from errors.yaml — do not edit manually
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    ERR_GENERIC = "ERR_GENERIC"
    ERR_INTERNAL = "ERR_INTERNAL"
    ERR_UNAUTHENTICATED = "ERR_UNAUTHENTICATED"
    ERR_PERMISSION_DENIED = "ERR_PERMISSION_DENIED"
    ERR_NOT_AUTHORIZED = "ERR_NOT_AUTHORIZED"
    ERR_INPUT_MISSING = "ERR_INPUT_MISSING"
    ERR_MATCH_NOT_FOUND = "ERR_MATCH_NOT_FOUND"
    ERR_STATE_NOT_FOUND = "ERR_STATE_NOT_FOUND"
    ERR_MATCH_NOT_ACTIVE = "ERR_MATCH_NOT_ACTIVE"
    ERR_INVALID_JOIN_CODE = "ERR_INVALID_JOIN_CODE"
    ERR_INVALID_SEAT_CODE = "ERR_INVALID_SEAT_CODE"
    ERR_SEAT_TAKEN = "ERR_SEAT_TAKEN"
    ERR_SEAT_NOT_FOUND = "ERR_SEAT_NOT_FOUND"
    ERR_NOT_YOUR_TURN = "ERR_NOT_YOUR_TURN"
    ERR_CARD_NOT_IN_HAND = "ERR_CARD_NOT_IN_HAND"
    ERR_CARD_BLOCKED = "ERR_CARD_BLOCKED"
    ERR_NOT_MATCHING_CARD = "ERR_NOT_MATCHING_CARD"
    ERR_TARGET_OCCUPIED = "ERR_TARGET_OCCUPIED"
    ERR_UNKNOWN_MOVE = "ERR_UNKNOWN_MOVE"
    ERR_INVALID_ACTION = "ERR_INVALID_ACTION"
    ERR_REQUEST_NOT_FOUND = "ERR_REQUEST_NOT_FOUND"
    ERR_INVALID_TO_UID = "ERR_INVALID_TO_UID"
    ERR_INVALID_JACK_USE = "ERR_INVALID_JACK_USE"
    ERR_CANNOT_REMOVE_OWN_CHIP = "ERR_CANNOT_REMOVE_OWN_CHIP"


class EngineError(Exception):
    """Base exception for game engine errors.

    Attributes:
        code: ErrorCode enum
        message: optional human message (not shown to client; for logs)
        details: optional structured data (e.g. {'r':3,'c':2,'card':'AS'})
    """

    def __init__(
        self,
        code: ErrorCode,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.details = details or {}
        super().__init__(message or code.value)

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code.value, "details": self.details}
