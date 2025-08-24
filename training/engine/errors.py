"""
Error codes and EngineError exception for rule violations.
"""
from enum import Enum

class ErrorCode(str, Enum):
    ERR_GENERIC = "ERR_GENERIC"
    ERR_CARD_NOT_IN_HAND = "ERR_CARD_NOT_IN_HAND"
    ERR_TARGET_OCCUPIED = "ERR_TARGET_OCCUPIED"
    ERR_NOT_MATCHING_CARD = "ERR_NOT_MATCHING_CARD"
    ERR_INVALID_JACK_USE = "ERR_INVALID_JACK_USE"
    ERR_CANNOT_REMOVE_OWN_CHIP = "ERR_CANNOT_REMOVE_OWN_CHIP"
    ERR_CARD_BLOCKED = "ERR_CARD_BLOCKED"
    ERR_UNKNOWN_MOVE = "ERR_UNKNOWN_MOVE"

class EngineError(Exception):
    def __init__(self, code: ErrorCode, message: str = "", details: dict = None):
        super().__init__(message or code.value)
        self.code = code
        self.details = details or {}

    def to_dict(self):
        return {"code": self.code.value, "details": self.details}
