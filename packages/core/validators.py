"""
Strict validation helpers for AnalysisPayload and AnalysisResult.

Each function raises ``ValidationError`` on failure and returns the
validated model instance on success — so callers can rely on the return
value directly without re-parsing.

Usage::

    from packages.core.validators import validate_payload, validate_result
    payload = validate_payload(raw_dict)
    result  = validate_result(raw_dict)
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .models import AnalysisPayload, AnalysisResult


def _friendly_errors(exc: ValidationError) -> list[str]:
    """Convert Pydantic v2 validation errors into human-readable messages.

    Each message follows the pattern::

        <dot-separated field path>: <description>

    so callers can surface the exact field that failed without parsing raw
    Pydantic internals.
    """
    messages: list[str] = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) if err["loc"] else "payload"
        msg = err["msg"]
        # Strip the generic pydantic prefix where present for brevity
        msg = msg.removeprefix("Value error, ")
        messages.append(f"{loc}: {msg}")
    return messages


class PayloadValidationError(ValueError):
    """Raised when AnalysisPayload validation fails.

    Attributes
    ----------
    field_errors:
        List of ``"field.path: description"`` strings, one per failing field.
    """

    def __init__(self, field_errors: list[str]) -> None:
        self.field_errors = field_errors
        bullet_list = "\n  - ".join(field_errors)
        super().__init__(f"Invalid AnalysisPayload:\n  - {bullet_list}")


class ResultValidationError(ValueError):
    """Raised when AnalysisResult validation fails.

    Attributes
    ----------
    field_errors:
        List of ``"field.path: description"`` strings, one per failing field.
    """

    def __init__(self, field_errors: list[str]) -> None:
        self.field_errors = field_errors
        bullet_list = "\n  - ".join(field_errors)
        super().__init__(f"Invalid AnalysisResult:\n  - {bullet_list}")


def validate_payload(raw: Any) -> AnalysisPayload:
    """Parse and validate *raw* as an :class:`AnalysisPayload`.

    Parameters
    ----------
    raw:
        A dict (or any mapping) representing the incoming JSON payload.

    Returns
    -------
    AnalysisPayload
        The validated model instance, ready for use downstream.

    Raises
    ------
    PayloadValidationError
        If any required field is missing, has the wrong type, or fails a
        constraint.  ``exc.field_errors`` contains one entry per failing
        field, formatted as ``"field.path: description"``.

    Examples
    --------
    >>> payload = validate_payload({"ticker": "NQU2026", ...})
    >>> payload.ticker
    'NQU2026'

    >>> try:
    ...     validate_payload({"ticker": "NQU2026"})  # missing many fields
    ... except PayloadValidationError as e:
    ...     print(e.field_errors[0])
    'date_et: Field required'
    """
    try:
        return AnalysisPayload.model_validate(raw)
    except ValidationError as exc:
        raise PayloadValidationError(_friendly_errors(exc)) from exc


def validate_result(raw: Any) -> AnalysisResult:
    """Parse and validate *raw* as an :class:`AnalysisResult`.

    Parameters
    ----------
    raw:
        A dict representing the engine's output before serialisation.

    Returns
    -------
    AnalysisResult
        The validated model instance.

    Raises
    ------
    ResultValidationError
        If any field is missing, incorrectly typed, or violates a
        constraint (e.g. a bucket has != 2 items).
    """
    try:
        return AnalysisResult.model_validate(raw)
    except ValidationError as exc:
        raise ResultValidationError(_friendly_errors(exc)) from exc
