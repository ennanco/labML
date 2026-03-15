"""Error taxonomy for benchmark execution paths."""

from __future__ import annotations

import typer


ERROR_TYPE_CONFIG_DATA = "config_data"
ERROR_TYPE_INCOMPATIBLE_COMBO = "incompatible_combo"
ERROR_TYPE_MODEL_EXECUTION = "model_execution"
ERROR_TYPE_INTERNAL = "internal"


class ModelExecutionError(RuntimeError):
    """Recoverable error raised while fitting/scoring one combination."""


def classify_exception(exc: Exception) -> str:
    """Map exceptions to a stable error taxonomy label."""
    if isinstance(exc, typer.BadParameter):
        return ERROR_TYPE_CONFIG_DATA
    if isinstance(exc, ModelExecutionError):
        return ERROR_TYPE_MODEL_EXECUTION
    return ERROR_TYPE_INTERNAL
