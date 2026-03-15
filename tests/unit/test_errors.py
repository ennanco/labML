import typer

from labml.core.errors import (
    ERROR_TYPE_CONFIG_DATA,
    ERROR_TYPE_INTERNAL,
    ERROR_TYPE_MODEL_EXECUTION,
    ModelExecutionError,
    classify_exception,
)


def test_classify_exception_for_config_data() -> None:
    exc = typer.BadParameter("bad config")
    assert classify_exception(exc) == ERROR_TYPE_CONFIG_DATA


def test_classify_exception_for_model_execution() -> None:
    exc = ModelExecutionError("model failed")
    assert classify_exception(exc) == ERROR_TYPE_MODEL_EXECUTION


def test_classify_exception_for_internal_error() -> None:
    exc = RuntimeError("boom")
    assert classify_exception(exc) == ERROR_TYPE_INTERNAL
