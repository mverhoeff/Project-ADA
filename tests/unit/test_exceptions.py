"""Unit tests for :mod:`core.exceptions`."""

from __future__ import annotations

import pytest

from core.exceptions import (
    ADAError,
    ConfigurationError,
    ServiceUnavailableError,
    ToolExecutionError,
    VRAMCriticalError,
)


def test_ada_error_stores_both_messages() -> None:
    err = ADAError("internal: socket refused", "I can't reach the assistant.")
    assert str(err) == "internal: socket refused"
    assert err.user_message == "I can't reach the assistant."


@pytest.mark.parametrize(
    "subclass",
    [
        ServiceUnavailableError,
        ConfigurationError,
        ToolExecutionError,
        VRAMCriticalError,
    ],
)
def test_subclasses_are_catchable_as_ada_error(
    subclass: type[ADAError],
) -> None:
    with pytest.raises(ADAError) as info:
        raise subclass("technical detail", "spoken detail")
    assert isinstance(info.value, subclass)
    assert info.value.user_message == "spoken detail"


def test_user_message_distinct_from_technical_message() -> None:
    err = ConfigurationError(
        "missing key 'llm.model' in config/default.yaml",
        "My configuration is incomplete.",
    )
    assert err.user_message != str(err)
