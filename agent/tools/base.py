"""Abstract interface for agent tools.

Every tool the LLM can invoke subclasses :class:`BaseTool` and is registered
in :mod:`agent.tools`. The orchestrator's executor catches all exceptions
raised by :meth:`BaseTool.execute` and feeds them back to the LLM as text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class BaseTool(ABC):
    """Abstract base class for an agent tool.

    Subclasses must set the three class attributes (``name``, ``description``,
    ``schema``) and implement :meth:`execute`.

    Attributes:
        name: Stable identifier referenced by the LLM in tool-use blocks.
        description: Human-readable summary, used in logs and prompts.
        schema: JSON Schema describing the ``params`` accepted by
            :meth:`execute`.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    schema: ClassVar[dict[str, Any]]

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> str:
        """Run the tool and return its result as plain text.

        Args:
            params: Parameters validated against :attr:`schema`.

        Returns:
            Plain-text result that will be appended to the LLM history.
        """
