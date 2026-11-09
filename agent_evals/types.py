"""The shape this library needs from your agent.

Deliberately small. Any agent that can say which tools it called, in what
order, which of them failed, and what it finally answered can be evaluated
here. Everything else — frameworks, providers, message formats — stays your
problem, which is what keeps this usable against an agent it has never seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolCall:
    name: str
    ok: bool = True
    arguments: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Run:
    """One attempt at one question."""

    calls: list[ToolCall] = field(default_factory=list)
    answer: Any = None
    steps: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    stopped_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> list[str]:
        """Tool names in order, failures marked with a trailing '!'."""
        return [c.name if c.ok else f"{c.name}!" for c in self.calls]

    @property
    def failed_calls(self) -> list[ToolCall]:
        return [c for c in self.calls if not c.ok]

    def names(self) -> list[str]:
        return [c.name for c in self.calls]


@runtime_checkable
class Runner(Protocol):
    """Anything that answers a question and reports what it did."""

    def __call__(self, question: str) -> Run: ...


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


class Check(Protocol):
    """A predicate over a single run.

    `name` is what shows up in the report, so it should read as the property
    being asserted rather than as a function call.
    """

    name: str

    def __call__(self, run: Run) -> CheckResult: ...
