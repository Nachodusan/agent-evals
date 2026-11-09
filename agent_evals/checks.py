"""Built-in trajectory checks.

Outcome checks answer "was it right". These answer "how did it get there",
which is the part that decides whether an agent is affordable, safe, and worth
keeping once it is right.

Every check is a small callable with a readable `name`, so a failing report
names the property that broke rather than an index into a list.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_evals.types import CheckResult, Run


class _Check:
    def __init__(self, name: str, fn: Callable[[Run], tuple[bool, str]]):
        self.name = name
        self._fn = fn

    def __call__(self, run: Run) -> CheckResult:
        passed, detail = self._fn(run)
        return CheckResult(self.name, passed, detail)


# --- what it called -------------------------------------------------------

def calls(tool: str, *, at_least: int = 1, at_most: int | None = None) -> _Check:
    """The agent used a tool, within a count range."""
    bound = f"≥{at_least}" + (f" and ≤{at_most}" if at_most is not None else "")

    def fn(run: Run) -> tuple[bool, str]:
        n = run.names().count(tool)
        ok = n >= at_least and (at_most is None or n <= at_most)
        return ok, f"called {n}×"

    return _Check(f"calls {tool} ({bound})", fn)


def never_calls(tool: str) -> _Check:
    """The agent stayed away from a tool entirely.

    Useful for distractors, and for anything with a side effect that should
    not happen while answering a read-only question.
    """

    def fn(run: Run) -> tuple[bool, str]:
        n = run.names().count(tool)
        return n == 0, "never called" if n == 0 else f"called {n}×"

    return _Check(f"never calls {tool}", fn)


def calls_before(first: str, second: str) -> _Check:
    """Ordering. The canonical case is reading a schema before querying it."""

    def fn(run: Run) -> tuple[bool, str]:
        names = run.names()
        if second not in names:
            return True, f"{second} never called"
        if first not in names:
            return False, f"{second} called without {first}"
        ok = names.index(first) < names.index(second)
        return ok, f"path: {' → '.join(run.path)}"

    return _Check(f"{first} before {second}", fn)


# --- how well it went -----------------------------------------------------

def no_failures() -> _Check:
    """Nothing errored. A clean run."""

    def fn(run: Run) -> tuple[bool, str]:
        bad = run.failed_calls
        if not bad:
            return True, "no failed calls"
        return False, f"{len(bad)} failed: " + ", ".join(c.name for c in bad)

    return _Check("no failed tool calls", fn)


def recovers() -> _Check:
    """Something failed and the agent carried on to a successful call.

    Note what this does *not* assert: that anything failed at all. A run with
    no failures passes, because the property being tested is "failure is not
    fatal", not "please fail".
    """

    def fn(run: Run) -> tuple[bool, str]:
        for i, call in enumerate(run.calls):
            if not call.ok:
                later = run.calls[i + 1:]
                if not any(c.ok for c in later):
                    return False, f"{call.name} failed and nothing succeeded after it"
        return True, "no unrecovered failure"

    return _Check("recovers from failures", fn)


def finished() -> _Check:
    """The run ended because the agent was done, not because it ran out."""

    def fn(run: Run) -> tuple[bool, str]:
        reason = run.stopped_reason or "unknown"
        return reason in ("finished", "ok", "done"), f"stopped: {reason}"

    return _Check("terminated cleanly", fn)


# --- what it cost ---------------------------------------------------------

def max_steps(n: int) -> _Check:
    def fn(run: Run) -> tuple[bool, str]:
        return run.steps <= n, f"{run.steps} steps"

    return _Check(f"at most {n} steps", fn)


def max_cost(usd: float) -> _Check:
    def fn(run: Run) -> tuple[bool, str]:
        return run.cost_usd <= usd, f"${run.cost_usd:.4f}"

    return _Check(f"at most ${usd:.4f}", fn)


def no_repeated_calls() -> _Check:
    """The agent never made the identical call twice.

    Repeating a call with the same arguments is work it already paid for and
    an answer it already had. It is the cheapest signal that a loop is
    spinning rather than progressing.
    """

    def fn(run: Run) -> tuple[bool, str]:
        seen: set[tuple[str, str]] = set()
        for call in run.calls:
            key = (call.name, repr(sorted(call.arguments.items())))
            if key in seen:
                return False, f"repeated {call.name} with identical arguments"
            seen.add(key)
        return True, f"{len(seen)} distinct calls"

    return _Check("no repeated calls", fn)


# --- what it answered -----------------------------------------------------

def answer(predicate: Callable[[Any], bool], *, name: str = "answer is correct") -> _Check:
    """Your outcome check.

    The library stays out of this on purpose: comparing SQL result sets,
    grading prose and matching a number are different problems, and a generic
    'is it right' helper would be wrong for all three.
    """

    def fn(run: Run) -> tuple[bool, str]:
        try:
            ok = bool(predicate(run.answer))
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return ok, "" if ok else "predicate returned False"

    return _Check(name, fn)
