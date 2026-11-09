"""Cases, and running them more than once.

The one opinion this library holds: **a case that passed once has not passed.**

Tool-calling models are non-deterministic even at temperature 0 — different
tool order, different arguments, occasionally a different answer. A harness
that runs each case once reports a number that will not reproduce, and the
first time it disagrees with itself someone stops trusting the harness instead
of the agent.

So every case runs `repeats` times and a case only passes if every attempt
passed. Anything in between is flaky, and flaky is reported as its own
category rather than rounded in either direction.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from agent_evals.types import Check, CheckResult, Run, Runner


@dataclass
class Case:
    id: str
    question: str
    checks: Sequence[Check] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)
    notes: str = ""


@dataclass
class Attempt:
    """One execution of one case."""

    index: int
    run: Run
    results: list[CheckResult]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(r.passed for r in self.results)

    @property
    def failed_checks(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]


@dataclass
class CaseOutcome:
    case: Case
    attempts: list[Attempt]

    @property
    def passes(self) -> int:
        return sum(1 for a in self.attempts if a.passed)

    @property
    def total(self) -> int:
        return len(self.attempts)

    @property
    def pass_rate(self) -> float:
        return self.passes / self.total if self.total else 0.0

    @property
    def verdict(self) -> str:
        """`pass`, `fail` or `flaky`. Never a rounded pass rate."""
        if self.passes == self.total:
            return "pass"
        if self.passes == 0:
            return "fail"
        return "flaky"

    @property
    def check_failures(self) -> dict[str, int]:
        """How many attempts each check failed, worst first."""
        counts: dict[str, int] = {}
        for attempt in self.attempts:
            for name in attempt.failed_checks:
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def distinct_paths(self) -> dict[str, int]:
        """Distinct tool paths and how often each occurred.

        More than one path across identical inputs is the clearest evidence
        that a single run would have been a coin flip.
        """
        counts: dict[str, int] = {}
        for attempt in self.attempts:
            key = " → ".join(attempt.run.path) or "(no calls)"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def stat(self, attribute: str) -> tuple[float, float]:
        """Mean and population standard deviation of a numeric run field."""
        values = [float(getattr(a.run, attribute, 0) or 0) for a in self.attempts]
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return mean, var ** 0.5


class Suite:
    def __init__(
        self,
        runner: Runner,
        cases: Sequence[Case],
        *,
        repeats: int = 3,
        workers: int = 1,
        label: str = "",
    ):
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        self.runner = runner
        self.cases = list(cases)
        self.repeats = repeats
        self.workers = max(1, workers)
        self.label = label

    # ----------------------------------------------------------------
    def _attempt(self, case: Case, index: int) -> Attempt:
        started = time.perf_counter()
        try:
            run = self.runner(case.question)
        except Exception as exc:
            # A crashed runner is a failed attempt, not a crashed suite. One
            # bad case should never cost you the other twenty.
            elapsed = int((time.perf_counter() - started) * 1000)
            return Attempt(index, Run(latency_ms=elapsed, stopped_reason="runner_error"),
                           [], error=f"{type(exc).__name__}: {exc}")
        results = [check(run) for check in case.checks]
        return Attempt(index, run, results)

    def run(self, *, on_attempt=None) -> "Report":
        from agent_evals.report import Report

        jobs = [(case, i) for case in self.cases for i in range(self.repeats)]
        outcomes: dict[str, list[Attempt]] = {c.id: [] for c in self.cases}

        if self.workers == 1:
            for case, i in jobs:
                attempt = self._attempt(case, i)
                outcomes[case.id].append(attempt)
                if on_attempt:
                    on_attempt(case, attempt)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._attempt, c, i): c for c, i in jobs}
                for future, case in futures.items():
                    attempt = future.result()
                    outcomes[case.id].append(attempt)
                    if on_attempt:
                        on_attempt(case, attempt)

        for attempts in outcomes.values():
            attempts.sort(key=lambda a: a.index)

        return Report(
            label=self.label,
            repeats=self.repeats,
            outcomes=[CaseOutcome(c, outcomes[c.id]) for c in self.cases],
        )
