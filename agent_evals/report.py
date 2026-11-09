"""Reporting, and comparing two reports.

The rendering has one job beyond being readable: make flakiness impossible to
skim past. A suite where every case passes three times out of three and a
suite where half of them pass two out of three are very different systems, and
a single accuracy percentage renders them identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_evals.suite import CaseOutcome

MARK = {"pass": "✓", "fail": "✗", "flaky": "~"}


@dataclass
class Report:
    outcomes: list[CaseOutcome]
    repeats: int = 1
    label: str = ""

    # -- aggregates ----------------------------------------------------
    @property
    def counts(self) -> dict[str, int]:
        out = {"pass": 0, "flaky": 0, "fail": 0}
        for o in self.outcomes:
            out[o.verdict] += 1
        return out

    @property
    def strict_pass_rate(self) -> float:
        """Share of cases that passed *every* attempt."""
        return self.counts["pass"] / len(self.outcomes) if self.outcomes else 0.0

    @property
    def attempt_pass_rate(self) -> float:
        """Share of individual attempts that passed.

        Always ≥ the strict rate, and the gap between the two is the
        flakiness of the system. A single-run harness reports something close
        to this number and calls it accuracy.
        """
        total = sum(o.total for o in self.outcomes)
        passed = sum(o.passes for o in self.outcomes)
        return passed / total if total else 0.0

    @property
    def total_cost_usd(self) -> float:
        return round(sum(a.run.cost_usd for o in self.outcomes for a in o.attempts), 6)

    @property
    def flaky_cases(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if o.verdict == "flaky"]

    # -- rendering -----------------------------------------------------
    def render(self) -> str:
        lines: list[str] = []
        head = f"{self.label} · " if self.label else ""
        lines.append(f"{head}{len(self.outcomes)} cases × {self.repeats} runs")
        lines.append("")

        width = max((len(o.case.id) for o in self.outcomes), default=10)
        for o in self.outcomes:
            steps, steps_sd = o.stat("steps")
            cost, _ = o.stat("cost_usd")
            lines.append(
                f"  {MARK[o.verdict]} {o.case.id:<{width}}  "
                f"{o.passes}/{o.total}  "
                f"{steps:>4.1f}±{steps_sd:<4.1f} steps  ${cost:.4f}"
            )
            if o.verdict != "pass":
                for name, n in o.check_failures.items():
                    lines.append(f"      {name}  — failed {n}/{o.total}")
            if len(o.distinct_paths) > 1:
                lines.append(f"      {len(o.distinct_paths)} different tool paths across runs")

        c = self.counts
        lines += [
            "",
            f"  {c['pass']} pass · {c['flaky']} flaky · {c['fail']} fail",
            f"  strict {self.strict_pass_rate:.0%} (every run)   "
            f"per-attempt {self.attempt_pass_rate:.0%}   "
            f"${self.total_cost_usd:.4f}",
        ]
        if c["flaky"]:
            names = ", ".join(o.case.id for o in self.flaky_cases)
            lines.append(f"  flaky: {names}")
        return "\n".join(lines)

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "repeats": self.repeats,
            "summary": {
                **self.counts,
                "strict_pass_rate": round(self.strict_pass_rate, 4),
                "attempt_pass_rate": round(self.attempt_pass_rate, 4),
                "total_cost_usd": self.total_cost_usd,
            },
            "cases": [
                {
                    "id": o.case.id,
                    "question": o.case.question,
                    "tags": list(o.case.tags),
                    "verdict": o.verdict,
                    "passes": o.passes,
                    "total": o.total,
                    "check_failures": o.check_failures,
                    "distinct_paths": o.distinct_paths,
                    "steps_mean": round(o.stat("steps")[0], 2),
                    "steps_sd": round(o.stat("steps")[1], 2),
                    "cost_mean_usd": round(o.stat("cost_usd")[0], 6),
                    "latency_mean_ms": round(o.stat("latency_ms")[0], 1),
                    "attempts": [
                        {
                            "index": a.index,
                            "passed": a.passed,
                            "path": a.run.path,
                            "steps": a.run.steps,
                            "cost_usd": a.run.cost_usd,
                            "stopped_reason": a.run.stopped_reason,
                            "failed_checks": a.failed_checks,
                            "error": a.error,
                        }
                        for a in o.attempts
                    ],
                }
                for o in self.outcomes
            ],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def compare(a: Report, b: Report) -> str:
    """Two reports side by side, case by case.

    Built for the two questions you actually ask: did this model change break
    anything, and is the cheaper one good enough.
    """
    by_id_b = {o.case.id: o for o in b.outcomes}
    width = max((len(o.case.id) for o in a.outcomes), default=10)
    la, lb = a.label or "A", b.label or "B"

    lines = [f"  {'case':<{width}}  {la:>14}  {lb:>14}", ""]
    regressions: list[str] = []
    improvements: list[str] = []

    for oa in a.outcomes:
        ob = by_id_b.get(oa.case.id)
        if ob is None:
            lines.append(f"  {oa.case.id:<{width}}  {oa.passes}/{oa.total:<12}  {'—':>14}")
            continue
        flag = " "
        if oa.verdict == "pass" and ob.verdict != "pass":
            flag, _ = "↓", regressions.append(oa.case.id)
        elif oa.verdict != "pass" and ob.verdict == "pass":
            flag, _ = "↑", improvements.append(oa.case.id)
        lines.append(
            f"{flag} {oa.case.id:<{width}}  "
            f"{MARK[oa.verdict]} {oa.passes}/{oa.total:<10}  "
            f"{MARK[ob.verdict]} {ob.passes}/{ob.total:<10}"
        )

    lines += [
        "",
        f"  strict      {a.strict_pass_rate:>13.0%}  {b.strict_pass_rate:>14.0%}",
        f"  cost        {'$%.4f' % a.total_cost_usd:>13}  {'$%.4f' % b.total_cost_usd:>14}",
    ]
    if regressions:
        lines.append(f"  regressions: {', '.join(regressions)}")
    if improvements:
        lines.append(f"  improvements: {', '.join(improvements)}")
    return "\n".join(lines)
