"""Evaluate github.com/Nachodusan/sql-agent with this library.

Reuses that project's own golden questions and reference SQL, and adds what a
single-run harness cannot report: how often each case passes across repeated
runs, and how much the trajectory varies between them.

    python examples/eval_sql_agent.py --repeats 3 --workers 3
    python examples/eval_sql_agent.py --compare meta-llama/Meta-Llama-3.1-8B-Instruct

Point SQL_AGENT_PATH at your checkout if it is not a sibling directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SQL_AGENT = Path(os.getenv("SQL_AGENT_PATH", HERE.parents[1] / "sql-agent"))
if not (SQL_AGENT / "sql_agent").is_dir():
    sys.exit(f"sql-agent not found at {SQL_AGENT}. Set SQL_AGENT_PATH.")
sys.path.insert(0, str(SQL_AGENT))
sys.path.insert(0, str(SQL_AGENT / "evals"))

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from agent_evals import Case, Suite, checks as c, compare  # noqa: E402
from agent_evals.adapters.sql_agent import to_run  # noqa: E402

load_dotenv(SQL_AGENT / ".env")

from run_eval import matches, normalise  # noqa: E402
from sql_agent import db  # noqa: E402
from sql_agent.agent import ask  # noqa: E402

DB = db.DEFAULT_DB_PATH if Path(db.DEFAULT_DB_PATH).exists() else db.build()


def expected_rows(reference_sql: str) -> list[tuple]:
    """Run the hand-written reference query once, up front."""
    conn = db.connect(DB)
    rows = normalise(conn.execute(reference_sql).fetchall())
    conn.close()
    return rows


def _canonical(sql: str) -> str:
    return " ".join((sql or "").split()).rstrip(";").strip().lower()


def sql_was_executed(run) -> bool:
    """Is the SQL the agent reported one it actually ran?

    This check exists because of a real failure. Asked how many orders there
    were, the agent called `list_tables` — which returns row counts — read the
    answer off that, and then reported a `SELECT COUNT(*) FROM orders` it had
    never executed. The outcome grader scored it correct, because running that
    query does give the right number.

    The answer was right and the evidence was fabricated. Only the trajectory
    shows the difference.
    """
    reported = _canonical((run.extra or {}).get("final_sql") or "")
    if not reported:
        return False
    executed = {
        _canonical(call.arguments.get("query", ""))
        for call in run.calls
        if call.name == "run_sql" and call.ok
    }
    return reported in executed


def build_cases() -> list[Case]:
    spec = yaml.safe_load((SQL_AGENT / "evals" / "golden.yaml").read_text(encoding="utf-8"))
    cases: list[Case] = []

    for item in spec:
        expected = expected_rows(item["reference_sql"])
        ordered = bool(item.get("ordered"))

        def correct(run, expected=expected, ordered=ordered) -> bool:
            sql = (run.extra or {}).get("final_sql")
            if not sql:
                return False
            try:
                conn = db.connect(DB)
                actual = normalise(conn.execute(sql).fetchall())
                conn.close()
            except Exception:
                return False
            return matches(expected, actual, ordered=ordered)

        cases.append(
            Case(
                id=item["id"],
                question=" ".join(item["question"].split()),
                tags=tuple(item.get("traps", [])),
                checks=[
                    # Outcome.
                    _RunCheck("answer matches reference rows", correct),
                    # Provenance. The reason this one exists is in the README:
                    # the agent will happily report SQL it never executed.
                    _RunCheck("reported SQL was actually run", sql_was_executed),
                    # Trajectory: the part a single pass/fail hides.
                    c.calls_before("describe_table", "run_sql"),
                    c.no_repeated_calls(),
                    c.finished(),
                    c.max_steps(14),
                ],
            )
        )
    return cases


class _RunCheck:
    """A check that needs the whole run, not just the answer."""

    def __init__(self, name, fn):
        self.name = name
        self._fn = fn

    def __call__(self, run):
        from agent_evals.types import CheckResult
        try:
            ok = bool(self._fn(run))
        except Exception as exc:
            return CheckResult(self.name, False, f"{type(exc).__name__}: {exc}")
        return CheckResult(self.name, ok, "" if ok else "rows differ from reference")


def runner_for(model: str | None):
    def run(question: str):
        return to_run(ask(question, db_path=DB, model=model))
    return run


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--model", default=None, help="Defaults to SQL_AGENT_MODEL.")
    p.add_argument("--compare", default=None, help="A second model to compare against.")
    p.add_argument("--out", default="reports", help="Directory for JSON reports.")
    args = p.parse_args()

    cases = build_cases()
    primary = args.model or os.getenv("SQL_AGENT_MODEL", "default")

    print(f"running {len(cases)} cases × {args.repeats} on {primary}\n")
    report = Suite(runner_for(args.model), cases, repeats=args.repeats,
                   workers=args.workers, label=primary.split("/")[-1]).run()
    print(report.render())
    report.save(Path(args.out) / f"{primary.split('/')[-1]}.json")

    if args.compare:
        print(f"\nrunning the same set on {args.compare}\n")
        other = Suite(runner_for(args.compare), cases, repeats=args.repeats,
                      workers=args.workers, label=args.compare.split("/")[-1]).run()
        print(other.render())
        other.save(Path(args.out) / f"{args.compare.split('/')[-1]}.json")
        print("\n" + compare(report, other))

    return 0


if __name__ == "__main__":
    sys.exit(main())
