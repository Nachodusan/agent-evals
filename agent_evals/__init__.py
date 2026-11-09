"""agent-evals — evaluate what an agent did, not just what it answered.

    from agent_evals import Case, Suite, checks as c

    suite = Suite(
        runner=my_runner,                  # question -> Run
        cases=[
            Case(
                id="top_customers",
                question="Who are the top 5 customers by net spend?",
                checks=[
                    c.calls_before("describe_table", "run_sql"),
                    c.no_repeated_calls(),
                    c.max_steps(12),
                    c.answer(lambda a: a is not None),
                ],
            ),
        ],
        repeats=5,
    )
    print(suite.run().render())

The core has no dependencies. Adapters for specific agents live in
`agent_evals.adapters` and are imported only if you use them.
"""

from agent_evals import checks
from agent_evals.report import Report, compare
from agent_evals.suite import Attempt, Case, CaseOutcome, Suite
from agent_evals.types import Check, CheckResult, Run, Runner, ToolCall

__version__ = "0.1.0"

__all__ = [
    "Attempt", "Case", "CaseOutcome", "Check", "CheckResult",
    "Report", "Run", "Runner", "Suite", "ToolCall",
    "checks", "compare",
]
