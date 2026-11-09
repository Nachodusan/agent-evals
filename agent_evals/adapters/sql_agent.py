"""Adapter for https://github.com/Nachodusan/sql-agent.

Thirty lines, and it is the whole integration. If an adapter for your agent
needs much more than this, the `Run` protocol is probably asking for the wrong
things.
"""

from __future__ import annotations

from typing import Any

from agent_evals.types import Run, ToolCall


def to_run(trace: Any) -> Run:
    """Convert a sql_agent `Trace` into a `Run`."""
    calls = [
        ToolCall(
            name=step.name,
            ok=step.ok,
            arguments=dict(step.arguments or {}),
            error=step.error,
        )
        for step in trace.steps
        if step.kind == "tool"
    ]
    return Run(
        calls=calls,
        answer=trace.answer,
        steps=len(trace.steps),
        cost_usd=trace.cost_usd,
        latency_ms=trace.latency_ms,
        stopped_reason=trace.stopped_reason,
        extra={"final_sql": trace.final_sql, "run_id": trace.run_id},
    )


def runner(**kwargs: Any):
    """Build a `Runner` bound to a database and model.

        runner(db_path="shop.db", model="deepseek-ai/DeepSeek-V3")
    """
    from sql_agent.agent import ask

    def run(question: str) -> Run:
        return to_run(ask(question, **kwargs))

    return run
