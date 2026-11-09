from __future__ import annotations

from agent_evals import checks as c
from agent_evals.types import Run, ToolCall


def run_with(*calls: ToolCall, **kwargs) -> Run:
    return Run(calls=list(calls), **kwargs)


OK = lambda n, **a: ToolCall(n, True, a)          # noqa: E731
BAD = lambda n, **a: ToolCall(n, False, a, "boom")  # noqa: E731


# --- calls ----------------------------------------------------------------

def test_calls_counts_occurrences():
    r = run_with(OK("run_sql"), OK("run_sql"))
    assert c.calls("run_sql")(r).passed
    assert c.calls("run_sql", at_least=2)(r).passed
    assert not c.calls("run_sql", at_least=3)(r).passed
    assert not c.calls("run_sql", at_most=1)(r).passed
    assert not c.calls("describe_table")(r).passed


def test_never_calls():
    assert c.never_calls("drop_table")(run_with(OK("run_sql"))).passed
    assert not c.never_calls("run_sql")(run_with(OK("run_sql"))).passed


def test_calls_before_enforces_order():
    good = run_with(OK("describe_table"), OK("run_sql"))
    bad = run_with(OK("run_sql"), OK("describe_table"))
    assert c.calls_before("describe_table", "run_sql")(good).passed
    assert not c.calls_before("describe_table", "run_sql")(bad).passed


def test_calls_before_is_vacuously_true_when_the_second_never_happens():
    r = run_with(OK("list_tables"))
    assert c.calls_before("describe_table", "run_sql")(r).passed


def test_calls_before_fails_when_the_prerequisite_is_missing():
    r = run_with(OK("run_sql"))
    result = c.calls_before("describe_table", "run_sql")(r)
    assert not result.passed
    assert "without" in result.detail


# --- quality --------------------------------------------------------------

def test_no_failures():
    assert c.no_failures()(run_with(OK("a"), OK("b"))).passed
    assert not c.no_failures()(run_with(OK("a"), BAD("b"))).passed


def test_recovers_passes_when_a_failure_is_followed_by_success():
    assert c.recovers()(run_with(BAD("run_sql"), OK("run_sql"))).passed


def test_recovers_fails_when_the_run_ends_on_a_failure():
    assert not c.recovers()(run_with(OK("list_tables"), BAD("run_sql"))).passed


def test_recovers_does_not_demand_that_anything_fail():
    """The property is 'failure is not fatal', not 'please fail'."""
    assert c.recovers()(run_with(OK("run_sql"))).passed


def test_finished_reads_the_stop_reason():
    assert c.finished()(run_with(OK("a"), stopped_reason="finished")).passed
    assert not c.finished()(run_with(OK("a"), stopped_reason="max_steps")).passed
    assert not c.finished()(run_with(OK("a"))).passed


# --- budget ---------------------------------------------------------------

def test_max_steps_and_cost():
    r = run_with(OK("a"), steps=9, cost_usd=0.004)
    assert c.max_steps(10)(r).passed
    assert not c.max_steps(8)(r).passed
    assert c.max_cost(0.01)(r).passed
    assert not c.max_cost(0.001)(r).passed


def test_no_repeated_calls_catches_an_identical_repeat():
    spinning = run_with(OK("run_sql", query="SELECT 1"), OK("run_sql", query="SELECT 1"))
    assert not c.no_repeated_calls()(spinning).passed


def test_no_repeated_calls_allows_the_same_tool_with_different_arguments():
    r = run_with(OK("describe_table", table="orders"), OK("describe_table", table="customers"))
    assert c.no_repeated_calls()(r).passed


# --- outcome --------------------------------------------------------------

def test_answer_predicate():
    r = run_with(OK("a"), answer=42)
    assert c.answer(lambda a: a == 42)(r).passed
    assert not c.answer(lambda a: a == 7)(r).passed


def test_a_raising_predicate_fails_the_check_rather_than_the_suite():
    r = run_with(OK("a"), answer=None)
    result = c.answer(lambda a: a.missing_attribute)(r)
    assert not result.passed
    assert "AttributeError" in result.detail


def test_check_names_are_readable():
    assert c.calls("run_sql").name.startswith("calls run_sql")
    assert c.never_calls("x").name == "never calls x"
    assert c.max_steps(5).name == "at most 5 steps"
