"""Tests for repetition, verdicts and the flaky category.

The central assertion of this library is in `test_a_case_that_passes_sometimes
_is_flaky_not_passing`. If that ever starts rounding, the project has lost its
point.
"""

from __future__ import annotations

import itertools

import pytest

from agent_evals import Case, Suite, checks as c, compare
from agent_evals.types import Run, ToolCall

OK = lambda n, **a: ToolCall(n, True, a)          # noqa: E731
BAD = lambda n, **a: ToolCall(n, False, a, "boom")  # noqa: E731


def constant_runner(run: Run):
    return lambda question: run


def cycling_runner(runs):
    """Returns each run in turn, cycling — a deterministic stand-in for a
    non-deterministic agent."""
    it = itertools.cycle(runs)
    return lambda question: next(it)


GOOD = Run(calls=[OK("run_sql")], answer=42, steps=3, cost_usd=0.001, stopped_reason="finished")
WRONG = Run(calls=[OK("run_sql")], answer=0, steps=3, cost_usd=0.001, stopped_reason="finished")


def case(checks=None) -> Case:
    return Case(id="c1", question="q", checks=checks or [c.answer(lambda a: a == 42)])


# --- verdicts -------------------------------------------------------------

def test_all_attempts_passing_is_a_pass():
    report = Suite(constant_runner(GOOD), [case()], repeats=4).run()
    o = report.outcomes[0]
    assert o.verdict == "pass"
    assert (o.passes, o.total) == (4, 4)
    assert report.strict_pass_rate == 1.0


def test_no_attempt_passing_is_a_fail():
    report = Suite(constant_runner(WRONG), [case()], repeats=4).run()
    assert report.outcomes[0].verdict == "fail"
    assert report.strict_pass_rate == 0.0


def test_a_case_that_passes_sometimes_is_flaky_not_passing():
    """The whole argument of the library, in one assertion."""
    report = Suite(cycling_runner([GOOD, GOOD, WRONG]), [case()], repeats=3).run()
    o = report.outcomes[0]
    assert o.verdict == "flaky"
    assert o.passes == 2 and o.total == 3
    # Strict counts it as not passing; per-attempt would have called it 67%.
    assert report.strict_pass_rate == 0.0
    assert report.attempt_pass_rate == pytest.approx(2 / 3)
    assert report.counts == {"pass": 0, "flaky": 1, "fail": 0}


def test_the_gap_between_the_two_rates_is_the_flakiness():
    report = Suite(cycling_runner([GOOD, WRONG]), [case()], repeats=2).run()
    assert report.strict_pass_rate < report.attempt_pass_rate


# --- diagnostics ----------------------------------------------------------

def test_check_failures_are_counted_per_check():
    checks = [c.answer(lambda a: a == 42), c.max_steps(1)]
    report = Suite(constant_runner(GOOD), [case(checks)], repeats=3).run()
    o = report.outcomes[0]
    assert o.check_failures["at most 1 steps"] == 3
    assert "answer is correct" not in o.check_failures


def test_distinct_paths_expose_non_determinism():
    a = Run(calls=[OK("list_tables"), OK("run_sql")], answer=42, stopped_reason="finished")
    b = Run(calls=[OK("run_sql")], answer=42, stopped_reason="finished")
    report = Suite(cycling_runner([a, b]), [case()], repeats=4).run()
    paths = report.outcomes[0].distinct_paths
    assert len(paths) == 2
    assert sum(paths.values()) == 4


def test_failed_calls_are_marked_in_the_path():
    r = Run(calls=[BAD("run_sql"), OK("run_sql")], answer=42, stopped_reason="finished")
    report = Suite(constant_runner(r), [case()], repeats=1).run()
    assert report.outcomes[0].attempts[0].run.path == ["run_sql!", "run_sql"]


def test_stat_returns_mean_and_spread():
    a = Run(calls=[OK("x")], answer=42, steps=2)
    b = Run(calls=[OK("x")], answer=42, steps=8)
    report = Suite(cycling_runner([a, b]), [case()], repeats=2).run()
    mean, sd = report.outcomes[0].stat("steps")
    assert mean == 5.0
    assert sd == 3.0


# --- robustness -----------------------------------------------------------

def test_a_crashing_runner_fails_one_attempt_not_the_suite():
    def boom(question):
        raise RuntimeError("provider timeout")

    report = Suite(boom, [case()], repeats=2).run()
    o = report.outcomes[0]
    assert o.verdict == "fail"
    assert all(a.error and "provider timeout" in a.error for a in o.attempts)
    assert o.attempts[0].run.stopped_reason == "runner_error"


def test_repeats_must_be_positive():
    with pytest.raises(ValueError):
        Suite(constant_runner(GOOD), [case()], repeats=0)


def test_workers_do_not_change_the_result():
    cases = [Case(id=f"c{i}", question="q", checks=[c.answer(lambda a: a == 42)])
             for i in range(4)]
    serial = Suite(constant_runner(GOOD), cases, repeats=3, workers=1).run()
    parallel = Suite(constant_runner(GOOD), cases, repeats=3, workers=4).run()
    assert serial.counts == parallel.counts
    assert [o.case.id for o in serial.outcomes] == [o.case.id for o in parallel.outcomes]
    assert all(len(o.attempts) == 3 for o in parallel.outcomes)


# --- report ---------------------------------------------------------------

def test_render_shows_the_fraction_and_flags_flaky():
    report = Suite(cycling_runner([GOOD, WRONG, GOOD]), [case()], repeats=3).run()
    text = report.render()
    assert "2/3" in text
    assert "flaky" in text


def test_serialisation_round_trips_the_essentials():
    report = Suite(cycling_runner([GOOD, WRONG]), [case()], repeats=2, label="m1").run()
    d = report.to_dict()
    assert d["label"] == "m1"
    assert d["summary"]["flaky"] == 1
    assert len(d["cases"][0]["attempts"]) == 2
    assert d["cases"][0]["verdict"] == "flaky"


def test_compare_flags_regressions_and_improvements():
    good = Suite(constant_runner(GOOD), [case()], repeats=2, label="before").run()
    bad = Suite(constant_runner(WRONG), [case()], repeats=2, label="after").run()
    text = compare(good, bad)
    assert "regressions: c1" in text
    assert "improvements" not in text
    assert "improvements: c1" in compare(bad, good)
