# agent-evals

[![tests](https://github.com/Nachodusan/agent-evals/actions/workflows/tests.yml/badge.svg)](https://github.com/Nachodusan/agent-evals/actions/workflows/tests.yml)

Evaluate what an agent **did**, not just what it answered.

Trajectory checks, repeated runs, and a verdict that refuses to round. No
dependencies in the core.

```python
from agent_evals import Case, Suite, checks as c

suite = Suite(
    runner=my_runner,                       # question -> Run
    cases=[
        Case(
            id="top_customers",
            question="Who are the top 5 customers by net spend?",
            checks=[
                c.answer(lambda a: a is not None),
                c.calls_before("describe_table", "run_sql"),
                c.no_repeated_calls(),
                c.max_steps(12),
            ],
        ),
    ],
    repeats=5,
)
print(suite.run().render())
```

---

## The one opinion

**A case that passed once has not passed.**

Tool-calling models are non-deterministic even at temperature 0. Different
tool order, different arguments, sometimes a different answer. A harness that
runs each case once reports a number that will not reproduce, and the first
time it disagrees with itself, people stop trusting the harness rather than the
agent.

So every case runs `repeats` times, and a case passes only if **every** attempt
passed. Anything in between is `flaky` — its own category, never rounded in
either direction:

```
  ~ count_orders              1/3   5.3±1.9  steps  $0.0008
      reported SQL was actually run  — failed 2/3
      2 different tool paths across runs
```

Two numbers are reported, and the gap between them is the point:

| | |
|---|---|
| **strict** | share of cases that passed every attempt |
| **per-attempt** | share of individual attempts that passed |

A single-run harness reports something close to the second one and calls it
accuracy.

## What it found

The first thing this library was pointed at was
[sql-agent](https://github.com/Nachodusan/sql-agent), which already had its own
single-run evaluation harness reporting **6/8 (75%)**.

Three runs per case, with trajectory checks:

| | DeepSeek-V3 | Llama-3.1-8B |
|---|---|---|
| strict (passed all 3) | **4/8 — 50%** | 0/8 — 0% |
| per-attempt | 54% | 8% |
| flaky | 1 | 1 |
| cost, full suite | $0.0403 | $0.0172 |

Three findings, none of which a single run could produce.

### 1. One of the passing cases was a coin flip

`refund_rate` passed the single-run harness. Across three attempts it passed
**zero**, then **one** on a later run. That ✓ was luck.

### 2. The agent reported SQL it never executed

Asked how many orders there were, the agent called `list_tables` — which
returns row counts — read the answer off that, and then typed a
`SELECT COUNT(*) FROM orders` into its final answer that it **never ran**.

The outcome grader scored it correct, because executing that query does give
the right number. The answer was right and the evidence was fabricated.

That is what motivated the provenance check in
[`examples/eval_sql_agent.py`](examples/eval_sql_agent.py): *is the SQL the
agent reported one it actually ran?* It fails on exactly the runs where the
agent took the shortcut, and no outcome-only grader can see it.

Same question, same model, same temperature, three runs:

```
run 0:  list_tables → finish                              ✗ fabricated SQL
run 1:  list_tables → describe_table → run_sql → finish   ✓
run 2:  list_tables → finish                              ✗ fabricated SQL
```

### 3. A third of the "model failures" were bugs in the agent

Llama-3.1-8B scored 0/8, which reads like a verdict on the model. Opening the
per-attempt errors, **6 of its 24 attempts had not failed — they had crashed**,
inside sql-agent, with an `AttributeError`. The model was sending `arguments`
as a JSON array where the schema asked for an object, and the agent called
`.get()` on the parsed list.

After fixing that in sql-agent:

| | before | after |
|---|---|---|
| crashed attempts | 6 / 24 | **0 / 24** |
| per-attempt pass rate | 8% | 8% |

The pass rate did not move. The fix did not make the model better — it moved
six failures from the model's column to the agent's, where they belonged. A
harness that only reported a score would have left that misattribution in
place.

## Checks

Outcome checks answer *was it right*. These answer *how did it get there*,
which is what decides whether an agent is affordable, safe and worth keeping
once it is right.

| Check | Asserts |
|---|---|
| `calls(tool, at_least=, at_most=)` | a tool was used, within a count range |
| `never_calls(tool)` | a tool was not touched — distractors, side effects |
| `calls_before(a, b)` | ordering, e.g. read the schema before querying it |
| `no_failures()` | nothing errored |
| `recovers()` | no failure was left unrecovered |
| `finished()` | ended because it was done, not because it ran out |
| `max_steps(n)` / `max_cost(usd)` | budget |
| `no_repeated_calls()` | never made the identical call twice |
| `answer(predicate)` | your outcome check |

Two deliberate omissions:

**`recovers()` does not require that anything failed.** The property is
"failure is not fatal", not "please fail". A clean run passes.

**There is no built-in correctness check.** Comparing SQL result sets, grading
prose and matching a number are different problems, and a generic *is it right*
helper would be wrong for all three. You pass a predicate.

## Reports

```python
report = suite.run()
print(report.render())
report.save("reports/deepseek.json")
```

Per case: the pass fraction, which checks failed and how often, mean ± standard
deviation of steps and cost, and **how many distinct tool paths occurred across
runs**. More than one path on identical input is the clearest evidence that a
single run would have been a coin flip.

Comparing two models, or the same agent before and after a change:

```python
from agent_evals import compare
print(compare(before, after))
```

```
  case                         DeepSeek-V3  Meta-Llama-3.1-8B-Instruct

  count_orders              ✗ 0/3           ✗ 0/3
↓ revenue_by_category       ✓ 3/3           ✗ 0/3
↓ avg_order_value           ✓ 3/3           ✗ 0/3
  refund_rate               ~ 1/3           ✗ 0/3

  strict                50%              0%
  cost              $0.0403         $0.0172
  regressions: revenue_by_category, avg_order_value, …
```

## Adapting your agent

The protocol is deliberately small — any agent that can say which tools it
called, in what order, which failed, and what it answered:

```python
from agent_evals import Run, ToolCall

def my_runner(question: str) -> Run:
    result = my_agent.run(question)
    return Run(
        calls=[ToolCall(c.name, ok=c.ok, arguments=c.args) for c in result.calls],
        answer=result.answer,
        steps=result.step_count,
        cost_usd=result.cost,
        stopped_reason=result.reason,
    )
```

The [sql-agent adapter](agent_evals/adapters/sql_agent.py) is thirty lines, and
that is the whole integration. If an adapter needs much more, the protocol is
asking for the wrong things.

A crashing runner fails one attempt, not the suite — one bad case should never
cost you the other twenty.

## Install

```bash
pip install -e .          # core: no dependencies
pip install -e ".[dev]"   # plus pytest
```

Python ≥ 3.11.

Running the sql-agent example additionally needs that project checked out
beside this one (or `SQL_AGENT_PATH` set), plus its own dependencies:

```bash
python examples/eval_sql_agent.py --repeats 3 --workers 3
python examples/eval_sql_agent.py --compare meta-llama/Meta-Llama-3.1-8B-Instruct
```

`workers` runs attempts concurrently; results are collected in a deterministic
order regardless.

## Tests

```bash
pytest
```

Thirty tests, no network, 0.03s. The central one is
`test_a_case_that_passes_sometimes_is_flaky_not_passing` — if that ever starts
rounding, the project has lost its point.

## What this is not

- **Not a benchmark.** It brings no tasks and no datasets. You supply the
  cases; it tells you how consistently your agent handles them.
- **Not a tracing backend.** It does not collect spans or run a server. Point
  it at an agent you already have and it reads the trace the agent already
  produces.
- **Not a correctness oracle.** Whether an answer is right is your predicate,
  because only you know what right means for your task.

## Licence

[MIT](LICENSE).
