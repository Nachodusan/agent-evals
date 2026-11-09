# Evidence

The reports behind the numbers in the top-level README. Produced by:

```bash
python examples/eval_sql_agent.py --repeats 3 --workers 3 \
    --compare meta-llama/Meta-Llama-3.1-8B-Instruct
```

against [sql-agent](https://github.com/Nachodusan/sql-agent) on its own golden
set, three runs per case.

| file | what it is |
|---|---|
| `DeepSeek-V3.before.json` | before the two sql-agent fixes |
| `Meta-Llama-3.1-8B-Instruct.before.json` | same — 6 of 24 attempts crashed inside the agent |
| `DeepSeek-V3.json` | after |
| `Meta-Llama-3.1-8B-Instruct.json` | after — 0 crashed attempts, same 8% pass rate |

The pass rate not moving is the finding: the fix corrected which column six
failures belonged in, not the model's performance.
