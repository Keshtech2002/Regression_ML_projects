# Reproducible ML experiment automation — offline demo

**Original educational prototype. No LLM, agent, API provider or GPU is used.**

The pipeline checks a synthetic regression dataset, runs six gradient-boosting configurations, records validation scores, selects parameters based on validation, refits, and produces a **separate held-out test** metric plus JSON/CSV reports.

## Run

```bash
pip install -r requirements.txt
python experiment_runner.py --output outputs
pytest -q
```

The synthetic data and its scores are for software demonstration, not domain validation.

## Extensions for FortyGuard-style use cases
- Use spatial/time-blocked splits on weather observations and provide per-region/per-hour metrics.
- Integrate MLflow, xarray, dataset provenance, and automated schema/data quality checks.
- Add an optional **LLM agent** for suggesting a bounded experiment configuration; require validation and human approval before execution, and compare performance against the non-agent baseline.
- Add retries, time/cost budgets, and alerting for failures.

## Credit
Original code; open-source inspiration for more advanced agentic research: [a0merr/autolab](https://github.com/a0merr/autolab). No code was copied.
