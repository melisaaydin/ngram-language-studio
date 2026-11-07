# Evaluation artifacts

- `training_results.json`: per-model perplexities used in the Gradio dashboard.
- `tuning_results.json`: validation-only backoff search and final held-out scores.

`prepared_data.json` is intentionally not tracked; generate it with `python prepare_data.py` (requires internet for an uncached dataset).
