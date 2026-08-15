# Note on GPQA data

GPQA is a gated dataset and its authors ask that questions not be posted in
plain text on the public internet, to avoid contaminating future models.

For that reason the GPQA result files in this repository contain **only the
metrics** — per-problem prediction, reference answer, correctness and truncation
flag. Question statements and full reasoning traces have been removed.

Everything needed to audit the reported accuracy is present. To regenerate the
traces, request access to the dataset from its authors and re-run
`best-of-n/src/eval/run_benchmark.py` with `BENCHMARK=gpqa`.

AIME and GSM8K traces are published in full: their problems are publicly
available and their licences permit redistribution.
