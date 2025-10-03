# Experiment Guide

## Designing Fair Comparisons
- Keep dataset splits identical across systems. Use the `num_queries` and `random_query_sample` config options for consistent subsets.
- Always record the exact configuration. The runner writes `config.used.yaml` for you.
- Run at least three seeds for headline numbers. Inspect the `metrics.json` mean ± std output to understand variance.

## Seed Handling
- Use `experiments.seed.fix_seeds` at the start of every pipeline run.
- Explicitly set the `seed` value in your config files and propagate it to downstream components (retriever initialisation, samplers, etc.).
- When comparing runs, re-use the same seed list so bootstrap tests remain paired.

## Significance Testing
- Use `evaluation.significance.paired_bootstrap` with per-query scores to obtain 95% confidence intervals.
- Treat p-values below 0.05 as statistically significant improvements.
- Pair samples by query ID to control for dataset variance.

## Scaling Within a Team
- Store the `runs/` directory on a shared NAS. Each run folder is self-contained.
- The tracker logs metrics to CSV and JSONL, which can be ingested into lightweight BI tools.
- Version control the `experiments/configs/` directory so everyone can reproduce past studies.

## Interpreting Reports
- The generated report references markdown tables (`tables/*.md`) and plots (`plots/*.png`).
- Use the `ragx-report` CLI to rebuild the report after editing annotations or adding failure cases.
- The reproducibility checklist is auto-populated from `env.json` and the resolved config.
