# Reproducibility Toolkit

This directory contains utilities that make experiment runs reproducible:

- `env_capture.py`: capture Python, platform, and library versions at the start of every run.
- `README.md`: guidance on locking dependencies, capturing environment details, and seeding best practices.

## Recommended Workflow

1. **Lock dependencies** – use `poetry lock` or `pip freeze > requirements.lock`. Commit the lock file.
2. **Record the environment** – call `write_environment_snapshot()` at the start of each run. The experiment runner writes `env.json` automatically.
3. **Fix seeds** – use `experiments.seed.fix_seeds(SEED)` and propagate the seed through configs. Document all seeds.
4. **Capture hardware** – note CPU, GPU model, and RAM. The env capture utility logs what Python can observe.
5. **Archive runs** – persist the entire `runs/<run_id>` directory, including raw outputs and config.

Following these steps enables reproducible, auditable experiments without external services.
