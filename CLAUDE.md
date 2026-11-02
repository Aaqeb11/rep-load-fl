# Working rules for this repo

- Never commit or push to `main`. Work on a feature branch (e.g. `step2-attacks`) and open a pull request into `main`.
- Keep code simple and readable: small functions, plain loops, comments explaining *why*. Avoid clever one-liners.
- Selection, scoring and aggregation stay as separate, swappable functions in `src/bcfl/selection.py`, `scoring.py` and `aggregation.py`, registered in their dictionaries and chosen through the YAML config.
- Every experiment is reproducible from its YAML config plus `--set` overrides; results go under `results/` (not committed).
- Run `pytest` and the smoke config (`python scripts/run.py --config configs/smoke.yaml`) before opening a PR.
- Research context (research questions, novelty, plan): `docs/PROJECT_CONTEXT.md`.
