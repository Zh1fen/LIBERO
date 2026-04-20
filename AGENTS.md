# Repository Guidelines

## Project Structure & Module Organization
- `libero/libero/`: core benchmark package (environments, assets, BDDL files, task definitions, utilities).
- `libero/lifelong/`: training and evaluation pipeline (algorithms, policies, datasets, metrics).
- `libero/configs/`: Hydra config tree for training/eval/policy/lifelong settings.
- `scripts/`: utility scripts for dataset creation, task templates, and data inspection.
- `benchmark_scripts/`: benchmark helpers (dataset download, checks, rendering).
- `notebooks/`: walkthroughs and custom object examples.
- `templates/` and `images/`: generation templates and documentation assets.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies.
- `pip install -e .`: install LIBERO in editable mode (recommended for development).
- `python benchmark_scripts/download_libero_datasets.py --use-huggingface`: fetch demonstration datasets.
- `python libero/lifelong/main.py benchmark_name=LIBERO_10 policy=bc_rnn_policy lifelong=base seed=0`: run a training experiment.
- `python libero/lifelong/evaluate.py --benchmark LIBERO_10 --task_id 0 --algo base --policy bc_rnn_policy --seed 0 --ep 50 --load_task 9 --device_id 0`: evaluate a checkpoint.
- `python benchmark_scripts/check_task_suites.py`: quick benchmark/task-suite sanity check.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and PEP 8 style; keep functions/modules in `snake_case`.
- Class names use `PascalCase`; constants use `UPPER_SNAKE_CASE`.
- Follow existing Hydra config naming patterns (lowercase YAML files such as `bc_rnn_policy.yaml`).
- Keep changes local to the relevant subsystem (`libero/libero` vs. `libero/lifelong`) and avoid cross-module side effects.

## Testing Guidelines
- There is no dedicated `tests/` directory; validation is script-based.
- For functional checks, run targeted scripts from `benchmark_scripts/` and a short `lifelong/main.py` run on a small benchmark.
- For dataset-related changes, run `scripts/check_dataset_integrity.py` before submitting.

## Commit & Pull Request Guidelines
- Commit messages in this repo are short, imperative, and direct (for example, `Add support for dataset download from huggingface`).
- Keep commits focused by topic (dataset tooling, policy code, configs, docs).
- PRs should include: purpose, changed paths, run commands for validation, and any dataset/config assumptions.
- Link related issues when available and include logs/screenshots for training or evaluation behavior changes.
