# Configuration Reference (TOML)

This document explains the available options for labML TOML configs.

- Use `examples/prepare.toml` and `examples/demo_prepare.toml` as copy-ready templates for `prepare`.
- Use `examples/benchmark_regression.toml`, `examples/benchmark_classification.toml`, `examples/demo_benchmark_regression.toml`, and `examples/demo_benchmark_regression_parallel.toml` as templates for `benchmark-*`.

All relative paths are resolved from the directory where the TOML file lives.

## 1) Prepare config (`labml prepare --config ...`)

### Required sections

- `[input]`
- `[partition]`
- `[output]`
- `[dataset]` is expected in practice because target/task are validated during prepare.

### `[input]`

- `path` (string, required): input data file path (`.csv`, `.xlsx/.xls`, `.parquet`).
- `sheet` (string, optional): Excel sheet name for `.xlsx/.xls`.

### `[hook]` (optional)

- `path` (string, optional): Python file containing a transform function.
- `function` (string, optional, default `"transform"`): function name.
- `params` (table, optional): custom params passed to the hook.

Hook contract:

- Signature: `transform(df, params) -> df`
- Return value must be a pandas DataFrame.

### `[dataset]`

- `task` (string, optional, default `"regression"`): `"regression"` or `"classification"`.
- `target` (string, required in practice): target column name present in data.

### `[partition]`

- `mode` (string, optional, default `"random"`): `"random"` or `"group"`.
- `n_splits` (int, optional, default `10`).
- `shuffle` (bool, optional, default `true`).
- `random_state` (int or null, optional, default `42`).
- `group_column` (string, required when `mode = "group"`): group column in data.

### `[output]`

- `dir` (string, optional, default `"_artifacts_/prepared/default"`): output directory.

Prepare writes:

- `data.parquet`
- `folds.csv`
- `metadata.json`

## 2) Benchmark config (`labml benchmark-regression/classification --config ...`)

### Required sections

- `[input]`
- `[search]` with child sections:
  - `[search.scale]`
  - `[search.filter]`
  - `[search.reduction]`
  - `[search.model]`
- `[output]`

`[evaluation]` and `[partition]` are optional, with defaults.

### `[input]`

- `source` (string, optional, default `"prepared"`): `"prepared"` or `"external"`.
- `data_path` (string, required): dataset path.
- `target` (string, optional if present in metadata; otherwise required).
- `features` (list[string], optional):
  - if omitted, labML uses all columns except target;
  - if provided, must be a non-empty list.
- `folds_path` (string, optional): use precomputed folds.
- `metadata_path` (string, optional): metadata JSON (used, for example, to infer target).

### `[partition]` (optional when `folds_path` is omitted)

Same keys and defaults as prepare:

- `mode`, `n_splits`, `shuffle`, `random_state`, `group_column`

### `[evaluation]`

- `n_jobs` (int, optional, default `-1`): `-1` or positive integer.
- `metrics` (list[string], optional): scoring metrics.
  - regression default: `neg_root_mean_squared_error`, `r2`
  - classification default: `f1_macro`, `accuracy`
- `primary_metric` (string, optional):
  - regression default: `neg_root_mean_squared_error`
  - classification default: `f1_macro`
  - auto-included into `metrics` if missing.

### `[search.*]` (pipeline search space)

Each subsection supports:

- `enabled` (list[string]): enabled techniques.
- `params` (nested tables, optional): parameter grids by technique.

#### Available techniques

- `search.scale`: `none`, `norm`, `std`
- `search.filter` (regression): `none`, `fscore`, `mi`
- `search.filter` (classification): `none`, `fscore`
- `search.reduction`: `none`, `pca`, `ica`, `nmf`
- `search.model` (regression): `pls`, `sgd`, `svm`, `bag`, `gbr`, `rf`, `mlp`
- `search.model` (classification): `logreg`, `svc`, `rfc`, `bagc`

At least one model must be enabled in `[search.model]`.

### Parameter value syntax (`params`)

labML supports three forms:

- Scalar: `n_estimators = 100`
- List: `n_estimators = [50, 100, 200]`
- Range string (inclusive): `"start:step:stop"`
  - example: `C = "0.1:0.1:0.5"` -> `0.1, 0.2, 0.3, 0.4, 0.5`

You can also combine range strings inside lists.

### `[output]`

- `file` (string, optional): Excel output path (default `benchmark_<task>.xlsx`).
- `latex` (bool, optional, default `false`): generate LaTeX tables.
- `latex_dir` (string, optional): directory for `.tex` files (default: output file directory).

## 3) Error behavior you should expect

- Config/data issues (`config_data`) fail fast before execution output is written.
- Incompatible combinations are marked as `skipped` with `error_type = "incompatible_combo"`.
- Recoverable model failures are marked as `failed` with `error_type = "model_execution"`.
- Unexpected internal errors propagate (`error_type = "internal"`, fail-fast behavior).

## 4) Recommended templates (from `examples/`)

- Prepare baseline: `examples/prepare.toml`
- Prepare demo: `examples/demo_prepare.toml`
- Regression benchmark baseline: `examples/benchmark_regression.toml`
- Classification benchmark baseline: `examples/benchmark_classification.toml`
- Regression demo: `examples/demo_benchmark_regression.toml`
- Regression demo parallel: `examples/demo_benchmark_regression_parallel.toml`

You can copy one of these files and only adjust paths, target/features, and search grids.
