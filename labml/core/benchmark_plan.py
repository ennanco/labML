"""Planning and inspection helpers for benchmark workloads."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from .benchmark_context import _expand_search, _prepare_input, _resolve_n_jobs
from .benchmark_engine import _apply_inner_parallelism_guard
from .benchmark_models import BenchmarkPlan, MachineProfile
from .config import read_toml, require_section
from .common_config import resolve_path
from .partitioning import fold_ids_to_splits
from .benchmark_context import _can_run_nmf


MODEL_COST_WEIGHTS: dict[str, float] = {
    "pls": 1.0,
    "sgd": 1.1,
    "svm": 2.2,
    "bag": 1.7,
    "gbr": 2.0,
    "rf": 2.4,
    "mlp": 2.8,
    "logreg": 1.2,
    "svc": 2.3,
    "rfc": 2.5,
    "bagc": 1.8,
}

REDUCTION_COST_WEIGHTS: dict[str, float] = {
    "none": 1.0,
    "pca": 1.15,
    "ica": 1.35,
    "nmf": 1.4,
}

SCALE_COST_WEIGHTS: dict[str, float] = {
    "none": 1.0,
    "norm": 1.02,
    "std": 1.04,
}

FILTER_COST_WEIGHTS: dict[str, float] = {
    "none": 1.0,
    "fscore": 1.05,
    "mi": 1.2,
}


def _parse_meminfo() -> tuple[float, float]:
    """Read Linux memory totals/available from `/proc/meminfo` in GB."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return 0.0, 0.0
    values: dict[str, float] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        tokens = rest.strip().split()
        if not tokens:
            continue
        try:
            values[key] = float(tokens[0])
        except ValueError:
            continue
    total_gb = values.get("MemTotal", 0.0) / (1024 * 1024)
    avail_gb = values.get("MemAvailable", 0.0) / (1024 * 1024)
    return total_gb, avail_gb


def _detect_cpu_model() -> str:
    """Read a human-readable CPU model from `/proc/cpuinfo` when available."""
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return "unknown"
    for line in cpuinfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return "unknown"


def _machine_profile() -> MachineProfile:
    """Build machine profile used by the benchmark inspection estimate."""
    total_gb, avail_gb = _parse_meminfo()
    return MachineProfile(
        cpu_count=max(1, os.cpu_count() or 1),
        cpu_model=_detect_cpu_model(),
        mem_total_gb=total_gb,
        mem_available_gb=avail_gb,
    )


def _parallel_speedup(n_jobs_effective: int) -> float:
    """Estimate non-linear speedup for fold-level parallel execution."""
    if n_jobs_effective <= 1:
        return 1.0
    return 1.0 + 0.75 * ((n_jobs_effective - 1) ** 0.8)


def _seconds_to_hms(total_seconds: float) -> str:
    """Format seconds as HH:MM:SS for user-facing estimates."""
    safe_seconds = max(0, int(round(total_seconds)))
    hours, rem = divmod(safe_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _estimate_combo_weight(scale: str, filt: str, reduction: str, model: str) -> float:
    """Heuristic relative cost for one configured combination."""
    model_weight = MODEL_COST_WEIGHTS.get(model, 1.5)
    reduction_weight = REDUCTION_COST_WEIGHTS.get(reduction, 1.1)
    scale_weight = SCALE_COST_WEIGHTS.get(scale, 1.02)
    filter_weight = FILTER_COST_WEIGHTS.get(filt, 1.03)
    return model_weight * reduction_weight * scale_weight * filter_weight


def _collect_benchmark_plan(
    config_path: Path, task: str
) -> tuple[BenchmarkPlan, MachineProfile]:
    """Collect search-space, machine profile and runtime estimate for inspection."""
    config = read_toml(config_path)
    base_dir = config_path.parent.resolve()
    X, _y, fold_ids, _feature_list, _metadata = _prepare_input(
        config, task=task, base_dir=base_dir
    )
    scales, filters, reductions, models = _expand_search(task, config)

    configured_n_jobs = int(config.get("evaluation", {}).get("n_jobs", -1))
    effective_n_jobs = _resolve_n_jobs(configured_n_jobs)
    all_combinations = [
        (scale, filt, reduction, model)
        for scale in scales
        for filt in filters
        for reduction in reductions
        for model in models
    ]
    total_experiments = len(all_combinations)
    n_folds = len(list(fold_ids_to_splits(fold_ids)))

    skipped = 0
    weighted_sum = 0.0
    for scale, filt, reduction, model in all_combinations:
        (
            guarded_scale,
            guarded_filter,
            guarded_reduction,
            guarded_model,
            _forced_count,
        ) = _apply_inner_parallelism_guard(
            scale, filt, reduction, model, effective_n_jobs
        )
        can_run, _reason = _can_run_nmf(guarded_scale, guarded_reduction, X)
        if not can_run:
            skipped += 1
            continue
        weighted_sum += _estimate_combo_weight(
            guarded_scale.key,
            guarded_filter.key,
            guarded_reduction.key,
            guarded_model.key,
        )

    runnable = total_experiments - skipped
    total_evals = runnable * n_folds

    output_cfg = require_section(config, "output")
    output_file = resolve_path(
        base_dir, str(output_cfg.get("file", f"benchmark_{task}.xlsx"))
    )
    latex_enabled = bool(output_cfg.get("latex", False))
    latex_dir_cfg = output_cfg.get("latex_dir")
    latex_dir = (
        resolve_path(base_dir, str(latex_dir_cfg))
        if latex_dir_cfg
        else output_file.parent
    )

    machine = _machine_profile()
    data_scale = max(0.6, ((len(X) * max(1, X.shape[1])) / 5000) ** 0.5)
    base_seconds_per_eval = 0.03
    if total_evals > 0:
        avg_weight = weighted_sum / total_evals * n_folds
    else:
        avg_weight = 1.0
    parallel_speedup = _parallel_speedup(effective_n_jobs)
    memory_penalty = 1.0
    if machine.mem_available_gb > 0 and machine.mem_available_gb < 2.0:
        memory_penalty = 1.2

    estimated_mid = (
        total_evals
        * base_seconds_per_eval
        * data_scale
        * avg_weight
        * memory_penalty
        / parallel_speedup
    )

    plan = BenchmarkPlan(
        task=task,
        data_rows=len(X),
        data_features=X.shape[1],
        n_folds=n_folds,
        n_variants_scale=len(scales),
        n_variants_filter=len(filters),
        n_variants_reduction=len(reductions),
        n_variants_model=len(models),
        combinations_total=total_experiments,
        combinations_skipped=skipped,
        combinations_runnable=runnable,
        estimated_fold_evaluations=total_evals,
        n_jobs_configured=configured_n_jobs,
        n_jobs_effective=effective_n_jobs,
        inner_parallelism_guard=effective_n_jobs > 1,
        output_file=output_file,
        latex_enabled=latex_enabled,
        latex_dir=latex_dir,
        estimated_seconds_low=estimated_mid * 0.7,
        estimated_seconds_mid=estimated_mid,
        estimated_seconds_high=estimated_mid * 1.6,
    )
    return plan, machine


def _echo_benchmark_plan(
    plan: BenchmarkPlan, machine: MachineProfile, title: str
) -> None:
    """Render a human-readable benchmark plan summary to CLI output."""
    typer.echo(title)
    typer.echo("Search Space")
    typer.echo(
        "  Variants: "
        f"scale={plan.n_variants_scale}, "
        f"filter={plan.n_variants_filter}, "
        f"reduction={plan.n_variants_reduction}, "
        f"model={plan.n_variants_model}"
    )
    typer.echo(f"  Combinations total: {plan.combinations_total}")
    typer.echo(f"  Combinations runnable: {plan.combinations_runnable}")
    typer.echo(f"  Combinations skipped: {plan.combinations_skipped}")
    typer.echo(f"  Folds per combination: {plan.n_folds}")
    typer.echo(f"  Estimated fold evaluations: {plan.estimated_fold_evaluations}")

    typer.echo("Machine Profile")
    typer.echo(f"  CPU logical cores: {machine.cpu_count}")
    typer.echo(f"  CPU model: {machine.cpu_model}")
    typer.echo(f"  Memory total: {machine.mem_total_gb:.1f} GB")
    typer.echo(f"  Memory available: {machine.mem_available_gb:.1f} GB")
    typer.echo(f"  n_jobs configured: {plan.n_jobs_configured}")
    typer.echo(f"  n_jobs effective: {plan.n_jobs_effective}")
    typer.echo(
        "  Inner parallelism guard: "
        + (
            "enabled (estimators without explicit n_jobs run with n_jobs=1)"
            if plan.inner_parallelism_guard
            else "disabled"
        )
    )

    typer.echo("Execution Estimate")
    typer.echo(f"  Low: {_seconds_to_hms(plan.estimated_seconds_low)}")
    typer.echo(f"  Mid: {_seconds_to_hms(plan.estimated_seconds_mid)}")
    typer.echo(f"  High: {_seconds_to_hms(plan.estimated_seconds_high)}")

    typer.echo("Output Targets")
    typer.echo(f"  Excel: {plan.output_file}")
    typer.echo(f"  LaTeX enabled: {plan.latex_enabled}")
    if plan.latex_enabled:
        typer.echo(f"  LaTeX directory: {plan.latex_dir}")


def inspect_benchmark(config_path: Path, task: str) -> None:
    """CLI helper used by `inspect-config` command."""
    plan, machine = _collect_benchmark_plan(config_path, task)
    _echo_benchmark_plan(plan, machine, title="Inspection summary")
