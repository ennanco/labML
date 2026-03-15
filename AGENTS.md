# AGENTS

## Estado actual (cierre de sesion)

- Ultimo commit integrado y pusheado: `90befbf` en `main`.
- Benchmark refactorizado en modulos:
  - `labml/core/benchmark.py` (API publica)
  - `labml/core/benchmark_context.py`
  - `labml/core/benchmark_engine.py`
  - `labml/core/benchmark_plan.py`
  - `labml/core/benchmark_progress.py`
  - `labml/core/benchmark_results.py`
  - `labml/core/benchmark_models.py`
- API publica estable de benchmark:
  - `run_benchmark`
  - `inspect_benchmark`
- Test suite en verde: `88 passed`.
- Ejemplos revisados y ejecutables desde raiz.
- GIFs regenerados y sincronizados con el estado actual del CLI.
- Entrypoint principal movido a `labml/cli.py`.

## Convenciones de trabajo

- No commitear artefactos/privados:
  - `_data_/`
  - `results/`
  - `RESUME.md`
  - `.venv/`, `_artifacts_/`, caches
- Mantener `benchmark.py` pequeno (fachada/orquestacion).
- Mantener implementacion interna en modulos `benchmark_*`.

## Comandos rapidos

```bash
uv sync --extra dev
uv run python -m pytest -q
uv run labml --help
uv run labml prepare --config examples/prepare.toml
uv run labml benchmark-regression --config examples/benchmark_regression.toml
uv run labml benchmark-classification --config examples/benchmark_classification.toml
uv run labml inspect-config --config examples/benchmark_regression.toml --task regression
scripts/demo/build_gifs.sh
```

## Pendientes prioritarios (proxima sesion)

- Sin pendientes prioritarios bloqueantes en este momento.

## Notas operativas

- Si se tocan ejemplos/README, revalidar siempre:
  - comandos de ejemplo,
  - `pytest`,
  - y regeneracion de GIFs si cambia la UX del CLI.
