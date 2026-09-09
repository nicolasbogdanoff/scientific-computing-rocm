# Scientific Computing ROCm Diagnostics

A small, honest foundation for future AI/ML and accelerated scientific-computing experiments. It reports what the local Python environment actually exposes instead of presenting a decorative ROCm claim.

## What it does

- Detects whether PyTorch is installed.
- Distinguishes a reported ROCm/HIP runtime from CUDA and CPU fallback.
- Reports accelerator visibility and device count when PyTorch exposes them.
- Records Python and PyTorch versions when available, supporting reproducible experiment logs.
- Selects `cuda` only when `torch.cuda.is_available()` is true; otherwise it selects CPU, unless `require_accelerator=True` is used to fail explicitly.
- Includes the visible device name in the report when an accelerator is usable.

ROCm-enabled PyTorch uses the CUDA-compatible device API, so the code intentionally uses the portable `cuda` device string while the report records the HIP version when available.

## Quick start

```bash
python -m venv .venv
python -m pip install -e '.[test]'
pytest -q
python examples/report_backend.py
```

For an optional local PyTorch experiment:

```bash
python -m pip install -e '.[test,torch]'
```

The package does not install ROCm or PyTorch automatically. Hardware, drivers, and framework wheels must be selected for the host system and documented in the experiment record.

## Research track: 2D heat-equation PINN

The repository also contains a separate **draft experiment** for a physics-informed neural network (PINN) applied to the 2D heat equation. The implementation, environment notes, and result record are kept on [draft PR #1](https://github.com/nicolasbogdanoff/scientific-computing-rocm/pull/1); they are intentionally not part of the conservative diagnostics API on `main`. Results in that experiment are branch-specific and should be interpreted together with its documented hardware and software context.

## Research direction

Future additions can benchmark matrix multiplication, compare numerical tolerances, and record environment metadata. Any performance result should include hardware, software versions, tensor shapes, warm-up policy, and repetitions.

## Author

Nicolás Mauricio Bogdanoff · Universidad Paraguayo Alemana (UPA) · [ORCID](https://orcid.org/0009-0004-6275-3013)

## License and citation

MIT License. See `CITATION.cff` for citation metadata.
