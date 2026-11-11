# Recorded comparison data

These files preserve the measurements summarised in the main README.
`benchmark_results.json` contains all 30 timed runs, seeds, counts, summaries
and environment metadata for the five-seed, 5,000-electron-cap comparison.
`validation_results.json` contains the six ensemble summaries from 20 seeds
per backend/configuration at a 1,000-electron cap (seeds 0–19).

Recorded on Windows using MSYS2 UCRT64 GCC 16.1.0 (`-O3`), CPython 3.12,
Numba 0.68.0 and NumPy 2.5.4. Both scripts time the public API after warmup,
including NumPy conversion and excluding plotting and compilation. Backend
order alternates in the five-seed benchmark. Local timing variability was
substantial; at 1300 V individual Numba times were 0.83–2.41 s and C times
were 0.71–1.85 s. These records are configuration-specific measurements.

Reproduce from the repository root without overwriting these records:

```bash
python -m tests.benchmark_backends
python -m tests.validate_backends --output tests/results/latest/validation_results.json
```

New results go into the Git-ignored `latest/` directory. The original
10× Numba-over-baseline-Python result predates these C comparisons; its
individual timing samples are not included in these JSON records.
