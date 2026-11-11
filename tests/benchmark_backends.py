"""Warm backend timings, alternating order across five independent seeds."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numba
import numpy as np
from simulation import simulate_avalanche as numba_run
from simulation_c import simulate_avalanche as c_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results/latest/benchmark_results.json")
    args = parser.parse_args()
    for run in [numba_run, c_run]:
        run(600, .0006, 42, 10)
    rows = []
    for voltage, radius, cap in [(600, .0006, 5000), (1000, .003, 5000), (1300, .009, 5000)]:
        for index, seed in enumerate([0, 1, 2, 3, 42]):
            backends = [("Numba", numba_run), ("C99", c_run)]
            if index % 2:
                backends.reverse()
            for name, run in backends:
                start = time.perf_counter()
                result = run(voltage, radius, seed, cap)
                elapsed = time.perf_counter() - start
                rows.append(dict(voltage=voltage, radius_m=radius, cap=cap,
                    seed=seed, backend=name, seconds=elapsed,
                    electrons=result.electrons_processed, points=len(result.points),
                    truncated=result.truncated))
                print(f"{voltage} V, seed {seed}, {name}: {elapsed:.4f} s, "
                      f"{result.electrons_processed} electrons, {len(result.points)} points", flush=True)
    summaries = []
    for voltage in [600, 1000, 1300]:
        summary = {"voltage": voltage}
        for name in ["Numba", "C99"]:
            group = [r for r in rows if r["voltage"] == voltage and r["backend"] == name]
            summary[name] = dict(median_seconds=float(np.median([r["seconds"] for r in group])),
                min_seconds=min(r["seconds"] for r in group),
                max_seconds=max(r["seconds"] for r in group),
                mean_electrons=float(np.mean([r["electrons"] for r in group])),
                mean_points=float(np.mean([r["points"] for r in group])),
                truncated_runs=sum(r["truncated"] for r in group))
        summary["c_speedup"] = summary["Numba"]["median_seconds"]/summary["C99"]["median_seconds"]
        summaries.append(summary)
    output = dict(python=sys.version, platform=platform.platform(), numba=numba.__version__,
        numpy=np.__version__, method="Warm public API timings including NumPy conversion; alternating backend order; no plotting or JIT compilation; five seeds per configuration.",
        runs=rows, summaries=summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
