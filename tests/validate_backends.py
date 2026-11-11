"""Compare independent seeded ensembles; timings exclude JIT and library loading."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from simulation import simulate_avalanche as numba_run
from simulation_c import simulate_avalanche as c_run

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save results as UTF-8 JSON")
    args = parser.parse_args()
    numba_run(max_electrons=1)
    c_run(max_electrons=1)
    rows = []
    for voltage, radius in [(600, .0006), (1000, .003), (1300, .009)]:
        for name, run in [("Numba", numba_run), ("C99", c_run)]:
            counts, points, radii, z_values, seconds = [], [], [], [], []
            truncated = 0
            for seed in range(20):
                start = time.perf_counter()
                result = run(voltage, radius, seed, 1000)
                seconds.append(time.perf_counter()-start)
                counts.append(result.electrons_processed)
                points.append(len(result.points))
                radii.extend(np.hypot(result.points[:, 0], result.points[:, 1]))
                z_values.extend(result.points[:, 2])
                truncated += result.truncated
            rows.append(dict(voltage=voltage, radius=radius, backend=name,
                seeds=20, cap=1000, mean_electrons=float(np.mean(counts)),
                sd_electrons=float(np.std(counts, ddof=1)), mean_points=float(np.mean(points)),
                radial_quantiles_m=np.quantile(radii, [.1,.5,.9]).tolist(),
                z_rms_m=float(np.sqrt(np.mean(np.square(z_values)))),
                truncated_runs=truncated, median_seconds=float(np.median(seconds))))
    output = json.dumps(rows, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
