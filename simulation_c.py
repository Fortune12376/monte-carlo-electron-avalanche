"""ctypes interface to the C99 backend; build first with python build_c.py."""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
import math
import operator
from pathlib import Path
import sys
import subprocess
import threading

import numpy as np
from simulation import SimulationResult

class Vector3(ct.Structure):
    _fields_ = [("x", ct.c_double), ("y", ct.c_double), ("z", ct.c_double)]

class PointArray(ct.Structure):
    _fields_ = [("data", ct.POINTER(Vector3)), ("count", ct.c_size_t),
                ("capacity", ct.c_size_t)]

class CResult(ct.Structure):
    _fields_ = [("points", PointArray), ("electrons_processed", ct.c_size_t),
                ("secondary_electrons_created", ct.c_size_t), ("truncated", ct.c_int)]

# srand/rand have global state. Serialize calls made through this wrapper.
_lock = threading.Lock()
_build_lock = threading.Lock()

def _prepare_library(path, platform=None):
    """Build missing or outdated Linux libraries, including on Community Cloud."""
    if (platform or sys.platform) != "linux":
        return
    root = path.parent.parent
    sources = [root / "build_c.py", *path.parent.glob("*.c"), *path.parent.glob("*.h")]
    with _build_lock:
        if path.exists() and all(source.stat().st_mtime <= path.stat().st_mtime
                                 for source in sources):
            return
        try:
            subprocess.run([sys.executable, str(root / "build_c.py")], cwd=root,
                           check=True, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as error:
            detail = getattr(error, "stderr", None) or str(error)
            raise RuntimeError(
                "Could not build the Linux C backend. Install GCC (build-essential), "
                "or select Numba in the sidebar. Build details: " + str(detail)
            ) from error


@lru_cache(maxsize=1)
def _library():
    name = "simulation.dll" if sys.platform == "win32" else "libsimulation.so"
    path = Path(__file__).resolve().parent / "csrc" / name
    _prepare_library(path)
    try:
        lib = ct.CDLL(str(path))
    except OSError as error:
        code = getattr(error, "winerror", None)
        if code == 5:
            advice = "Windows denied access to the DLL. Rebuild with python build_c.py to restore its permissions."
        elif code == 4551:
            advice = "Windows Application Control blocked this DLL. Check Windows Security or your administrator's application policy."
        else:
            advice = "Run python build_c.py first."
        raise RuntimeError(f"Cannot load {path}. {advice} {error}") from error
    lib.simulate_avalanche.argtypes = [ct.c_double, ct.c_double, ct.c_uint,
                                      ct.c_size_t, ct.POINTER(CResult)]
    lib.simulate_avalanche.restype = ct.c_int
    lib.simulation_result_free.argtypes = [ct.POINTER(CResult)]
    lib.simulation_result_free.restype = None
    return lib

def simulate_avalanche(voltage_v=1000.0, start_radius_m=3.0e-3,
                       seed=42, max_electrons=20_000):
    voltage_v, start_radius_m = float(voltage_v), float(start_radius_m)
    if not math.isfinite(voltage_v) or voltage_v <= 0:
        raise ValueError("voltage_v must be finite and positive")
    if not 5.0e-4 < start_radius_m < 1.0e-2:
        raise ValueError("start_radius_m must lie between the secondary-suppression radius and cathode")
    max_electrons = operator.index(max_electrons)
    if not 1 <= max_electrons <= ct.c_size_t(-1).value:
        raise ValueError("max_electrons must fit a positive size_t")
    seed = int(seed) % (2 ** (8 * ct.sizeof(ct.c_uint)))
    lib = _library()
    result = CResult()
    with _lock:
        try:
            status = lib.simulate_avalanche(voltage_v, start_radius_m, seed,
                                            max_electrons, ct.byref(result))
            if status == -1:
                raise MemoryError("C simulation allocation failed")
            if status != 0:
                raise ValueError("Invalid C simulation parameters")
            count = result.points.count
            if count:
                # Copy before freeing the C allocation; Vector3 is three doubles.
                data = ct.cast(result.points.data, ct.POINTER(ct.c_double))
                points = np.ctypeslib.as_array(data, shape=(count * 3,)).reshape(count, 3).copy()
            else:
                points = np.empty((0, 3), dtype=np.float64)
            return SimulationResult(points, result.electrons_processed,
                                    result.secondary_electrons_created, bool(result.truncated))
        finally:
            lib.simulation_result_free(ct.byref(result))

if __name__ == "__main__":
    result = simulate_avalanche()
    print(f"Points: {len(result.points)}, electrons: {result.electrons_processed}, "
          f"secondaries: {result.secondary_electrons_created}, truncated: {result.truncated}")
