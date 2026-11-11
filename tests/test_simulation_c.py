"""Run from the repository root with python -m pytest -q."""
import ctypes as ct
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from simulation import elastic_cross_section_cm2, inelastic_cross_section_cm2
from simulation_c import simulate_avalanche, _library, CResult

@pytest.fixture(scope="session", autouse=True)
def build_current_sources():
    # Never test a stale DLL left over from an earlier successful build.
    subprocess.run([sys.executable, "build_c.py"],
                   cwd=Path(__file__).resolve().parents[1], check=True)

@pytest.mark.parametrize("voltage,radius,seed,cap", [
    (600, .0006, 0, 200), (1000, .003, 42, 1000),
    (1300, .009, 123, 200), (1000, .003, 9, 1)])
def test_transport(voltage, radius, seed, cap):
    result = simulate_avalanche(voltage, radius, seed, cap)
    repeat = simulate_avalanche(voltage, radius, seed, cap)
    np.testing.assert_array_equal(result.points, repeat.points)
    assert 1 <= result.electrons_processed <= cap
    assert result.electrons_processed == result.secondary_electrons_created + 1
    assert result.points.shape[1] == 3
    assert np.isfinite(result.points).all()
    radii = np.hypot(result.points[:, 0], result.points[:, 1])
    assert (radii >= 8e-5 * (1-1e-12)).all()
    assert (radii < .01).all()
    if cap == 1:
        assert result.truncated
    # The returned array owns its storage after the C allocation is freed.
    assert result.points.flags.owndata

@pytest.mark.parametrize("kwargs", [dict(voltage_v=0), dict(voltage_v=float("nan")),
    dict(start_radius_m=.0005), dict(start_radius_m=.01), dict(max_electrons=0)])
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        simulate_avalanche(**kwargs)

def test_physics_and_cleanup():
    lib = _library()
    for name, reference in [("elastic_cross_section", elastic_cross_section_cm2),
                             ("inelastic_cross_section", inelastic_cross_section_cm2)]:
        function = getattr(lib, name)
        function.argtypes = [ct.c_double]
        function.restype = ct.c_double
        for energy in [0, 2, 2.001, 10, 20, 24.599, 24.6, 100, 100.001, 150]:
            assert function(energy) == pytest.approx(reference(energy), rel=1e-14, abs=0)
    result = CResult()
    assert lib.simulate_avalanche(0, .003, 42, 10, ct.byref(result)) == -2
    lib.simulation_result_free(ct.byref(result))
    lib.simulation_result_free(ct.byref(result))
    assert not result.points.data and result.points.count == 0

@pytest.mark.parametrize("backend", ["Numba", "C99"])
def test_streamlit(backend):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()
    app.selectbox[0].select(backend)
    app.slider[0].set_value(600)
    app.slider[1].set_value(0.6)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 4
    assert len(app.get("image")) == 4
    assert app.session_state["backend"] == backend
    import matplotlib.pyplot as plt
    assert not plt.get_fignums()


@pytest.mark.parametrize("backend", ["Numba", "C99"])
@pytest.mark.parametrize("kwargs", [dict(voltage_v=float("inf")),
    dict(voltage_v=float("nan")), dict(start_radius_m=float("nan")),
    dict(max_electrons=0), dict(max_electrons=0.5)])
def test_invalid_both_backends(backend, kwargs):
    from simulation import simulate_avalanche as numba_run
    run = numba_run if backend == "Numba" else simulate_avalanche
    with pytest.raises((ValueError, TypeError)):
        run(**kwargs)


def test_field():
    from simulation import cylindrical_field_magnitude
    lib = _library()
    lib.electric_field.argtypes = [ct.c_double, ct.c_double]
    lib.electric_field.restype = ct.c_double
    for radius in [.00008, .0005, .003, .01]:
        assert lib.electric_field(radius, 1000) == pytest.approx(
            cylindrical_field_magnitude(radius, 1000), rel=1e-14)
    assert cylindrical_field_magnitude(.002, 1000) == pytest.approx(
        2*cylindrical_field_magnitude(.004, 1000))


def test_linux_build_on_demand(tmp_path, monkeypatch):
    import simulation_c
    root = tmp_path
    csrc = root / "csrc"
    csrc.mkdir()
    script = root / "build_c.py"
    script.write_text("# build script")
    source = csrc / "simulation.c"
    source.write_text("/* source */")
    library = csrc / "libsimulation.so"
    calls = []

    def build(command, **kwargs):
        calls.append((command, kwargs))
        library.write_bytes(b"test library")

    monkeypatch.setattr(simulation_c.subprocess, "run", build)
    simulation_c._prepare_library(library, platform="linux")
    assert len(calls) == 1
    assert calls[0][0] == [sys.executable, str(script)]
    simulation_c._prepare_library(library, platform="linux")
    assert len(calls) == 1
    import os
    stamp = library.stat().st_mtime + 10
    os.utime(source, (stamp, stamp))
    simulation_c._prepare_library(library, platform="linux")
    assert len(calls) == 2


def test_linux_build_failure_is_actionable(tmp_path, monkeypatch):
    import simulation_c
    csrc = tmp_path / "csrc"
    csrc.mkdir()
    (tmp_path / "build_c.py").write_text("# build script")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "gcc", stderr="compiler unavailable")

    monkeypatch.setattr(simulation_c.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="select Numba.*compiler unavailable"):
        simulation_c._prepare_library(csrc / "libsimulation.so", platform="linux")


def test_streamlit_c_failure_preserves_numba(monkeypatch):
    import simulation_c
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()
    app.slider[0].set_value(600)
    app.slider[1].set_value(0.6)
    app.button[0].click().run()
    previous = app.session_state["result"]

    def fail(**kwargs):
        raise RuntimeError("C compiler unavailable")

    monkeypatch.setattr(simulation_c, "simulate_avalanche", fail)
    app.selectbox[0].select("C99")
    app.button[0].click().run()
    assert not app.exception
    assert "C compiler unavailable" in app.error[0].value
    assert any("Select Numba" in item.value for item in app.info)
    assert app.session_state["backend"] == "Numba"
    np.testing.assert_array_equal(app.session_state["result"].points, previous.points)
    assert len(app.get("image")) == 4
