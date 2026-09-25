"""Interactive Streamlit front end for the electron-avalanche simulation."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import streamlit as st

from simulation import (
    ANODE_RADIUS_M,
    CATHODE_RADIUS_M,
    SECONDARY_SUPPRESSION_RADIUS_M,
    DEFAULT_START_RADIUS_M,
    elastic_cross_section_cm2,
    inelastic_cross_section_cm2,
    simulate_avalanche,
)

st.set_page_config(page_title="Electron Avalanche Monte Carlo", layout="wide")

st.title("Monte Carlo Electron Avalanche Simulator")
st.caption(
    "A simplified null-collision Monte Carlo model of electron transport and "
    "avalanche formation in helium inside a cylindrical proportional-counter geometry."
)

with st.sidebar:
    st.header("Simulation controls")
    voltage = st.slider("Anode voltage (V)", 600, 1300, 1000, 50)
    start_radius_mm = st.slider(
        "Starting radius (mm)", 0.6, 9.0, float(DEFAULT_START_RADIUS_M * 1e3), 0.1
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    max_electrons = st.select_slider(
        "Safety cap (electrons)", options=[2_000, 5_000, 10_000, 20_000, 50_000, 100_000], value=20_000
    )
    run_clicked = st.button("Run simulation", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.runtime = None

if run_clicked:
    with st.spinner("Running Monte Carlo transport…"):
        t0 = time.perf_counter()
        st.session_state.result = simulate_avalanche(
            voltage_v=float(voltage),
            start_radius_m=float(start_radius_mm) * 1e-3,
            seed=int(seed),
            max_electrons=int(max_electrons),
        )
        st.session_state.runtime = time.perf_counter() - t0

result = st.session_state.result

if result is None:
    st.info("Choose parameters in the sidebar and click **Run simulation**.")
else:
    data = result.points
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stored interaction points", f"{len(data):,}")
    m2.metric("Electrons processed", f"{result.electrons_processed:,}")
    m3.metric("Secondaries created", f"{result.secondary_electrons_created:,}")
    m4.metric("Runtime", f"{st.session_state.runtime:.2f} s")

    if result.truncated:
        st.warning(
            "The safety cap was reached. Increase it for a fuller avalanche, "
            "or lower the voltage."
        )

    avalanche_tab, chamber_tab, view3d_tab, cross_tab, model_tab = st.tabs(
        ["Avalanche", "Full chamber", "3D view", "Cross sections", "Model"]
    )

    with avalanche_tab:
        fig, ax = plt.subplots(figsize=(8, 7))
        if len(data):
            ax.scatter(data[:, 0] * 1e6, data[:, 1] * 1e6, s=3.0, alpha=0.45)
        anode = patches.Circle(
            (0, 0), ANODE_RADIUS_M * 1e6, fill=True, alpha=0.35, label="Anode"
        )
        ax.add_patch(anode)
        ax.add_patch(
            patches.Circle(
                (0, 0),
                SECONDARY_SUPPRESSION_RADIUS_M * 1e6,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label="Queued-secondary suppression radius",
            )
        )
        zoom_um = 550
        ax.set(xlim=(-zoom_um, zoom_um), ylim=(-zoom_um, zoom_um))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        ax.set_title("Near-anode avalanche region")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    with chamber_tab:
        fig, ax = plt.subplots(figsize=(8, 7))
        if len(data):
            ax.scatter(data[:, 0] * 1e3, data[:, 1] * 1e3, s=0.2, alpha=0.2)
        ax.add_patch(
            patches.Circle((0, 0), ANODE_RADIUS_M * 1e3, fill=True, alpha=0.5)
        )
        ax.add_patch(
            patches.Circle(
                (0, 0), CATHODE_RADIUS_M * 1e3, fill=False, linestyle="--", linewidth=1.5
            )
        )
        limit = CATHODE_RADIUS_M * 1e3 * 1.05
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Full cylindrical chamber")
        st.pyplot(fig, clear_figure=True)

    with view3d_tab:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        if len(data):
            ax.scatter(
                data[:, 0] * 1e3,
                data[:, 1] * 1e3,
                data[:, 2] * 1e3,
                s=0.5,
                alpha=0.12,
            )
            zmin, zmax = data[:, 2].min() * 1e3, data[:, 2].max() * 1e3
            ax.plot([0, 0], [0, 0], [zmin, zmax], linewidth=2.5, label="Anode wire")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.set_title("3D interaction cloud")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    with cross_tab:
        energies = np.linspace(0.0, 150.0, 500)
        elastic = np.array([elastic_cross_section_cm2(e) for e in energies])
        inelastic = np.array([inelastic_cross_section_cm2(e) for e in energies])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(energies, elastic, label="Elastic")
        ax.plot(energies, inelastic, label="Ionisation / inelastic")
        ax.axvline(24.6, linestyle="--", alpha=0.6, label="24.6 eV threshold")
        ax.set_xlabel("Electron energy [eV]")
        ax.set_ylabel("Cross section [cm²]")
        ax.set_title("Cross-section approximations used by the model")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    with model_tab:
        st.markdown(
            r"""
### Model summary

The detector is represented as a long cylindrical cathode surrounding a thin
anode wire. Neglecting end effects, the field magnitude is

$$E(r)=\frac{V}{r\ln(r_c/r_a)}.$$

Electrons are accelerated inward, candidate free-flight times are sampled from
an exponential distribution, and the null-collision method is used to decide
whether each proposal corresponds to a real elastic collision.  Ionisation creates a secondary electron at the collision position.

The assignment's 0.5 mm avalanche-control rule applies only to **new electron
starting positions popped from the charge pool**. An electron already being
transported continues inward until it reaches the physical 80 µm anode radius.

This is intentionally a **simplified educational model**. In particular, the
elastic angular distribution is approximated by random forward-hemisphere
scattering rather than a measured differential cross section.
"""
        )
