"""Monte Carlo electron transport in a cylindrical helium proportional counter.

This module contains the numerical model used by both the command-line/demo code
and the Streamlit application.  The cross-section approximations are retained
from the original coursework model, while the surrounding interface has been
cleaned up for reproducibility and reuse.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numba
import numpy as np

# --- Geometry / operating defaults -------------------------------------------------
ANODE_RADIUS_M = 8.0e-5
CATHODE_RADIUS_M = 1.0e-2
SECONDARY_SUPPRESSION_RADIUS_M = 5.0e-4
DEFAULT_START_RADIUS_M = 3.0e-3
DEFAULT_VOLTAGE_V = 1000.0

# --- Physical constants / model parameters ---------------------------------------
HELIUM_DENSITY_KG_M3 = 0.1664
HELIUM_MOLAR_MASS_KG_MOL = 0.0040026
AVOGADRO = 6.022e23
ELECTRON_MASS_KG = 9.1093837015e-31
ELEMENTARY_CHARGE_C = 1.602176634e-19
THERMAL_ENERGY_EV = 0.025
NULL_COLLISION_KMAX_M3_S = 2.0e-12


@dataclass(frozen=True)
class SimulationResult:
    """Container returned by :func:`simulate_avalanche`."""

    points: np.ndarray
    electrons_processed: int
    secondary_electrons_created: int
    truncated: bool


def elastic_cross_section_cm2(energy_ev: float) -> float:
    """Piecewise elastic electron-helium cross section used by the model."""
    if energy_ev <= 2.0:
        return 7.0e-16
    if energy_ev <= 10.0:
        return energy_ev * (-0.27231e-16) + 6.455375e-16
    if energy_ev <= 20.0:
        return energy_ev * (-0.20977e-16) + 6.8801e-16
    if energy_ev <= 100.0:
        return energy_ev * (-0.02981e-16) + 3.2809e-16
    return 3.0e-17


def inelastic_cross_section_cm2(energy_ev: float) -> float:
    """Simplified ionisation cross section with a 24.6 eV threshold."""
    slope = 0.047745e-17
    intercept = -1.174527e-17
    threshold_ev = 24.6
    if energy_ev < threshold_ev:
        return 0.0
    if energy_ev > 100.0:
        return 3.6e-17
    return energy_ev * slope + intercept


_fast_elastic_cross_section = numba.njit(cache=True)(elastic_cross_section_cm2)
_fast_inelastic_cross_section = numba.njit(cache=True)(inelastic_cross_section_cm2)


@numba.njit(cache=True)
def _choose_cross_section(energy_ev: float):
    """Return (is_inelastic, elastic_cross_section_m2).

    The event-selection rule follows the original project model: once the
    inelastic channel is energetically available, the relative elastic and
    inelastic cross sections determine which process is proposed.
    """
    elastic = _fast_elastic_cross_section(energy_ev) * 1.0e-4
    inelastic = _fast_inelastic_cross_section(energy_ev) * 1.0e-4

    if inelastic <= 0.0:
        return False, elastic

    inelastic_ratio = inelastic / (inelastic + elastic)
    if random.random() < inelastic_ratio:
        return True, 0.0
    return False, elastic


def cylindrical_field_magnitude(radius_m: float, voltage_v: float) -> float:
    """Ideal coaxial-cylinder electric-field magnitude in V/m."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    return voltage_v / (radius_m * math.log(CATHODE_RADIUS_M / ANODE_RADIUS_M))


@numba.njit(cache=True, fastmath=True)
def _transport_kernel(
    start_position,
    voltage_v,
    seed,
    max_electrons,
):
    random.seed(seed)

    charges = numba.typed.List()
    charges.append(start_position.copy())
    interaction_points = numba.typed.List()

    number_density = (
        HELIUM_DENSITY_KG_M3 * AVOGADRO / HELIUM_MOLAR_MASS_KG_MOL
    )
    tau = 1.0 / (number_density * NULL_COLLISION_KMAX_M3_S)
    electron_accel_per_field = ELEMENTARY_CHARGE_C / ELECTRON_MASS_KG
    thermal_speed = math.sqrt(
        2.0 * THERMAL_ENERGY_EV * ELEMENTARY_CHARGE_C / ELECTRON_MASS_KG
    )
    field_constant = voltage_v / math.log(CATHODE_RADIUS_M / ANODE_RADIUS_M)

    electrons_processed = 0
    secondaries_created = 0
    truncated = False

    while len(charges) > 0:
        if electrons_processed >= max_electrons:
            truncated = True
            break

        pos = charges.pop()
        electrons_processed += 1

        if math.hypot(pos[0], pos[1]) < SECONDARY_SUPPRESSION_RADIUS_M:
            interaction_points.append(pos.copy())
            continue

        # Isotropic thermal starting velocity.
        phi = 2.0 * math.pi * random.random()
        cos_theta = 2.0 * random.random() - 1.0
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        vx = thermal_speed * sin_theta * math.cos(phi)
        vy = thermal_speed * sin_theta * math.sin(phi)
        vz = thermal_speed * cos_theta

        # Initial inward acceleration from the cylindrical field.
        radius = math.hypot(pos[0], pos[1])
        e_mag = field_constant / radius
        ax = electron_accel_per_field * e_mag * (-pos[0] / radius)
        ay = electron_accel_per_field * e_mag * (-pos[1] / radius)

        # Free-flight displacement accumulated across null-collision proposals.
        dx_acc = 0.0
        dy_acc = 0.0
        dz_acc = 0.0
        stop = False

        while not stop:
            # Protect log(0) while keeping the draw effectively uniform.
            u = max(random.random(), 1.0e-15)
            dt = -tau * math.log(u)

            # Advance the *candidate free-flight displacement*.  Position is only
            # committed when a real collision occurs, matching the null-collision
            # construction used in the coursework model.
            dx_acc += vx * dt + 0.5 * ax * dt * dt
            dy_acc += vy * dt + 0.5 * ay * dt * dt
            dz_acc += vz * dt
            vx += ax * dt
            vy += ay * dt

            candidate_x = pos[0] + dx_acc
            candidate_y = pos[1] + dy_acc
            candidate_radius = math.hypot(candidate_x, candidate_y)

            # The 0.5 mm rule in the assignment applies only when a *new*
            # electron is popped from the charge pool.  An electron already in
            # flight continues through this region until it reaches the anode.
            if candidate_radius <= ANODE_RADIUS_M:
                # Store a final point on the wire surface instead of a point
                # inside the conductor.  The simple radial projection is
                # sufficient for this educational trajectory model.
                if candidate_radius > 0.0:
                    scale = ANODE_RADIUS_M / candidate_radius
                    hit_x = candidate_x * scale
                    hit_y = candidate_y * scale
                else:
                    hit_x = ANODE_RADIUS_M
                    hit_y = 0.0
                interaction_points.append(
                    np.array([hit_x, hit_y, pos[2] + dz_acc])
                )
                stop = True
                continue
            if candidate_radius >= CATHODE_RADIUS_M:
                stop = True
                continue

            speed_sq = vx * vx + vy * vy + vz * vz
            energy_ev = 0.5 * ELECTRON_MASS_KG * speed_sq / ELEMENTARY_CHARGE_C
            is_inelastic, sigma_elastic = _choose_cross_section(energy_ev)

            if not is_inelastic:
                speed = math.sqrt(speed_sq)
                collision_rate = speed * sigma_elastic
                if random.random() > collision_rate / NULL_COLLISION_KMAX_M3_S:
                    # Null collision: keep accumulating the same free flight.
                    continue

                # Real elastic collision: commit accumulated displacement.
                pos[0] += dx_acc
                pos[1] += dy_acc
                pos[2] += dz_acc
                dx_acc = dy_acc = dz_acc = 0.0

                radius = math.hypot(pos[0], pos[1])
                e_mag = field_constant / radius
                ax = electron_accel_per_field * e_mag * (-pos[0] / radius)
                ay = electron_accel_per_field * e_mag * (-pos[1] / radius)

                if random.random() < 0.001:
                    interaction_points.append(pos.copy())

                # Simplified forward-hemisphere elastic scattering: preserve
                # speed, randomise direction, then flip outward directions.
                phi = 2.0 * math.pi * random.random()
                cos_theta = 2.0 * random.random() - 1.0
                sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
                nx = sin_theta * math.cos(phi)
                ny = sin_theta * math.sin(phi)
                nz = cos_theta

                if nx * pos[0] + ny * pos[1] > 0.0:
                    nx = -nx
                    ny = -ny
                    nz = -nz

                vx = speed * nx
                vy = speed * ny
                vz = speed * nz
                continue

            # Real inelastic collision: commit displacement and create a new
            # electron at the collision site.  The original electron continues
            # with thermal energy as specified by the original model.
            pos[0] += dx_acc
            pos[1] += dy_acc
            pos[2] += dz_acc
            dx_acc = dy_acc = dz_acc = 0.0

            phi = 2.0 * math.pi * random.random()
            cos_theta = 2.0 * random.random() - 1.0
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
            vx = thermal_speed * sin_theta * math.cos(phi)
            vy = thermal_speed * sin_theta * math.sin(phi)
            vz = thermal_speed * cos_theta

            radius = math.hypot(pos[0], pos[1])
            e_mag = field_constant / radius
            ax = electron_accel_per_field * e_mag * (-pos[0] / radius)
            ay = electron_accel_per_field * e_mag * (-pos[1] / radius)

            if electrons_processed + len(charges) < max_electrons:
                charges.append(pos.copy())
                secondaries_created += 1
            else:
                truncated = True

            interaction_points.append(pos.copy())

    return interaction_points, electrons_processed, secondaries_created, truncated


def simulate_avalanche(
    voltage_v: float = DEFAULT_VOLTAGE_V,
    start_radius_m: float = DEFAULT_START_RADIUS_M,
    seed: int = 42,
    max_electrons: int = 20_000,
) -> SimulationResult:
    """Run the Monte Carlo avalanche simulation.

    Parameters
    ----------
    voltage_v:
        Anode bias in volts.
    start_radius_m:
        Initial electron radius from the wire axis.  The initial azimuth is 0.
    seed:
        Seed for deterministic reruns.
    max_electrons:
        Safety cap for interactive use at high multiplication.

    Notes
    -----
    The original assignment suppresses *new electron starting positions* closer
    than 0.5 mm to the wire.  It does not stop an electron that is already in
    flight when that electron crosses 0.5 mm; transported electrons continue to
    the physical anode radius.
    """
    if not (SECONDARY_SUPPRESSION_RADIUS_M < start_radius_m < CATHODE_RADIUS_M):
        raise ValueError(
            "start_radius_m must lie between the secondary-suppression radius and cathode"
        )
    if voltage_v <= 0:
        raise ValueError("voltage_v must be positive")
    if max_electrons < 1:
        raise ValueError("max_electrons must be at least 1")

    start_position = np.array([start_radius_m, 0.0, 0.0], dtype=np.float64)
    points_typed, processed, secondaries, truncated = _transport_kernel(
        start_position,
        float(voltage_v),
        int(seed),
        int(max_electrons),
    )

    points = np.empty((len(points_typed), 3), dtype=np.float64)
    for i in range(len(points_typed)):
        points[i] = points_typed[i]

    return SimulationResult(
        points=points,
        electrons_processed=int(processed),
        secondary_electrons_created=int(secondaries),
        truncated=bool(truncated),
    )


if __name__ == "__main__":
    import time

    # First call includes Numba compilation; benchmark repeat runs separately.
    t0 = time.perf_counter()
    result = simulate_avalanche()
    elapsed = time.perf_counter() - t0
    print(f"Stored interaction points: {len(result.points):,}")
    print(f"Electrons processed: {result.electrons_processed:,}")
    print(f"Secondary electrons: {result.secondary_electrons_created:,}")
    print(f"Runtime (including any JIT compilation): {elapsed:.3f} s")
