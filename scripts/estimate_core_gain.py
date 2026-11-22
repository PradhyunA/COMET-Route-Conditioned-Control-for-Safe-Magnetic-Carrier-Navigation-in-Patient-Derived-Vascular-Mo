"""
Estimate the far-field gain an iron core gives one coil of the array.

The far field of a cored solenoid is a dipole whose moment is the sum of
the winding moment and the core's induced moment:

    m_air  = N I pi R^2
    m_core = chi_app * <H_axial> * V_core
    gain   = 1 + m_core / m_air

The core's apparent susceptibility is limited by its own demagnetising
field, chi_app = (mu_r - 1) / (1 + N_d (mu_r - 1)), with N_d taken from a
prolate spheroid of the same aspect ratio (Osborn 1945). This is a
first-order estimate, not a substitute for an FEM solve of the coil.

Defaults match ElectromagnetArray3D.
"""

import argparse

import numpy as np


def prolate_demagnetizing_factor(aspect_ratio):
    """Axial demagnetising factor of a prolate spheroid, length/diameter = m."""
    m = float(aspect_ratio)
    if m <= 1.0:
        return 1.0 / 3.0
    s = np.sqrt(m * m - 1.0)
    return (m / s * np.log(m + s) - 1.0) / (m * m - 1.0)


def mean_axial_field_per_amp_turn(length, radius, n=2001):
    """Average on-axis H of a finite solenoid over its own length, per (N I)."""
    z = np.linspace(-0.5 * length, 0.5 * length, n)
    a = z + 0.5 * length
    b = z - 0.5 * length
    h = (a / np.hypot(a, radius) - b / np.hypot(b, radius)) / (2.0 * length)
    return float(h.mean())


def core_gain(mu_r, coil_radius, coil_length, core_radius, winding_radius):
    n_d = prolate_demagnetizing_factor(coil_length / (2.0 * core_radius))
    chi_app = (mu_r - 1.0) / (1.0 + n_d * (mu_r - 1.0))
    h_per_ni = mean_axial_field_per_amp_turn(coil_length, winding_radius)
    v_core = np.pi * core_radius ** 2 * coil_length
    m_air_per_ni = np.pi * coil_radius ** 2
    return 1.0 + chi_app * h_per_ni * v_core / m_air_per_ni, n_d, chi_app


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coil-radius", type=float, default=0.070)
    parser.add_argument("--coil-length", type=float, default=0.160)
    parser.add_argument("--core-radius", type=float, default=0.030)
    args = parser.parse_args()

    mean_winding = 0.5 * (args.core_radius + args.coil_radius)
    print(
        f"coil R={args.coil_radius * 1e3:.0f} mm, L={args.coil_length * 1e3:.0f} mm, "
        f"core r={args.core_radius * 1e3:.0f} mm"
    )
    for label, winding in (("outer", args.coil_radius), ("mean", mean_winding)):
        for mu_r in (70.0, 1000.0, 1e9):
            gain, n_d, chi_app = core_gain(
                mu_r, args.coil_radius, args.coil_length, args.core_radius, winding,
            )
            mu_label = "inf" if mu_r > 1e6 else f"{mu_r:g}"
            print(
                f"  winding radius {label:5s} | mu_r {mu_label:>5s} | N_d {n_d:.3f} "
                f"| chi_app {chi_app:5.2f} | far-field gain {gain:.2f}"
            )


if __name__ == "__main__":
    main()
