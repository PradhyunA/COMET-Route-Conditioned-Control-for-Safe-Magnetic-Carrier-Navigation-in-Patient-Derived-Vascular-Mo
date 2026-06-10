"""
Physical and geometric constants used across all simulators.

All quantities in SI units unless otherwise noted.
"""

import numpy as np

MU_0 = 4.0 * np.pi * 1e-7       # vacuum permeability  (T*m/A)
K_B = 1.381e-23                  # Boltzmann constant   (J/K)
PI = np.pi

INCH_TO_METER = 0.0254
SIDE = 6.0 * INCH_TO_METER      # workspace side length 0.1524 m
HALF = SIDE / 2.0                # half-side 0.0762 m

EPS = 1e-30                      # numerical floor to avoid division by zero

TEST_RANGE = 0.04                # default trial extent (m); keeps targets > 2x coil_radius from magnets
DEFAULT_TARGET_TOLERANCE = 0.0005  # 0.5 mm final-target / waypoint-arrival tolerance

WATER_VISCOSITY_25C = 8.9e-4     # Pa*s
WATER_DENSITY_25C = 997.0        # kg/m^3
