"""
simulation_3d_realistic.py -- Backward-compatibility redirect.

All physics and simulation logic lives in ``dynamag_simulation.core``.
The primary entry point is ``simulation_3d_mpc.simulate_trial_mpc()``.
"""

from dynamag_simulation.core.constants import *      # noqa: F401,F403
from dynamag_simulation.core.workspace import *      # noqa: F401,F403
from dynamag_simulation.core.magnets import ElectromagnetArray3D as ElectromagnetArray3D_Real  # noqa: F401
from dynamag_simulation.core.particle import *       # noqa: F401,F403

MAX_STEP = 5e-3  # kept for any remaining external imports
