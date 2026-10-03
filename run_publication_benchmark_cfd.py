"""
run_publication_benchmark_cfd.py
=================================
Drop-in replacement for run_publication_benchmark.py that uses CFD-backed
flow fields instead of the analytical Poiseuille model.

DIFF from the original benchmark
---------------------------------
Only three changes from your existing run_publication_benchmark.py:

    1. Import RealVesselNetworkCFD instead of RealVesselNetwork
    2. Import SteadyCFDFlowField
    3. Pass cfd_flow=... when constructing the network

Everything else — controllers, integrator, reporting, trial loop — is
identical.  Copy this file next to run_publication_benchmark.py and run it.

Usage
-----
    python run_publication_benchmark_cfd.py

It picks up the .npz files from data/cfd/ automatically.
If a .npz is missing for a vessel, it falls back to the analytical model
and prints a warning — so you can run partial results during development.
"""

import os
import sys

# Repo root on path
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from dynamag_simulation.core.patched_vessel_network import RealVesselNetworkCFD
from dynamag_simulation.core.cfd_flow_field import SteadyCFDFlowField

# ---------------------------------------------------------------------------
# CFD interpolator paths — edit if your layout differs
# ---------------------------------------------------------------------------
_CFD_DIR = os.path.join(_repo, "data", "cfd")

CFD_NPZ = {
    "abdominal_aorta": os.path.join(_CFD_DIR, "abdominal_aorta_steady.npz"),
    "aortic_root":     os.path.join(_CFD_DIR, "aortic_root_steady.npz"),
    "pulmonary":       os.path.join(_CFD_DIR, "pulmonary_steady.npz"),
}

# ---------------------------------------------------------------------------
# Helper: build a CFD-backed network, falling back gracefully
# ---------------------------------------------------------------------------
def make_cfd_network(pkl_path: str, vessel_key: str, flow_scale: float = 1.0):
    """
    Construct a RealVesselNetworkCFD.

    If the .npz for this vessel isn't built yet, falls back to the
    analytical Poiseuille model with a clear warning.
    """
    npz_path = CFD_NPZ.get(vessel_key)

    cfd_flow = None
    if npz_path and os.path.exists(npz_path):
        print(f"[CFD] Loading {vessel_key} interpolator: {npz_path}")
        cfd_flow = SteadyCFDFlowField(npz_path, flow_scale=flow_scale)
    else:
        print(f"[CFD WARNING] No interpolator found for '{vessel_key}' at:")
        print(f"              {npz_path}")
        print(f"              Falling back to analytical Poiseuille model.")
        print(f"              Run run_all_vessels_cfd.sh to build it.")

    return RealVesselNetworkCFD(
        pkl_file=pkl_path,
        cfd_flow=cfd_flow,
        flow_scale=flow_scale,
    )


# ---------------------------------------------------------------------------
# Below: paste your existing run_publication_benchmark.py body here,
# replacing lines like:
#
#   network = RealVesselNetwork(pkl, flow_scale=fs)
#
# with:
#
#   network = make_cfd_network(pkl, vessel_key="abdominal_aorta", flow_scale=fs)
#
# The vessel_key must match one of the keys in CFD_NPZ above.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Minimal example showing the exact substitution pattern
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Example: abdominal aorta, flow_scale=1.0 ---
    PKL_ABDOMINAL = os.path.join(_repo, "data", "real_vessel_segments.pkl")
    PKL_AORTIC    = os.path.join(_repo, "data", "aortic_root_segments.pkl")
    PKL_PULMONARY = os.path.join(_repo, "data", "pulmonary_segments.pkl")

    FLOW_SCALES = [1.0, 2.0, 3.0, 5.0]

    # Map your pkl filenames to the CFD keys defined above
    VESSEL_CONFIGS = [
        (PKL_ABDOMINAL, "abdominal_aorta"),
        (PKL_AORTIC,    "aortic_root"),
        (PKL_PULMONARY, "pulmonary"),
    ]

    for pkl, key in VESSEL_CONFIGS:
        if not os.path.exists(pkl):
            print(f"[SKIP] PKL not found: {pkl}")
            continue

        for fs in FLOW_SCALES:
            network = make_cfd_network(pkl, vessel_key=key, flow_scale=fs)
            print(network.segment_summary())

            # ----------------------------------------------------------------
            # Your existing trial loop goes here.
            # network.velocity_at(pos, time) now queries the CFD field.
            # All controller constructors, the integrator, and reporting
            # are completely unchanged.
            # ----------------------------------------------------------------
            # Example:
            #   from dynamag_simulation.core.comet_controller import COMETController
            #   from dynamag_simulation.core.integrator import run_trial
            #   result = run_trial(network, start, target, COMETController(), ...)
            # ----------------------------------------------------------------

    print("\nDone. Paste your existing trial loop body above and remove this message.")
