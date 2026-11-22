"""
Electromagnet array: 9-magnet triangular-prism regularized dipole model.

Optimized hardware-constrained body-scale / large-phantom scanner version.

Layout:
    Front triangle (z = -d)
        A
       / \\
      B---C

    Back triangle (z = +d)
        D
       / \\
      E---F

    Mid-edge magnets:
        G between A-D
        H between B-E
        I between C-F

All magnets point toward the configured array center.

Each solenoid is modeled as a regularized magnetic dipole.

Magnetic moment:
    m = N I pi R^2

Regularized dipole field:
    B(r) = (mu_0 / 4pi)
           [3(m·r_hat)r_hat - m]
           / (r^2 + a^2)^(3/2)

Important:
    This version is still not a certified medical-device model.
    It is a hardware-constrained simulation model intended for physically
    plausible controller benchmarking.

    It includes:
        - finite wire resistance
        - voltage clipping
        - per-coil electrical power clipping
        - simple lumped thermal model
        - temperature-dependent copper resistance
        - thermal current derating
        - soft core saturation

    It still does NOT replace:
        - finite-element electromagnetic modeling
        - measured B-field calibration
        - full power electronics simulation
        - insulation design
        - cooling validation
        - biomedical safety certification
        - human or animal use approval

References:
    Jackson, Classical Electrodynamics, ch. 5
    Furlani, Permanent Magnet and Electromechanical Devices (2001)
"""

from __future__ import annotations

import numpy as np

from .constants import MU_0, PI, EPS


class ElectromagnetArray3D:
    """
    9 electromagnets arranged in a triangular prism.

    Optimized realistic benchmark:
        placement_radius = 0.300 m
        coil_radius      = 0.070 m
        coil_length      = 0.160 m
        core_radius      = 0.030 m
        turns            = 600
        max_current      = 16 A peak command limit

    Electrical model:
        wire_awg                 = 10
        supply_voltage_limit     = 48 V
        max_coil_power_w         = 250 W
        copper_temp_coeff        = 0.00393 1/C
        copper_resistivity_20c   = 1.68e-8 ohm*m

    Thermal model:
        ambient_temp_c             = 22 C
        max_coil_temp_c            = 80 C
        thermal_mass_j_per_c       = 900 J/C
        thermal_resistance_c_per_w = 0.25 C/W

    Core model:
        core_relative_permeability = 70 effective
        core_saturation_tesla      = 1.8 T

    Calibration:
        ampere_turn_scale multiplies every coil's dipole moment at a given
        command current. It stands in for a stronger (e.g. water-cooled)
        coil whose electrical and thermal limits are expressed per unit of
        command current, so the hardware model is unchanged. 1.0 is the
        coil described above.

    Notes:
        - This is far more realistic than the old 20,000-turn / 60 A model.
        - It is stronger than the first conservative 900-turn / 8 A model.
        - 16 A is treated as a peak command limit, not guaranteed continuous DC.
        - Actual current is clipped by voltage, power, and temperature limits.
    """

    def __init__(
        self,
        coil_radius: float = 0.070,
        coil_length: float = 0.160,
        turns: float = 600.0,
        max_current: float = 16.0,
        placement_radius: float = 0.300,
        center=None,
        core_radius: float = 0.030,
        core_relative_permeability: float = 70.0,
        core_saturation_tesla: float = 1.8,
        wire_awg: int = 10,
        supply_voltage_limit: float = 48.0,
        max_coil_power_w: float = 250.0,
        ambient_temp_c: float = 22.0,
        max_coil_temp_c: float = 80.0,
        thermal_mass_j_per_c: float = 900.0,
        thermal_resistance_c_per_w: float = 0.25,
        copper_resistivity_20c: float = 1.68e-8,
        copper_temp_coeff_per_c: float = 0.00393,
        winding_length_factor: float = 1.20,
        insulation_fill_factor: float = 0.70,
        enforce_hardware_limits: bool = True,
        ampere_turn_scale: float = 1.0,
    ):
        if ampere_turn_scale <= 0:
            raise ValueError(f"ampere_turn_scale must be > 0, got {ampere_turn_scale}")
        self.placement_radius = float(placement_radius)
        self.center = (
            np.zeros(3, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )

        self.coil_radius = float(coil_radius)
        self.coil_length = float(coil_length)
        self.turns = float(turns)
        self.max_current = float(max_current)
        self.ampere_turn_scale = float(ampere_turn_scale)

        self.core_radius = float(core_radius)
        self.core_relative_permeability = float(core_relative_permeability)
        self.core_saturation_tesla = float(core_saturation_tesla)

        self.wire_awg = int(wire_awg)
        self.supply_voltage_limit = float(supply_voltage_limit)
        self.max_coil_power_w = float(max_coil_power_w)

        self.ambient_temp_c = float(ambient_temp_c)
        self.max_coil_temp_c = float(max_coil_temp_c)
        self.thermal_mass_j_per_c = float(thermal_mass_j_per_c)
        self.thermal_resistance_c_per_w = float(thermal_resistance_c_per_w)

        self.copper_resistivity_20c = float(copper_resistivity_20c)
        self.copper_temp_coeff_per_c = float(copper_temp_coeff_per_c)

        self.winding_length_factor = float(winding_length_factor)
        self.insulation_fill_factor = float(insulation_fill_factor)
        self.enforce_hardware_limits = bool(enforce_hardware_limits)

        self._mu_over_4pi = MU_0 / (4.0 * PI)

        # Regularization length.
        # This prevents the dipole approximation from exploding near a source.
        reg_len = max(0.5 * self.coil_radius, self.core_radius, 1e-4)
        self._a_sq = reg_len ** 2

        self._local_positions = self._build_local_positions()
        self.positions = self._local_positions + self.center
        self.n_magnets = len(self.positions)

        self.orientations = self._build_orientations(self._local_positions)

        self.currents = np.zeros(self.n_magnets, dtype=float)
        self.commanded_currents = np.zeros(self.n_magnets, dtype=float)

        self.coil_temps_c = np.full(
            self.n_magnets,
            self.ambient_temp_c,
            dtype=float,
        )

        # Magnetic moment per ampere.
        # m = N I A, where A = pi R^2, times the calibration multiplier.
        self.moment_per_amp = (
            self.ampere_turn_scale * self.turns * PI * self.coil_radius ** 2
        )

        self.wire_diameter_m = self._awg_diameter_m(self.wire_awg)
        self.wire_area_m2 = PI * (0.5 * self.wire_diameter_m) ** 2

        self.estimated_wire_length_m = self._estimate_wire_length()
        self.coil_resistance_20c_ohm = self._coil_resistance_at_temp(
            self.ambient_temp_c
        )

        self.last_voltage_v = np.zeros(self.n_magnets, dtype=float)
        self.last_power_w = np.zeros(self.n_magnets, dtype=float)
        self.last_hardware_scale = np.ones(self.n_magnets, dtype=float)

    @staticmethod
    def _awg_diameter_m(awg: int) -> float:
        """
        Convert AWG gauge to bare copper diameter in meters.

        AWG diameter formula:
            d_inch = 0.005 inch * 92^((36 - AWG) / 39)
        """
        d_inch = 0.005 * (92.0 ** ((36.0 - float(awg)) / 39.0))
        return d_inch * 0.0254

    def _estimate_wire_length(self) -> float:
        """
        Estimate total copper length per coil.

        For a solenoid, each turn is roughly one circumference.
        The winding_length_factor accounts for layering, leads, imperfect packing,
        and the fact that outer layers are slightly larger than inner layers.
        """
        mean_turn_length = 2.0 * PI * self.coil_radius
        return self.turns * mean_turn_length * self.winding_length_factor

    def _coil_resistance_at_temp(self, temp_c: float) -> float:
        """
        Copper resistance with linear temperature correction.

        R(T) = R20 * [1 + alpha * (T - 20 C)]
        """
        r20 = (
            self.copper_resistivity_20c
            * self.estimated_wire_length_m
            / max(self.wire_area_m2 * self.insulation_fill_factor, EPS)
        )

        return r20 * (
            1.0 + self.copper_temp_coeff_per_c * (float(temp_c) - 20.0)
        )

    def _coil_resistances_now(self):
        return np.array(
            [self._coil_resistance_at_temp(t) for t in self.coil_temps_c],
            dtype=float,
        )

    def _thermal_current_derating(self):
        """
        Reduce allowable current as coil temperature approaches max temperature.

        Conservative derating:
            - full current below 60% of the allowed temperature rise
            - linearly derated after that
            - near zero at max temperature
        """
        temp_span = max(self.max_coil_temp_c - self.ambient_temp_c, EPS)
        hot_fraction = (self.coil_temps_c - self.ambient_temp_c) / temp_span
        hot_fraction = np.clip(hot_fraction, 0.0, 1.0)

        derating = np.ones_like(hot_fraction)

        start_derate = 0.60
        mask = hot_fraction > start_derate

        derating[mask] = (
            1.0
            - (hot_fraction[mask] - start_derate)
            / max(1.0 - start_derate, EPS)
        )

        return np.clip(derating, 0.0, 1.0)

    def _hardware_clip_currents(self, requested_currents):
        """
        Clip currents using:
            1. peak current limit
            2. thermal derating
            3. voltage limit
            4. per-coil power limit
        """
        requested_currents = np.asarray(requested_currents, dtype=float)

        current_limit = self.max_current * self._thermal_current_derating()

        currents = np.clip(
            requested_currents,
            -current_limit,
            current_limit,
        )

        resistances = self._coil_resistances_now()

        # Voltage limit:
        # |V| = |I| R <= Vmax
        voltage_current_limit = (
            self.supply_voltage_limit / np.maximum(resistances, EPS)
        )

        currents = np.clip(
            currents,
            -voltage_current_limit,
            voltage_current_limit,
        )

        # Power limit:
        # P = I^2 R <= Pmax
        power_current_limit = np.sqrt(
            self.max_coil_power_w / np.maximum(resistances, EPS)
        )

        currents = np.clip(
            currents,
            -power_current_limit,
            power_current_limit,
        )

        self.last_voltage_v = currents * resistances
        self.last_power_w = currents ** 2 * resistances

        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.divide(
                np.abs(currents),
                np.maximum(np.abs(requested_currents), EPS),
            )

        self.last_hardware_scale = np.clip(scale, 0.0, 1.0)

        return currents

    def _update_thermal_state(self, dt: float):
        """
        Simple lumped RC thermal model.

        dT/dt = (P - (T - Tambient) / Rth) / Cth

        where:
            P    = coil electrical heating power
            Rth  = thermal resistance, C/W
            Cth  = thermal mass, J/C
        """
        dt = float(dt)

        if dt <= 0.0:
            return

        cooling_power_w = (
            (self.coil_temps_c - self.ambient_temp_c)
            / max(self.thermal_resistance_c_per_w, EPS)
        )

        net_power_w = self.last_power_w - cooling_power_w

        dtemp = (
            net_power_w
            / max(self.thermal_mass_j_per_c, EPS)
        ) * dt

        self.coil_temps_c = self.coil_temps_c + dtemp

        # Do not let the toy model go below ambient.
        self.coil_temps_c = np.maximum(
            self.coil_temps_c,
            self.ambient_temp_c,
        )

    def _build_local_positions(self):
        d = self.placement_radius

        front = np.array(
            [
                [0.0, 2.0 * d / np.sqrt(3.0), -d],
                [-d, -d / np.sqrt(3.0), -d],
                [d, -d / np.sqrt(3.0), -d],
            ],
            dtype=float,
        )

        back = np.array(
            [
                [0.0, 2.0 * d / np.sqrt(3.0), d],
                [-d, -d / np.sqrt(3.0), d],
                [d, -d / np.sqrt(3.0), d],
            ],
            dtype=float,
        )

        mid = np.array(
            [
                0.5 * (front[0] + back[0]),
                0.5 * (front[1] + back[1]),
                0.5 * (front[2] + back[2]),
            ],
            dtype=float,
        )

        return np.vstack((front, back, mid))

    def _build_orientations(self, local_positions):
        orientations = np.zeros_like(local_positions)

        for i, pos in enumerate(local_positions):
            norm = np.linalg.norm(pos)

            if norm < 1e-12:
                orientations[i] = np.array([0.0, 0.0, 1.0])
            else:
                orientations[i] = -pos / norm

        return orientations

    def recenter(self, center):
        """
        Move the entire magnetic steering frame to a new center.

        This is useful for simulating a moving scanner/gantry frame.

        It does not change coil geometry, orientations, turns, or currents.
        """
        self.center = np.asarray(center, dtype=float)
        self.positions = self._local_positions + self.center

    def set_currents(self, currents, dt: float | None = None):
        """
        Set commanded currents.

        If enforce_hardware_limits=True, the requested current is clipped by:
            - max_current
            - temperature derating
            - voltage limit
            - power limit

        If dt is provided, the coil thermal state is updated for this timestep.
        """
        currents = np.asarray(currents, dtype=float)

        if currents.shape != (self.n_magnets,):
            raise ValueError(
                f"expected {self.n_magnets} currents, got shape {currents.shape}"
            )

        self.commanded_currents = currents.copy()

        if self.enforce_hardware_limits:
            safe_currents = self._hardware_clip_currents(currents)
        else:
            safe_currents = np.clip(
                currents,
                -self.max_current,
                self.max_current,
            )

            resistances = self._coil_resistances_now()
            self.last_voltage_v = safe_currents * resistances
            self.last_power_w = safe_currents ** 2 * resistances
            self.last_hardware_scale = np.ones(self.n_magnets, dtype=float)

        self.currents = safe_currents

        if dt is not None:
            self._update_thermal_state(float(dt))

    def step_thermal(self, dt: float):
        """
        Optional explicit thermal update.

        Use this if the caller cannot pass dt into set_currents().
        """
        self._update_thermal_state(float(dt))

    def _apply_core_model(self, B):
        """
        Effective soft-saturating core approximation.

        This is still simplified. It does not solve the real B-H curve.

        B_out = B_hat * B_sat * tanh(mu_eff * |B_air| / B_sat)
        """
        B = np.asarray(B, dtype=float)
        B_mag = float(np.linalg.norm(B))

        if B_mag < EPS:
            return B

        direction = B / B_mag

        effective_mag = self.core_saturation_tesla * np.tanh(
            self.core_relative_permeability * B_mag
            / (self.core_saturation_tesla + EPS)
        )

        return direction * effective_mag

    def field_at(self, point):
        """
        Net B-field in Tesla at point.
        """
        point = np.asarray(point, dtype=float)

        B_total_air = np.zeros(3, dtype=float)

        for i in range(self.n_magnets):
            current = self.currents[i]

            if abs(current) < EPS:
                continue

            m_mag = self.moment_per_amp * current
            m_vec = m_mag * self.orientations[i]

            r_vec = point - self.positions[i]
            r_sq = float(np.dot(r_vec, r_vec))

            r = np.sqrt(r_sq)

            if r < 1e-10:
                continue

            r_hat = r_vec / r
            m_dot_rhat = float(np.dot(m_vec, r_hat))

            denom = (r_sq + self._a_sq) ** 1.5

            B_i = self._mu_over_4pi * (
                3.0 * m_dot_rhat * r_hat - m_vec
            ) / denom

            B_total_air += B_i

        return self._apply_core_model(B_total_air)

    def grad_B_squared(self, point, h: float = 1e-4):
        """
        Gradient of |B|^2 using central differences.

        h = 1e-4 m gives a stable 0.1 mm finite-difference step.
        """
        point = np.asarray(point, dtype=float)

        def B_sq(p):
            B = self.field_at(p)
            return float(np.dot(B, B))

        grad = np.zeros(3, dtype=float)

        for axis in range(3):
            dp = np.zeros(3, dtype=float)
            dp[axis] = h

            grad[axis] = (
                B_sq(point + dp) - B_sq(point - dp)
            ) / (2.0 * h)

        return grad

    def hardware_diagnostics(self):
        """
        Return current electrical and thermal state.
        """
        resistances = self._coil_resistances_now()

        return {
            "max_commanded_current_A": float(
                np.max(np.abs(self.commanded_currents))
            ),
            "max_actual_current_A": float(np.max(np.abs(self.currents))),
            "max_voltage_V": float(np.max(np.abs(self.last_voltage_v))),
            "max_power_W": float(np.max(self.last_power_w)),
            "mean_power_W": float(np.mean(self.last_power_w)),
            "max_coil_temp_C": float(np.max(self.coil_temps_c)),
            "mean_coil_temp_C": float(np.mean(self.coil_temps_c)),
            "min_hardware_scale": float(np.min(self.last_hardware_scale)),
            "mean_hardware_scale": float(np.mean(self.last_hardware_scale)),
            "mean_resistance_ohm": float(np.mean(resistances)),
            "max_resistance_ohm": float(np.max(resistances)),
        }

    def params_dict(self):
        return {
            "n_magnets": self.n_magnets,
            "placement_radius_m": self.placement_radius,
            "coil_radius_m": self.coil_radius,
            "coil_diameter_m": 2.0 * self.coil_radius,
            "coil_length_m": self.coil_length,
            "turns": self.turns,
            "max_current_A": self.max_current,
            "current_mode": "hardware_constrained_peak_limit",
            "core_radius_m": self.core_radius,
            "core_diameter_m": 2.0 * self.core_radius,
            "core_relative_permeability_effective": (
                self.core_relative_permeability
            ),
            "core_saturation_tesla": self.core_saturation_tesla,
            "ampere_turn_scale": self.ampere_turn_scale,
            "moment_per_amp_A_m2": self.moment_per_amp,
            "wire_awg": self.wire_awg,
            "wire_diameter_m": self.wire_diameter_m,
            "wire_area_m2": self.wire_area_m2,
            "estimated_wire_length_m_per_coil": self.estimated_wire_length_m,
            "coil_resistance_20c_ohm": self.coil_resistance_20c_ohm,
            "supply_voltage_limit_V": self.supply_voltage_limit,
            "max_coil_power_W": self.max_coil_power_w,
            "ambient_temp_C": self.ambient_temp_c,
            "max_coil_temp_C": self.max_coil_temp_c,
            "thermal_mass_J_per_C": self.thermal_mass_j_per_c,
            "thermal_resistance_C_per_W": self.thermal_resistance_c_per_w,
            "thermal_model_included": True,
            "awg_model_included": True,
            "voltage_limit_included": True,
            "power_limit_included": True,
            "temperature_derating_included": True,
            "body_scale_scanner_model": True,
            "medical_safety_validated": False,
            "human_use_validated": False,
            "animal_use_validated": False,
            **self.hardware_diagnostics(),
        }