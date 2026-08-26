import unittest

import cupy as cp
import numpy as np

from rotating_shell_convection.config import ShellConvectionConfig
from rotating_shell_convection.geometry import ShellGeometry
from rotating_shell_convection.initial_conditions import ShellInitialCondition
from rotating_shell_convection.operators import ShellOperators
from rotating_shell_convection.radial import RadialGrid
from rotating_shell_convection.simulation import ShellConvectionSimulation
from rotating_shell_convection.sht import (
    SphericalHarmonicTransform,
    _lm_to_flat,
    collocation_grid_shape,
)


class RotatingShellConvectionTests(unittest.TestCase):
    def test_config_coefficients(self):
        config = ShellConvectionConfig(ra=1e5, pr=2.0, ek=1e-4, l_max=8, nr=16)
        self.assertAlmostEqual(config.viscous_coeff, 1e-4)
        self.assertAlmostEqual(config.thermal_diffusion_coeff, 1e-4 / 2.0)
        expected_buoyancy = 1e5 * (1e-4) ** 2 / 2.0
        self.assertAlmostEqual(config.buoyancy_coeff, expected_buoyancy)

    def test_collocation_grid_three_halves_rule(self):
        for l_max in (4, 8, 31):
            n_theta, n_phi = collocation_grid_shape(l_max)
            self.assertGreaterEqual(n_theta, int(np.ceil(1.5 * l_max)))
            self.assertGreaterEqual(n_phi, 3 * l_max)
            self.assertEqual(n_phi, 2 * n_theta)
            config = ShellConvectionConfig(l_max=l_max, nr=8)
            self.assertEqual(config.n_theta, n_theta)
            self.assertEqual(config.n_phi, n_phi)

    def test_sht_roundtrip(self):
        sht = SphericalHarmonicTransform(8, xp=cp)
        rng = np.random.default_rng(0)
        coeffs = cp.asarray(
            rng.standard_normal(sht.n_lm) + 1j * rng.standard_normal(sht.n_lm),
            dtype=cp.complex128,
        )
        field = sht.inverse(coeffs)
        coeffs_round = sht.forward(field)
        field_back = sht.inverse(coeffs_round)
        coeffs_again = sht.forward(field_back)
        err_grid = float(cp.max(cp.abs(field_back - field)).get())
        err_coeff = float(cp.max(cp.abs(coeffs_again - coeffs_round)).get())
        self.assertLess(err_grid, 1e-6)
        self.assertLess(err_coeff, 1e-6)

    def test_radial_map_and_derivative(self):
        eta = 0.5
        radial = RadialGrid(16, eta, xp=cp)
        xi = radial.xi
        r = radial.r
        expected_r = 0.5 * ((1.0 - eta) * xi + (1.0 + eta))
        err_r = float(cp.max(cp.abs(r - expected_r)).get())
        self.assertLess(err_r, 1e-12)
        self.assertLess(float(cp.abs(r[0] - 1.0).get()), 1e-12)
        self.assertLess(float(cp.abs(r[-1] - eta).get()), 1e-12)
        f = cp.ones_like(xi)
        df = radial.d_xi @ f
        interior = slice(1, -1)
        err_d = float(cp.max(cp.abs(df[interior])).get())
        self.assertLess(err_d, 1e-5)

    def test_pt_divergence_near_zero_for_zero_velocity(self):
        config = ShellConvectionConfig(l_max=8, nr=16, ra=1e4, ek=1e-3)
        geometry = ShellGeometry(config)
        operators = ShellOperators(geometry, config)
        w = cp.zeros((geometry.nr, geometry.n_lm), dtype=cp.complex128)
        z = cp.zeros((geometry.nr, geometry.n_lm), dtype=cp.complex128)
        div = operators.max_divergence(w, z)
        self.assertLess(div, 1e-6)

    def _single_mode_coeffs(self, geometry, l, m, radial):
        w = cp.zeros((geometry.nr, geometry.n_lm), dtype=cp.complex128)
        z = cp.zeros_like(w)
        w[:, _lm_to_flat(l, m)] = radial
        return w, z

    def test_torpol_spectral_div_constant_mode(self):
        config = ShellConvectionConfig(l_max=8, nr=16, ra=1e4, ek=1e-3)
        geometry = ShellGeometry(config)
        operators = ShellOperators(geometry, config)
        radial = cp.ones(geometry.nr, dtype=cp.complex128)
        w, z = self._single_mode_coeffs(geometry, 2, 0, radial)
        div = operators.max_divergence(w, z)
        self.assertLess(div, 1e-6)

    def test_torpol_spectral_div_linear_mode(self):
        config = ShellConvectionConfig(l_max=8, nr=16, ra=1e4, ek=1e-3)
        geometry = ShellGeometry(config)
        operators = ShellOperators(geometry, config)
        radial = geometry.r.astype(cp.complex128)
        w, z = self._single_mode_coeffs(geometry, 2, 1, radial)
        div = operators.max_divergence(w, z)
        self.assertLess(div, 1e-6)

    def test_torpol_ur_identity(self):
        config = ShellConvectionConfig(l_max=6, nr=12, ra=1e4, ek=1e-3)
        geometry = ShellGeometry(config)
        operators = ShellOperators(geometry, config)
        radial = geometry.r.astype(cp.complex128) ** 2
        w, z = self._single_mode_coeffs(geometry, 3, 0, radial)
        u = operators.pt.velocity_from_wz_coeffs(w, z)
        q = (
            geometry.l_values[None, :]
            * (geometry.l_values[None, :] + 1.0)
            * w
            / (geometry.r[:, None] ** 2)
        )
        ur_from_q = operators.spectral_to_grid(q)
        err = float(cp.max(cp.abs(u[0] - ur_from_q)).get())
        self.assertLess(err, 1e-10)

    def test_spat_to_qst_roundtrip(self):
        config = ShellConvectionConfig(l_max=6, nr=12, ra=1e4, ek=1e-3)
        geometry = ShellGeometry(config)
        operators = ShellOperators(geometry, config)
        rng = np.random.default_rng(1)
        w = cp.zeros((geometry.nr, geometry.n_lm), dtype=cp.complex128)
        z = cp.zeros_like(w)
        for l in range(1, geometry.l_max + 1):
            w[:, _lm_to_flat(l, 0)] = rng.normal()
            z[:, _lm_to_flat(l, 0)] = rng.normal()
        u = operators.pt.velocity_from_wz_coeffs(w, z)
        q, s, t = operators.pt.angular.spat_to_qst(u[0], u[1], u[2])
        r = geometry.r[:, None]
        q_exp = (
            geometry.l_values[None, :]
            * (geometry.l_values[None, :] + 1.0)
            * w
            / (r * r)
        )
        dw = operators.pt.angular.coeffs_dr(w)
        s_exp = dw / r
        t_exp = z / r
        mask = geometry.l_values[None, :] > 0
        err_q = float(cp.max(cp.abs((q - q_exp) * mask)).get())
        err_s = float(cp.max(cp.abs((s - s_exp) * mask)).get())
        err_t = float(cp.max(cp.abs((t - t_exp) * mask)).get())
        self.assertLess(err_q, 1e-8)
        self.assertLess(err_s, 1e-6)
        self.assertLess(err_t, 1e-6)

    def test_initial_velocity_near_divergence_free(self):
        config = ShellConvectionConfig(l_max=8, nr=16, ra=1e4, ek=1e-3)
        geometry = ShellGeometry(config)
        operators = ShellOperators(geometry, config)
        w, z, _ = ShellInitialCondition(geometry, perturbation_amplitude=1e-3).create()
        u = operators.pt.velocity_from_wz_coeffs(w, z)
        div = operators.max_divergence(w, z)
        max_u = float(cp.max(cp.sqrt(cp.sum(cp.real(u) ** 2, axis=0))).get())
        rel_div = div / max(max_u, 1e-12)
        self.assertLess(rel_div, 1e-6)

    def test_temperature_bc_inner_outer(self):
        config = ShellConvectionConfig(l_max=4, nr=8, ra=1e4, ek=1e-3)
        geometry = ShellGeometry(config)
        w, z, theta = ShellInitialCondition(geometry).create()
        theta_grid = geometry.sht.inverse_radial_stack(theta)
        self.assertLess(float(cp.max(cp.abs(theta_grid[0])).get()), 1e-8)
        self.assertLess(float(cp.max(cp.abs(theta_grid[-1] - 1.0)).get()), 1e-8)

    def test_short_run_finite(self):
        config = ShellConvectionConfig(
            l_max=4,
            nr=8,
            ra=1e4,
            ek=1e-3,
            dt=1e-2,
            total_time=0.02,
            save_every_time=0.01,
        )
        simulation = ShellConvectionSimulation(config, make_plots=False)
        w, z, theta, diagnostics = simulation.run()
        self.assertTrue(bool(cp.all(cp.isfinite(w))))
        self.assertTrue(bool(cp.all(cp.isfinite(z))))
        self.assertTrue(bool(cp.all(cp.isfinite(theta))))
        self.assertGreater(len(diagnostics.times), 1)
        self.assertGreater(diagnostics.nusselt_inner[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
