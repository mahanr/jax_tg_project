import math
import unittest

import cupy as cp

from taylor_green_cuda.grid import SpectralGrid
from taylor_green_mhd.config import (
    magnetic_reynolds_to_resistivity,
    resistivity_to_magnetic_reynolds,
    validate_timestep,
)
from taylor_green_mhd.initial_conditions import TaylorGreenMhdInitialCondition
from taylor_green_mhd.integrator import MhdRK4Integrator
from taylor_green_mhd.operators import MhdSpectralOperator
from taylor_green_mhd.simulation import TaylorGreenMhdSimulation
from taylor_green_mhd import TaylorGreenMhdConfig


LENGTH = 2.0 * math.pi


class TaylorGreenMhdTests(unittest.TestCase):
    def test_initial_condition_is_divergence_free(self):
        grid = SpectralGrid(16, LENGTH)
        operator = MhdSpectralOperator(grid)
        velocity, magnetic_field = TaylorGreenMhdInitialCondition(LENGTH, 0.5).create(16)
        velocity_hat = grid.to_spectral(velocity)
        magnetic_field_hat = grid.to_spectral(magnetic_field)
        div_u = operator.divergence_from_hat(velocity_hat)
        div_b = operator.divergence_from_hat(magnetic_field_hat)
        self.assertLess(float(cp.max(cp.abs(div_u)).get()), 1e-5)
        self.assertLess(float(cp.max(cp.abs(div_b)).get()), 1e-5)

    def test_dealias_mask_keeps_two_thirds_of_each_axis(self):
        grid = SpectralGrid(16, LENGTH)
        mask = grid.dealias_mask
        self.assertEqual(mask.shape, (16, 16, 16))
        self.assertEqual(int(mask.sum()), 11**3)
        self.assertFalse(bool(mask[8, 0, 0]))
        self.assertTrue(bool(mask[5, 0, 0]))

    def test_timestep_validation_rejects_unstable_values(self):
        grid = SpectralGrid(16, LENGTH)
        nu = LENGTH / 100.0
        eta = LENGTH / 100.0
        with self.assertRaisesRegex(ValueError, "stability limit"):
            validate_timestep(1.0, nu, eta, grid)
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_timestep(0.0, nu, eta, grid)

    def test_short_run_is_finite_and_divergence_free(self):
        config = TaylorGreenMhdConfig(
            n=8,
            dt=0.005,
            reynolds=100.0,
            magnetic_reynolds=100.0,
            magnetic_amplitude=0.5,
            total_time=0.02,
            save_every_time=0.01,
        )
        simulation = TaylorGreenMhdSimulation(config, make_plots=False)
        velocity_hat, magnetic_field_hat, diagnostics = simulation.run()
        self.assertTrue(bool(cp.all(cp.isfinite(velocity_hat))))
        self.assertTrue(bool(cp.all(cp.isfinite(magnetic_field_hat))))
        self.assertEqual(len(diagnostics.times), 3)
        self.assertLess(max(diagnostics.divergence_u_max), 1e-4)
        self.assertLess(max(diagnostics.divergence_b_max), 1e-4)

    def test_step_preserves_finite_values(self):
        grid = SpectralGrid(8, LENGTH)
        operator = MhdSpectralOperator(grid)
        integrator = MhdRK4Integrator(operator, 0.005, LENGTH / 100.0, LENGTH / 100.0)
        velocity, magnetic_field = TaylorGreenMhdInitialCondition(LENGTH, 0.5).create(8)
        velocity_hat = grid.to_spectral(velocity)
        magnetic_field_hat = grid.to_spectral(magnetic_field)
        velocity_hat, magnetic_field_hat = integrator.step((velocity_hat, magnetic_field_hat))
        self.assertTrue(bool(cp.all(cp.isfinite(velocity_hat))))
        self.assertTrue(bool(cp.all(cp.isfinite(magnetic_field_hat))))

    def test_magnetic_reynolds_conversion(self):
        rm = 100.0
        eta = magnetic_reynolds_to_resistivity(rm)
        rm_back = resistivity_to_magnetic_reynolds(eta)
        self.assertAlmostEqual(rm, rm_back, places=6)

    def test_magnetic_reynolds_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "magnetic_reynolds must be positive"):
            magnetic_reynolds_to_resistivity(-100.0)


if __name__ == "__main__":
    unittest.main()
