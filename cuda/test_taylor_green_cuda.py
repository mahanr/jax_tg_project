import math
import unittest

import cupy as cp

from taylor_green_cuda.grid import SpectralGrid
from taylor_green_cuda.integrator import RK4Integrator
from taylor_green_cuda.operators import CupySpectralOperator
from taylor_green_cuda.initial_conditions import TaylorGreenInitialCondition
from taylor_green_cuda.simulation import TaylorGreenSimulation
from taylor_green_cuda.config import TaylorGreenConfig


LENGTH = 2.0 * math.pi


class TaylorGreenCudaTests(unittest.TestCase):
    def test_viscosity_is_nondimensional(self):
        config = TaylorGreenConfig(reynolds=250.0)
        self.assertAlmostEqual(config.viscosity, 1.0 / 250.0)

    def test_spectral_step_preserves_finite_values(self):
        grid = SpectralGrid(8, LENGTH)
        operator = CupySpectralOperator(grid)
        integrator = RK4Integrator(operator, 0.005, 1.0 / 100.0)
        velocity_hat = grid.to_spectral(TaylorGreenInitialCondition(LENGTH).create(8))
        velocity_hat = integrator.step(velocity_hat)
        self.assertTrue(bool(cp.all(cp.isfinite(velocity_hat))))
        self.assertEqual(velocity_hat.dtype, cp.complex64)

    def test_short_run_is_finite(self):
        config = TaylorGreenConfig(
            n=8,
            dt=0.005,
            reynolds=100.0,
            total_time=0.02,
            save_every_time=0.01,
        )
        simulation = TaylorGreenSimulation(
            config,
            make_plots=False,
        )
        state_hat, diagnostics = simulation.run()
        self.assertTrue(bool(cp.all(cp.isfinite(state_hat))))
        self.assertEqual(state_hat.dtype, cp.complex64)
        self.assertEqual(len(diagnostics.times), 3)

    def test_project_and_diffuse_matches_unfused(self):
        grid = SpectralGrid(8, LENGTH)
        operator = CupySpectralOperator(grid)
        velocity_hat = grid.to_spectral(TaylorGreenInitialCondition(LENGTH).create(8))
        nonlinear_hat = (velocity_hat * cp.complex64(0.3 + 0.1j)).astype(velocity_hat.dtype, copy=False)
        viscosity = 1.0 / 100.0
        fused = operator.project_and_diffuse(nonlinear_hat, velocity_hat, viscosity)
        masked = nonlinear_hat * grid.dealias_mask
        safe_k_squared = cp.where(grid.k_squared == 0.0, 1.0, grid.k_squared)
        k_dot = grid.kx * masked[0] + grid.ky * masked[1] + grid.kz * masked[2]
        projected = masked - cp.stack((
            grid.kx * k_dot / safe_k_squared,
            grid.ky * k_dot / safe_k_squared,
            grid.kz * k_dot / safe_k_squared,
        ))
        reference = -projected - viscosity * grid.k_squared * velocity_hat
        self.assertTrue(bool(cp.allclose(fused, reference, rtol=1e-6, atol=1e-8)))


if __name__ == "__main__":
    unittest.main()
