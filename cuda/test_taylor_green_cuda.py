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
    def test_spectral_step_preserves_finite_values(self):
        grid = SpectralGrid(8, LENGTH)
        operator = CupySpectralOperator(grid)
        integrator = RK4Integrator(operator, 0.005, LENGTH / 100.0)
        velocity_hat = grid.to_spectral(TaylorGreenInitialCondition(LENGTH).create(8))
        velocity_hat = integrator.step(velocity_hat)
        self.assertTrue(bool(cp.all(cp.isfinite(velocity_hat))))

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
        self.assertEqual(len(diagnostics.times), 3)


if __name__ == "__main__":
    unittest.main()
