"""Compatibility entry point for the modular CuPy CUDA solver."""

import cupy as cp

from taylor_green_cuda.config import TaylorGreenConfig
from taylor_green_cuda.grid import SpectralGrid, SpectralGridView
from taylor_green_cuda.initial_conditions import TaylorGreenInitialCondition
from taylor_green_cuda.operators import CupySpectralOperator
from taylor_green_cuda.simulation import TaylorGreenSimulation


def make_wavenumbers(n, length):
    grid = SpectralGrid(n, length)
    return grid.kx, grid.ky, grid.kz


def make_dealias_mask(n, length):
    return SpectralGrid(n, length).dealias_mask


def initial_taylor_green(n, length=2.0 * cp.pi):
    return TaylorGreenInitialCondition(length).create(n)


def initial_taylor_green_spectral(n, length=2.0 * cp.pi):
    grid = SpectralGrid(n, length)
    return grid.to_spectral(TaylorGreenInitialCondition(length).create(n))


def _grid_from_arrays(kx, ky, kz, dealias_mask):
    return SpectralGridView(kx, ky, kz, dealias_mask)


def spectral_rhs(velocity_hat, nu, kx, ky, kz, dealias_mask):
    return CupySpectralOperator(_grid_from_arrays(kx, ky, kz, dealias_mask)).rhs(
        velocity_hat, nu
    )


def advance_one_step(velocity_hat, dt, nu, kx, ky, kz, dealias_mask):
    from taylor_green_cuda.integrator import RK4Integrator

    grid = _grid_from_arrays(kx, ky, kz, dealias_mask)
    operator = CupySpectralOperator(grid)
    integrator = RK4Integrator(operator, dt, nu)
    return grid.dealias(integrator.step(velocity_hat))


def kinetic_energy(velocity):
    return 0.5 * cp.mean(cp.sum(velocity * velocity, axis=0))


def enstrophy(velocity, kx, ky, kz):
    from taylor_green_cuda.diagnostics import Diagnostics

    operator = CupySpectralOperator(_grid_from_arrays(kx, ky, kz, cp.ones_like(kx, dtype=bool)))
    grid = SpectralGrid(int(kx.shape[0]), 2.0 * cp.pi)
    state_hat = grid.to_spectral(velocity)
    diagnostics = Diagnostics(operator)
    return cp.asarray(diagnostics.record(0.0, state_hat)[1])


def run_simulation(n=128, dt=0.005, reynolds=1000.0, total_time=1.0,
                   save_every_time=0.05):
    config = TaylorGreenConfig(n, dt, reynolds, total_time, save_every_time)
    simulation = TaylorGreenSimulation(config)
    state_hat, diagnostics = simulation.run()
    grid = simulation.grid
    return grid.to_real(state_hat), diagnostics.times, diagnostics.energies, diagnostics.enstrophies


def main():
    from taylor_green_cuda.cli import main as package_main

    package_main()


if __name__ == "__main__":
    main()
