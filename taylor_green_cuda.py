"""Compatibility entry point for the modular CuPy CUDA solver."""

import cupy as cp

from taylor_green_cuda.config import TaylorGreenConfig
from taylor_green_cuda.grid import SpectralGrid
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


def _operator_from_arrays(kx, ky, kz, dealias_mask):
    grid = type("GridView", (), {})()
    grid.kx, grid.ky, grid.kz = kx, ky, kz
    grid.k_squared = kx * kx + ky * ky + kz * kz
    grid.dealias_mask = dealias_mask
    return CupySpectralOperator(grid)


def spectral_rhs(velocity, nu, kx, ky, kz, dealias_mask):
    return _operator_from_arrays(kx, ky, kz, dealias_mask).rhs(velocity, nu)


def advance_one_step(velocity, dt, nu, kx, ky, kz, dealias_mask):
    from taylor_green_cuda.integrator import RK4Integrator

    operator = _operator_from_arrays(kx, ky, kz, dealias_mask)
    return RK4Integrator(operator, dt, nu).step(velocity)


def kinetic_energy(velocity):
    return 0.5 * cp.mean(cp.sum(velocity * velocity, axis=0))


def enstrophy(velocity, kx, ky, kz):
    from taylor_green_cuda.diagnostics import Diagnostics

    operator = _operator_from_arrays(kx, ky, kz, cp.ones_like(kx, dtype=bool))
    diagnostics = Diagnostics(operator)
    return cp.asarray(diagnostics.record(0.0, velocity)[1])


def run_simulation(n=128, dt=0.005, reynolds=1000.0, total_time=1.0,
                   save_every_time=0.05):
    config = TaylorGreenConfig(n, dt, reynolds, total_time, save_every_time)
    state, diagnostics = TaylorGreenSimulation(config).run()
    return state, diagnostics.times, diagnostics.energies, diagnostics.enstrophies


def main():
    from taylor_green_cuda.cli import main as package_main

    package_main()


if __name__ == "__main__":
    main()
