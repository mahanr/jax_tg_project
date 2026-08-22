"""Compatibility entry point for the modular CuPy Taylor-Green MHD solver."""

import cupy as cp

from taylor_green_cuda.grid import SpectralGrid
from taylor_green_mhd.config import TaylorGreenMhdConfig
from taylor_green_mhd.initial_conditions import TaylorGreenMhdInitialCondition
from taylor_green_mhd.operators import MhdSpectralOperator
from taylor_green_mhd.simulation import TaylorGreenMhdSimulation


def make_wavenumbers(n, length):
    grid = SpectralGrid(n, length)
    return grid.kx, grid.ky, grid.kz


def make_dealias_mask(n, length):
    return SpectralGrid(n, length).dealias_mask


def initial_taylor_green_mhd(n, length=2.0 * cp.pi, magnetic_amplitude=0.5):
    return TaylorGreenMhdInitialCondition(length, magnetic_amplitude).create(n)


def _operator_from_grid(grid):
    return MhdSpectralOperator(grid)


def spectral_rhs_mhd(velocity, magnetic_field, viscosity, resistivity, kx, ky, kz, dealias_mask):
    grid = type("GridView", (), {})()
    grid.kx, grid.ky, grid.kz = kx, ky, kz
    grid.k_squared = kx * kx + ky * ky + kz * kz
    grid.dealias_mask = dealias_mask
    return MhdSpectralOperator(grid).rhs(velocity, magnetic_field, viscosity, resistivity)


def advance_one_step(velocity, magnetic_field, dt, viscosity, resistivity, kx, ky, kz, dealias_mask):
    from taylor_green_mhd.integrator import MhdRK4Integrator

    grid = type("GridView", (), {})()
    grid.kx, grid.ky, grid.kz = kx, ky, kz
    grid.k_squared = kx * kx + ky * ky + kz * kz
    grid.dealias_mask = dealias_mask
    operator = MhdSpectralOperator(grid)
    integrator = MhdRK4Integrator(operator, dt, viscosity, resistivity)
    return integrator.step((velocity, magnetic_field))


def kinetic_energy(velocity):
    return 0.5 * cp.mean(cp.sum(velocity * velocity, axis=0))


def magnetic_energy(magnetic_field):
    return 0.5 * cp.mean(cp.sum(magnetic_field * magnetic_field, axis=0))


def run_simulation(
    n=128,
    dt=0.005,
    reynolds=1000.0,
    magnetic_reynolds=1000.0,
    magnetic_amplitude=0.5,
    total_time=1.0,
    save_every_time=0.05,
    make_plots=True,
):
    config = TaylorGreenMhdConfig(
        n=n,
        dt=dt,
        reynolds=reynolds,
        magnetic_reynolds=magnetic_reynolds,
        magnetic_amplitude=magnetic_amplitude,
        total_time=total_time,
        save_every_time=save_every_time,
    )
    velocity, magnetic_field, diagnostics = TaylorGreenMhdSimulation(
        config,
        make_plots=make_plots,
    ).run()
    return velocity, magnetic_field, diagnostics


def main():
    from taylor_green_mhd.cli import main as package_main

    package_main()


if __name__ == "__main__":
    main()
