"""Compatibility entry point for the modular CuPy Taylor-Green MHD solver."""

import cupy as cp

from taylor_green_cuda.grid import SpectralGrid
from taylor_green_mhd.config import TaylorGreenMhdConfig
from taylor_green_mhd.initial_conditions import TaylorGreenMhdInitialCondition
from taylor_green_mhd.operators import MhdSpectralOperator
from taylor_green_mhd.simulation import TaylorGreenMhdSimulation


def make_wavenumbers(n, length):
    """Create wavenumber arrays from SpectralGrid."""
    grid = SpectralGrid(n, length)
    return grid.kx, grid.ky, grid.kz


def make_dealias_mask(n, length):
    """Create dealiasing mask from SpectralGrid."""
    return SpectralGrid(n, length).dealias_mask


def initial_taylor_green_mhd(n, length=2.0 * cp.pi, magnetic_amplitude=0.5):
    """Initialize Taylor-Green MHD velocity and magnetic fields in real space."""
    return TaylorGreenMhdInitialCondition(length, magnetic_amplitude).create(n)


def initial_taylor_green_mhd_spectral(n, length=2.0 * cp.pi, magnetic_amplitude=0.5):
    """Initialize Taylor-Green MHD fields in spectral space."""
    grid = SpectralGrid(n, length)
    ic = TaylorGreenMhdInitialCondition(length, magnetic_amplitude)
    velocity, magnetic_field = ic.create(n)
    return grid.to_spectral(velocity), grid.to_spectral(magnetic_field)


def _grid_from_arrays(kx, ky, kz, dealias_mask):
    return SpectralGrid.from_arrays(kx, ky, kz, dealias_mask)


def spectral_rhs_mhd(
    velocity_hat, magnetic_field_hat, viscosity, resistivity, kx, ky, kz, dealias_mask
):
    """Compute RHS of spectral MHD equations in Fourier space."""
    return MhdSpectralOperator(_grid_from_arrays(kx, ky, kz, dealias_mask)).rhs(
        velocity_hat, magnetic_field_hat, viscosity, resistivity
    )


def advance_one_step(
    velocity_hat, magnetic_field_hat, dt, viscosity, resistivity, kx, ky, kz, dealias_mask
):
    """Advance MHD solution one timestep in spectral space."""
    from taylor_green_mhd.integrator import MhdRK4Integrator

    grid = _grid_from_arrays(kx, ky, kz, dealias_mask)
    operator = MhdSpectralOperator(grid)
    integrator = MhdRK4Integrator(operator, dt, viscosity, resistivity)
    velocity_hat, magnetic_field_hat = integrator.step((velocity_hat, magnetic_field_hat))
    return grid.dealias(velocity_hat), grid.dealias(magnetic_field_hat)


def kinetic_energy(velocity):
    """Compute kinetic energy density."""
    return 0.5 * cp.mean(cp.sum(velocity * velocity, axis=0))


def magnetic_energy(magnetic_field):
    """Compute magnetic energy density."""
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
    """Run Taylor-Green MHD simulation with specified parameters."""
    config = TaylorGreenMhdConfig(
        n=n,
        dt=dt,
        reynolds=reynolds,
        magnetic_reynolds=magnetic_reynolds,
        magnetic_amplitude=magnetic_amplitude,
        total_time=total_time,
        save_every_time=save_every_time,
    )
    simulation = TaylorGreenMhdSimulation(config, make_plots=make_plots)
    velocity_hat, magnetic_field_hat, diagnostics = simulation.run()
    grid = simulation.grid
    return (
        grid.to_real(velocity_hat),
        grid.to_real(magnetic_field_hat),
        diagnostics,
    )


def main():
    """Run CLI for Taylor-Green MHD."""
    from taylor_green_mhd.cli import main as package_main

    package_main()


if __name__ == "__main__":
    main()
