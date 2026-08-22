from dataclasses import dataclass
import math

import cupy as cp


@dataclass(frozen=True)
class TaylorGreenMhdConfig:
    n: int = 128
    dt: float = 0.005
    reynolds: float = 1000.0
    magnetic_reynolds: float = 1000.0
    magnetic_amplitude: float = 0.5
    total_time: float = 1.0
    save_every_time: float = 0.05
    domain_length: float = 2.0 * math.pi

    def __post_init__(self):
        if self.n < 4:
            raise ValueError("n must be at least 4")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.reynolds <= 0.0:
            raise ValueError("reynolds must be positive")
        if self.magnetic_reynolds <= 0.0:
            raise ValueError("magnetic_reynolds must be positive")
        if self.magnetic_amplitude < 0.0:
            raise ValueError("magnetic_amplitude must be nonnegative")
        if self.total_time <= 0.0:
            raise ValueError("total_time must be positive")
        if self.save_every_time <= 0.0:
            raise ValueError("save_every_time must be positive")
        if self.domain_length <= 0.0:
            raise ValueError("domain_length must be positive")

    @property
    def viscosity(self):
        return self.domain_length / self.reynolds

    @property
    def resistivity(self):
        return self.domain_length / self.magnetic_reynolds

    @property
    def steps(self):
        return int(self.total_time / self.dt)

    @property
    def save_every(self):
        return max(1, int(self.save_every_time / self.dt))


def magnetic_reynolds_to_resistivity(
    magnetic_reynolds,
    velocity_amplitude=1.0,
    domain_length=2.0 * math.pi,
):
    if magnetic_reynolds <= 0.0:
        raise ValueError("magnetic_reynolds must be positive")
    return float((velocity_amplitude * domain_length) / magnetic_reynolds)


def resistivity_to_magnetic_reynolds(
    resistivity,
    velocity_amplitude=1.0,
    domain_length=2.0 * math.pi,
):
    if resistivity < 0.0:
        raise ValueError("resistivity must be nonnegative")
    if resistivity == 0.0:
        return float("inf")
    return float((velocity_amplitude * domain_length) / resistivity)


def validate_timestep(
    dt,
    viscosity,
    resistivity,
    grid,
    max_velocity=1.0,
    max_field=0.5,
    cfl=0.5,
):
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if viscosity < 0.0:
        raise ValueError("viscosity must be nonnegative")
    if resistivity < 0.0:
        raise ValueError("resistivity must be nonnegative")
    if cfl <= 0.0:
        raise ValueError("cfl must be positive")
    if max_velocity <= 0.0:
        raise ValueError("max_velocity must be positive")
    if max_field <= 0.0:
        raise ValueError("max_field must be positive")

    kx, ky, kz = grid.kx, grid.ky, grid.kz
    max_wavenumber = float(cp.max(cp.abs(cp.stack((kx, ky, kz)))))
    if max_wavenumber == 0.0:
        raise ValueError("wavenumber arrays must contain nonzero modes")

    grid_spacing = cp.pi / max_wavenumber
    advective_limit = cfl * grid_spacing / max_velocity
    alfven_limit = cfl * grid_spacing / max_field
    k2_max = float(cp.max(grid.k_squared))
    viscous_limit = cp.inf if viscosity == 0.0 else 2.0 / (viscosity * k2_max)
    resistive_limit = cp.inf if resistivity == 0.0 else 2.0 / (resistivity * k2_max)
    limit = float(cp.min(cp.asarray([
        advective_limit,
        alfven_limit,
        viscous_limit,
        resistive_limit,
    ])))
    if dt > limit:
        raise ValueError(
            f"dt={dt:g} exceeds stability limit {limit:g}; reduce dt or increase cfl"
        )
    return limit
