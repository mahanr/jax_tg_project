"""Boundary condition enforcement for velocity and temperature.

Chebyshev-Gauss-Lobatto ordering: index 0 is the outer boundary r=1,
index -1 is the inner boundary r=η.
"""


def apply_velocity_bc_wz(w_grid, z_grid):
    """No-slip: W=Z=0 on inner and outer radial shells (dW/dr imposed in IMEX)."""
    w_grid[0] = 0.0
    w_grid[-1] = 0.0
    z_grid[0] = 0.0
    z_grid[-1] = 0.0
    return w_grid, z_grid


def apply_temperature_bc(theta_grid, inner_value=1.0, outer_value=0.0):
    """Fixed temperature: θ=inner at r=η (index -1), θ=outer at r=1 (index 0)."""
    theta_grid[0] = outer_value
    theta_grid[-1] = inner_value
    return theta_grid


def enforce_wz_bc_on_grid(w_grid, z_grid):
    return apply_velocity_bc_wz(w_grid, z_grid)


def enforce_theta_bc_on_grid(theta_grid, inner=1.0, outer=0.0):
    return apply_temperature_bc(theta_grid, inner, outer)
