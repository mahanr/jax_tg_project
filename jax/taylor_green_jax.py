"""JAX-based 3D Taylor-Green vortex spectral solver."""
import os
import time

# Let XLA request GPU memory as needed instead of reserving a large pool at startup.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


def reynolds_to_viscosity(reynolds, velocity_amplitude=1.0, domain_length=2.0 * jnp.pi):
    """
    Convert Reynolds number to kinematic viscosity.

    Re = (u_amp * L) / nu  =>  nu = (u_amp * L) / Re

    Args:
        reynolds: Reynolds number
        velocity_amplitude: Characteristic velocity (default: 1.0)
        domain_length: Domain size (default: 2π)

    Returns:
        Kinematic viscosity nu
    """
    if reynolds <= 0:
        raise ValueError("Reynolds number must be positive")
    return float((velocity_amplitude * domain_length) / reynolds)


def viscosity_to_reynolds(nu, velocity_amplitude=1.0, domain_length=2.0 * jnp.pi):
    """
    Convert kinematic viscosity to Reynolds number.

    Args:
        nu: Kinematic viscosity
        velocity_amplitude: Characteristic velocity (default: 1.0)
        domain_length: Domain size (default: 2π)

    Returns:
        Reynolds number
    """
    if nu < 0.0:
        raise ValueError("Viscosity must be nonnegative")
    if nu == 0.0:
        return float('inf')
    return float((velocity_amplitude * domain_length) / nu)


def make_wavenumbers(N, L):
    """Generate wavenumber arrays for spectral computation."""
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=L / N)
    kx, ky, kz = jnp.meshgrid(k, k, k, indexing="ij")
    return kx, ky, kz


def make_dealias_mask(N, L):
    """Create dealiasing mask using 2/3 rule."""
    cutoff = (N // 3) * (2.0 * jnp.pi / L)
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=L / N)
    mask_1d = jnp.abs(k) <= cutoff
    return mask_1d[:, None, None] * mask_1d[None, :, None] * mask_1d[None, None, :]


def initial_taylor_green(N, L=2.0 * jnp.pi, amp=1.0):
    """Initialize velocity field with Taylor-Green vortex."""
    x = jnp.linspace(0.0, L, N, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")

    u = amp * jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z)
    v = -amp * jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z)
    w = jnp.zeros_like(u)

    return jnp.stack([u, v, w], axis=0)


def spectral_gradient(field_hat, kx, ky, kz):
    """Compute gradient in spectral space and return in physical space."""
    dfdx = jnp.fft.ifftn(1j * kx * field_hat, axes=(-3, -2, -1)).real
    dfdy = jnp.fft.ifftn(1j * ky * field_hat, axes=(-3, -2, -1)).real
    dfdz = jnp.fft.ifftn(1j * kz * field_hat, axes=(-3, -2, -1)).real
    return dfdx, dfdy, dfdz


def spectral_divergence(u, kx, ky, kz):
    """Compute velocity divergence using spectral derivatives."""
    u_hat = jnp.fft.fftn(u, axes=(-3, -2, -1))
    divergence_hat = 1j * (
        kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    )
    return jnp.fft.ifftn(divergence_hat, axes=(-3, -2, -1)).real


def vorticity(u, kx, ky, kz):
    """Compute vorticity from velocity field."""
    u_hat = jnp.fft.fftn(u, axes=(-3, -2, -1))
    du_dx, du_dy, du_dz = spectral_gradient(u_hat[0], kx, ky, kz)
    dv_dx, dv_dy, dv_dz = spectral_gradient(u_hat[1], kx, ky, kz)
    dw_dx, dw_dy, dw_dz = spectral_gradient(u_hat[2], kx, ky, kz)
    return jnp.stack([
        dw_dy - dv_dz,
        du_dz - dw_dx,
        dv_dx - du_dy,
    ], axis=0)


def enstrophy(u, kx, ky, kz):
    """Compute enstrophy (kinetic energy of vorticity)."""
    omega = vorticity(u, kx, ky, kz)
    return 0.5 * jnp.mean(jnp.sum(omega * omega, axis=0))


def viscous_dissipation(u, nu, kx, ky, kz):
    """Compute viscous dissipation rate."""
    u_hat = jnp.fft.fftn(u, axes=(-3, -2, -1))
    gradients_squared = jnp.sum(
        jnp.abs(1j * kx * u_hat) ** 2
        + jnp.abs(1j * ky * u_hat) ** 2
        + jnp.abs(1j * kz * u_hat) ** 2,
        axis=0,
    )
    normalization = u.shape[-1] ** 6
    return nu * jnp.sum(gradients_squared) / normalization


def validate_timestep(dt, nu, kx, ky, kz, cfl=0.5, max_velocity=1.0):
    """Validate timestep satisfies CFL and diffusive stability conditions."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if nu < 0.0:
        raise ValueError("nu must be nonnegative")
    if cfl <= 0.0:
        raise ValueError("cfl must be positive")
    if max_velocity <= 0.0:
        raise ValueError("max_velocity must be positive")

    max_wavenumber = float(jnp.max(jnp.abs(jnp.stack([kx, ky, kz]))))
    if max_wavenumber == 0.0:
        raise ValueError("wavenumber arrays must contain nonzero modes")
    grid_spacing = jnp.pi / max_wavenumber
    advective_limit = cfl * grid_spacing / max_velocity
    k2_max = float(jnp.max(kx * kx + ky * ky + kz * kz))
    diffusive_limit = jnp.inf if nu == 0.0 else 2.0 / (nu * k2_max)
    limit = float(jnp.minimum(advective_limit, diffusive_limit))
    if dt > limit:
        raise ValueError(
            f"dt={dt:g} exceeds stability limit {limit:g}; "
            "reduce dt or increase cfl"
        )
    return limit


def spectral_rhs(u, nu, kx, ky, kz, dealias_mask):
    """Compute right-hand side of spectral Navier-Stokes."""
    u_hat = jnp.fft.fftn(u, axes=(-3, -2, -1))
    u_hat = u_hat * dealias_mask
    k2 = kx * kx + ky * ky + kz * kz
    k2_safe = jnp.where(k2 == 0.0, 1.0, k2)

    ux, uy, uz = u
    du_dx, du_dy, du_dz = spectral_gradient(u_hat[0], kx, ky, kz)
    dv_dx, dv_dy, dv_dz = spectral_gradient(u_hat[1], kx, ky, kz)
    dw_dx, dw_dy, dw_dz = spectral_gradient(u_hat[2], kx, ky, kz)

    nlin = jnp.stack([
        ux * du_dx + uy * du_dy + uz * du_dz,
        ux * dv_dx + uy * dv_dy + uz * dv_dz,
        ux * dw_dx + uy * dw_dy + uz * dw_dz,
    ], axis=0)

    nlin_hat = jnp.fft.fftn(nlin, axes=(-3, -2, -1)) * dealias_mask
    kdotn = kx * nlin_hat[0] + ky * nlin_hat[1] + kz * nlin_hat[2]
    nlin_proj = nlin_hat - jnp.stack([
        kx * kdotn / k2_safe,
        ky * kdotn / k2_safe,
        kz * kdotn / k2_safe,
    ], axis=0)

    rhs_hat = -nlin_proj - nu * k2 * u_hat
    return jnp.fft.ifftn(rhs_hat, axes=(-3, -2, -1)).real


@jax.jit
def advance_one_step(u, dt, nu, kx, ky, kz, dealias_mask):
    """Advance solution one timestep using RK4 integration."""
    # RK4 (4th-order Runge-Kutta) keeps each nonlinear evaluation projected and dealiased.
    rhs_1 = spectral_rhs(u, nu, kx, ky, kz, dealias_mask)
    rhs_2 = spectral_rhs(u + 0.5 * dt * rhs_1, nu, kx, ky, kz, dealias_mask)
    rhs_3 = spectral_rhs(u + 0.5 * dt * rhs_2, nu, kx, ky, kz, dealias_mask)
    rhs_4 = spectral_rhs(u + dt * rhs_3, nu, kx, ky, kz, dealias_mask)
    return u + (dt / 6.0) * (rhs_1 + 2.0 * rhs_2 + 2.0 * rhs_3 + rhs_4)


def kinetic_energy(u):
    """Compute kinetic energy density."""
    return 0.5 * jnp.mean(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)


def run_simulation(
    N=16,
    dt=0.005,
    nu=None,
    reynolds=None,
    total_time=0.5,
    save_every_time=0.05,
    cfl=0.5,
    return_diagnostics=False,
):
    """
    Run 3D decaying Taylor-Green vortex simulation.

    Args:
        N: Grid resolution (default: 16)
        dt: Timestep (default: 0.005)
        nu: Kinematic viscosity (default: computed from reynolds=100 if reynolds not specified)
        reynolds: Reynolds number; if provided, nu is computed from it
        total_time: Total diffusion time to simulate (default: 0.5)
        save_every_time: Save diagnostics every this much physical time (default: 0.05)
        cfl: CFL number for timestep validation (default: 0.5)
        return_diagnostics: If True, return detailed diagnostics dict (default: False)

    Returns:
        (u, energies) if return_diagnostics=False
        (u, energies, diagnostics) if return_diagnostics=True
    """
    if N < 4 or total_time <= 0.0 or save_every_time <= 0.0:
        raise ValueError(
            "N must be at least 4, total_time and save_every_time must be positive"
        )
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    n_steps = int(total_time / dt)
    save_every = max(1, int(save_every_time / dt))

    L = 2.0 * jnp.pi
    velocity_amplitude = 1.0

    # Determine viscosity from either reynolds or nu
    if reynolds is not None:
        nu = reynolds_to_viscosity(reynolds, velocity_amplitude, L)
    elif nu is None:
        # Default to Re=100
        reynolds = 100
        nu = reynolds_to_viscosity(reynolds, velocity_amplitude, L)

    # Compute actual Reynolds number for reporting
    actual_reynolds = viscosity_to_reynolds(nu, velocity_amplitude, L)
    kx, ky, kz = make_wavenumbers(N, L)
    dealias_mask = make_dealias_mask(N, L)
    u = initial_taylor_green(N, L=L, amp=1.0)
    validate_timestep(dt, nu, kx, ky, kz, cfl, max_velocity=float(jnp.max(jnp.abs(u))))

    times = [0.0]
    energies = [float(kinetic_energy(u))]
    divergence_max = [float(jnp.max(jnp.abs(spectral_divergence(u, kx, ky, kz))))]
    enstrophies = [float(enstrophy(u, kx, ky, kz))]
    dissipations = [float(viscous_dissipation(u, nu, kx, ky, kz))]
    t0 = time.time()

    for step in range(n_steps):
        u = advance_one_step(u, dt, nu, kx, ky, kz, dealias_mask)
        if (step + 1) % save_every == 0 or step == n_steps - 1:
            times.append((step + 1) * dt)
            energies.append(float(kinetic_energy(u)))
            div_max = float(jnp.max(jnp.abs(spectral_divergence(u, kx, ky, kz))))
            divergence_max.append(div_max)
            enstrophies.append(float(enstrophy(u, kx, ky, kz)))
            dissipations.append(float(viscous_dissipation(u, nu, kx, ky, kz)))
            print(f"Save time step: {step + 1}, Enstrophy: {enstrophies[-1]:.6e}")

    elapsed = time.time() - t0
    print(f"Reynolds number: Re = {actual_reynolds:.1f}")
    print(f"Simulated time: {total_time:.3f} ({n_steps} steps) in {elapsed:.3f} s")
    print(f"Energy trace: {energies[0]:.6f} -> {energies[-1]:.6f}")
    print(f"Max divergence: {max(divergence_max):.3e}")

    sl = u[:, :, N // 2]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        im = ax.imshow(sl[i].T, origin="lower", cmap="viridis")
        ax.set_title(f"u{i+1} slice")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("taylor_green_slice.png", dpi=200)
    print("Saved: taylor_green_slice.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(times, energies, "b-o", linewidth=2, markersize=4)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Kinetic Energy")
    ax1.set_title("Energy Decay")
    ax1.grid(True, alpha=0.3)
    ax2.plot(times, enstrophies, "r-s", linewidth=2, markersize=4)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Enstrophy")
    ax2.set_title("Enstrophy Evolution (Turbulence Growth/Decay)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("taylor_green_diagnostics.png", dpi=200)
    print("Saved: taylor_green_diagnostics.png")

    diagnostics = {
        "time": times,
        "energy": energies,
        "divergence_max": divergence_max,
        "enstrophy": enstrophies,
        "dissipation": dissipations,
    }
    if return_diagnostics:
        return u, energies, diagnostics
    return u, energies


if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    if not jax.devices("gpu"):
        print(
            "WARNING: No GPU detected by JAX. Check CUDA + driver setup before "
            "scaling up."
        )
    run_simulation(
        N=256,
        dt=0.005,
        reynolds=1000,
        total_time=10 * 2 * jnp.pi * 1000,
        save_every_time=0.05
    )
