"""CuPy/CUDA 3D Taylor-Green vortex solver.

Install a CuPy wheel matching the installed NVIDIA CUDA Toolkit, for example:
    python -m pip install cupy-cuda12x matplotlib
"""

import argparse
import time

import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PI = cp.pi


def make_wavenumbers(n, length):
    k = 2.0 * cp.pi * cp.fft.fftfreq(n, d=length / n)
    return cp.meshgrid(k, k, k, indexing="ij")


def make_dealias_mask(n, length):
    k = 2.0 * cp.pi * cp.fft.fftfreq(n, d=length / n)
    cutoff = (n // 3) * (2.0 * cp.pi / length)
    keep = cp.abs(k) <= cutoff
    return keep[:, None, None] & keep[None, :, None] & keep[None, None, :]


def initial_taylor_green(n, length=2.0 * cp.pi):
    x = cp.linspace(0.0, length, n, endpoint=False, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, x, x, indexing="ij")
    u = cp.sin(X) * cp.cos(Y) * cp.cos(Z)
    v = -cp.cos(X) * cp.sin(Y) * cp.cos(Z)
    return cp.stack((u, v, cp.zeros_like(u)))


def spectral_rhs(velocity, nu, kx, ky, kz, dealias_mask):
    velocity_hat = cp.fft.fftn(velocity, axes=(-3, -2, -1))
    velocity_hat *= dealias_mask

    gradients = []
    for component in range(3):
        component_hat = velocity_hat[component]
        gradients.append(tuple(
            cp.fft.ifftn(1j * wave * component_hat,
                         axes=(-3, -2, -1)).real
            for wave in (kx, ky, kz)
        ))

    nonlinear = cp.stack(tuple(
        sum(velocity[direction] * gradients[component][direction]
            for direction in range(3))
        for component in range(3)
    ))
    nonlinear_hat = cp.fft.fftn(nonlinear, axes=(-3, -2, -1))
    nonlinear_hat *= dealias_mask

    k_squared = kx * kx + ky * ky + kz * kz
    safe_k_squared = cp.where(k_squared == 0.0, 1.0, k_squared)
    k_dot_nonlinear = (
        kx * nonlinear_hat[0]
        + ky * nonlinear_hat[1]
        + kz * nonlinear_hat[2]
    )
    projected = nonlinear_hat - cp.stack((
        kx * k_dot_nonlinear / safe_k_squared,
        ky * k_dot_nonlinear / safe_k_squared,
        kz * k_dot_nonlinear / safe_k_squared,
    ))
    rhs_hat = -projected - nu * k_squared * velocity_hat
    return cp.fft.ifftn(rhs_hat, axes=(-3, -2, -1)).real


def advance_one_step(velocity, dt, nu, kx, ky, kz, dealias_mask):
    rhs_1 = spectral_rhs(velocity, nu, kx, ky, kz, dealias_mask)
    rhs_2 = spectral_rhs(velocity + 0.5 * dt * rhs_1, nu, kx, ky, kz, dealias_mask)
    rhs_3 = spectral_rhs(velocity + 0.5 * dt * rhs_2, nu, kx, ky, kz, dealias_mask)
    rhs_4 = spectral_rhs(velocity + dt * rhs_3, nu, kx, ky, kz, dealias_mask)
    return velocity + (dt / 6.0) * (rhs_1 + 2.0 * rhs_2 + 2.0 * rhs_3 + rhs_4)


def kinetic_energy(velocity):
    return 0.5 * cp.mean(cp.sum(velocity * velocity, axis=0))


def enstrophy(velocity, kx, ky, kz):
    velocity_hat = cp.fft.fftn(velocity, axes=(-3, -2, -1))
    du_dx = cp.fft.ifftn(1j * kx * velocity_hat[0], axes=(-3, -2, -1)).real
    du_dy = cp.fft.ifftn(1j * ky * velocity_hat[0], axes=(-3, -2, -1)).real
    du_dz = cp.fft.ifftn(1j * kz * velocity_hat[0], axes=(-3, -2, -1)).real
    dv_dx = cp.fft.ifftn(1j * kx * velocity_hat[1], axes=(-3, -2, -1)).real
    dv_dy = cp.fft.ifftn(1j * ky * velocity_hat[1], axes=(-3, -2, -1)).real
    dv_dz = cp.fft.ifftn(1j * kz * velocity_hat[1], axes=(-3, -2, -1)).real
    dw_dx = cp.fft.ifftn(1j * kx * velocity_hat[2], axes=(-3, -2, -1)).real
    dw_dy = cp.fft.ifftn(1j * ky * velocity_hat[2], axes=(-3, -2, -1)).real
    dw_dz = cp.fft.ifftn(1j * kz * velocity_hat[2], axes=(-3, -2, -1)).real
    omega = cp.stack((dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy))
    return 0.5 * cp.mean(cp.sum(omega * omega, axis=0))


def run_simulation(n=128, dt=0.005, reynolds=1000.0, total_time=1.0,
                   save_every_time=0.05):
    if n < 4 or dt <= 0.0 or reynolds <= 0.0 or total_time <= 0.0:
        raise ValueError("n, dt, reynolds, and total_time must be positive")
    length = 2.0 * cp.pi
    nu = 1.0 * length / reynolds
    kx, ky, kz = make_wavenumbers(n, length)
    dealias_mask = make_dealias_mask(n, length)
    velocity = initial_taylor_green(n, length)
    steps = int(total_time / dt)
    save_every = max(1, int(save_every_time / dt))
    times = [0.0]
    energies = [float(kinetic_energy(velocity).get())]
    enstrophies = [float(enstrophy(velocity, kx, ky, kz).get())]
    start = time.perf_counter()

    for step in range(1, steps + 1):
        velocity = advance_one_step(velocity, dt, nu, kx, ky, kz, dealias_mask)
        if step % save_every == 0 or step == steps:
            times.append(step * dt)
            energy = float(kinetic_energy(velocity).get())
            value = float(enstrophy(velocity, kx, ky, kz).get())
            energies.append(energy)
            enstrophies.append(value)
            print(f"Save time step: {step}, time: {step * dt:.6f}, "
                  f"enstrophy: {value:.8e}")

    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    print(f"CUDA device: {device_name}")
    print(f"Reynolds number: {reynolds:.1f}")
    print(f"Simulated time: {total_time:.3f} ({steps} steps) in {elapsed:.3f} s")
    print(f"Energy trace: {energies[0]:.6f} -> {energies[-1]:.6f}")

    slice_data = cp.asnumpy(velocity[:, :, :, n // 2])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for component, axis in enumerate(axes):
        image = axis.imshow(slice_data[component].T, origin="lower", cmap="viridis")
        axis.set_title(f"u{component + 1} slice")
        fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig("taylor_green_cuda_slice.png", dpi=200)
    plt.close(fig)

    fig, (axis_energy, axis_enstrophy) = plt.subplots(2, 1, figsize=(10, 8))
    axis_energy.plot(times, energies, "b-o")
    axis_energy.set_ylabel("Kinetic energy")
    axis_energy.grid(True, alpha=0.3)
    axis_enstrophy.plot(times, enstrophies, "r-s")
    axis_enstrophy.set_xlabel("Time")
    axis_enstrophy.set_ylabel("Enstrophy")
    axis_enstrophy.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("taylor_green_cuda_diagnostics.png", dpi=200)
    plt.close(fig)
    return velocity, times, energies, enstrophies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--reynolds", type=float, default=1000.0)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--save-every-time", type=float, default=0.05)
    args = parser.parse_args()
    run_simulation(args.n, args.dt, args.reynolds, args.total_time,
                   args.save_every_time)


if __name__ == "__main__":
    main()
