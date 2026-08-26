import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp


def save_plots(state_hat, diagnostics, n, grid, total_time=None):
    slice_path = os.path.abspath("taylor_green_cuda_slice.png")
    diagnostics_path = os.path.abspath("taylor_green_cuda_diagnostics.png")

    slice_data = cp.asnumpy(grid.to_real(state_hat)[:, :, :, n // 2])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for component, axis in enumerate(axes):
        image = axis.imshow(slice_data[component].T, origin="lower", cmap="viridis")
        axis.set_title(f"u{component + 1} slice")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(slice_path, dpi=200)
    plt.close(figure)

    times = diagnostics.times
    time_max = total_time if total_time is not None else times[-1]

    figure, (energy_axis, enstrophy_axis) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    energy_axis.plot(times, diagnostics.energies, "b-o", linewidth=2, markersize=4)
    energy_axis.set_ylabel("Kinetic energy")
    energy_axis.set_title("Energy decay")
    energy_axis.grid(True, alpha=0.3)
    enstrophy_axis.plot(times, diagnostics.enstrophies, "r-s", linewidth=2, markersize=4)
    enstrophy_axis.set_xlabel("Time")
    enstrophy_axis.set_ylabel("Enstrophy")
    enstrophy_axis.set_title("Enstrophy evolution")
    enstrophy_axis.grid(True, alpha=0.3)
    energy_axis.set_xlim(times[0], time_max)
    figure.tight_layout()
    figure.savefig(diagnostics_path, dpi=200)
    plt.close(figure)
    print(f"Wrote {slice_path}")
    print(f"Wrote {diagnostics_path}")
