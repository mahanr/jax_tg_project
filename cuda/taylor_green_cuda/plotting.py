import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp


def save_plots(state_hat, diagnostics, n, grid):
    slice_data = cp.asnumpy(grid.to_real(state_hat)[:, :, :, n // 2])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for component, axis in enumerate(axes):
        image = axis.imshow(slice_data[component].T, origin="lower", cmap="viridis")
        axis.set_title(f"u{component + 1} slice")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig("taylor_green_cuda_slice.png", dpi=200)
    plt.close(figure)

    figure, (energy_axis, enstrophy_axis) = plt.subplots(2, 1, figsize=(10, 8))
    energy_axis.plot(diagnostics.times, diagnostics.energies, "b-o")
    energy_axis.set_ylabel("Kinetic energy")
    energy_axis.grid(True, alpha=0.3)
    enstrophy_axis.plot(diagnostics.times, diagnostics.enstrophies, "r-s")
    enstrophy_axis.set_xlabel("Time")
    enstrophy_axis.set_ylabel("Enstrophy")
    enstrophy_axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig("taylor_green_cuda_diagnostics.png", dpi=200)
    plt.close(figure)
