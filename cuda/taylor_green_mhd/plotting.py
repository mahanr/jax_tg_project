import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp


def save_plots(velocity, magnetic_field, diagnostics, n):
    velocity_slice = cp.asnumpy(velocity[:, :, :, n // 2])
    magnetic_slice = cp.asnumpy(magnetic_field[:, :, :, n // 2])

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for component in range(3):
        image = axes[0, component].imshow(velocity_slice[component].T, origin="lower", cmap="viridis")
        axes[0, component].set_title(f"u{component + 1} slice")
        figure.colorbar(image, ax=axes[0, component])
        image = axes[1, component].imshow(magnetic_slice[component].T, origin="lower", cmap="plasma")
        axes[1, component].set_title(f"B{component + 1} slice")
        figure.colorbar(image, ax=axes[1, component])
    figure.tight_layout()
    figure.savefig("taylor_green_mhd_slice.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(diagnostics.times, diagnostics.kinetic_energies, "b-o", markersize=4)
    axes[0, 0].set_ylabel("Kinetic energy")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(diagnostics.times, diagnostics.magnetic_energies, "r-o", markersize=4)
    axes[0, 1].set_ylabel("Magnetic energy")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(diagnostics.times, diagnostics.total_energies, "g-o", markersize=4)
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Total energy")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(diagnostics.times, diagnostics.cross_helicities, "m-s", markersize=4)
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Cross helicity")
    axes[1, 1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig("taylor_green_mhd_diagnostics.png", dpi=200)
    plt.close(figure)
