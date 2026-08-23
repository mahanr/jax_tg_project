"""Plot mid-shell slices and diagnostic time series."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp


def save_plots(w_coeffs, z_coeffs, theta_coeffs, diagnostics, geometry):
    theta_grid = geometry.sht.inverse_radial_stack(theta_coeffs)
    mid = geometry.nr // 2
    theta_slice = cp.asnumpy(theta_grid[mid])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    image = axes[0].imshow(theta_slice, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[0].set_title("Temperature slice (mid-radius)")
    figure.colorbar(image, ax=axes[0])

    axes[1].plot(diagnostics.times, diagnostics.kinetic_energies, label="KE")
    axes[1].plot(diagnostics.times, diagnostics.nusselt_inner, label="Nu inner")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("value")
    axes[1].legend()
    axes[1].set_title("Diagnostics")
    axes[1].set_xlim(0, diagnostics.times[-1] if diagnostics.times else 1.0)

    figure.tight_layout()
    figure.savefig("rotating_shell_convection_diagnostics.png", dpi=150)
    plt.close(figure)

    figure2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.imshow(theta_slice, origin="lower", aspect="auto", cmap="RdBu_r")
    ax2.set_title("rotating_shell_convection slice")
    figure2.tight_layout()
    figure2.savefig("rotating_shell_convection_slice.png", dpi=150)
    plt.close(figure2)
