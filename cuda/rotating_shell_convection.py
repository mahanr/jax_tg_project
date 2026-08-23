"""Compatibility shim for rotating shell convection solver."""
from rotating_shell_convection.config import ShellConvectionConfig
from rotating_shell_convection.simulation import ShellConvectionSimulation


def run_simulation(
    ra=1e6,
    pr=1.0,
    ek=1e-5,
    eta=0.5,
    l_max=63,
    nr=64,
    dt=1e-3,
    total_time=10.0,
    save_every_time=0.5,
    make_plots=True,
):
    config = ShellConvectionConfig(
        ra=ra,
        pr=pr,
        ek=ek,
        eta=eta,
        l_max=l_max,
        nr=nr,
        dt=dt,
        total_time=total_time,
        save_every_time=save_every_time,
    )
    simulation = ShellConvectionSimulation(config, make_plots=make_plots)
    return simulation.run()


def main():
    from rotating_shell_convection.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
