"""CLI for rotating shell convection solver."""
import argparse

from .config import ShellConvectionConfig
from .simulation import ShellConvectionSimulation


def build_parser():
    parser = argparse.ArgumentParser(
        description="Rotating Boussinesq convection in a spherical shell (CuPy)"
    )
    parser.add_argument("--ra", type=float, default=1e6)
    parser.add_argument("--pr", type=float, default=1.0)
    parser.add_argument("--ek", type=float, default=1e-5)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--l-max", type=int, default=63)
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--total-time", type=float, default=10.0)
    parser.add_argument("--save-every-time", type=float, default=0.5)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = ShellConvectionConfig(
        ra=args.ra,
        pr=args.pr,
        ek=args.ek,
        eta=args.eta,
        l_max=args.l_max,
        nr=args.nr,
        dt=args.dt,
        total_time=args.total_time,
        save_every_time=args.save_every_time,
    )
    simulation = ShellConvectionSimulation(config, make_plots=not args.no_plots)
    simulation.run()


if __name__ == "__main__":
    main()
