import argparse

from .config import TaylorGreenConfig
from .simulation import TaylorGreenSimulation


def main():
    parser = argparse.ArgumentParser(description="3D Taylor-Green vortex with CuPy/CUDA")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--reynolds", type=float, default=1000.0)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--save-every-time", type=float, default=0.05)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config = TaylorGreenConfig(
        n=args.n,
        dt=args.dt,
        reynolds=args.reynolds,
        total_time=args.total_time,
        save_every_time=args.save_every_time,
    )
    TaylorGreenSimulation(config, make_plots=not args.no_plots).run()
