

from __future__ import annotations

from dataclasses import dataclass
import argparse
from typing import Callable, List

import numpy as np


@dataclass
class LMIResult:
    """Result of Algorithm 1 line 2: solve target function and LMIs."""

    feasible: bool
    eta_t_f: float
    target_value: float
    message: str = ""


@dataclass
class IterationRecord:
    k: int
    sigma: float
    h: float
    pi: float
    sigma_next: float
    h_next: float
    pi_next: float
    feasible: bool
    target_value: float
    sigma_max: float


def regularization_l(sigma: float, rho: float) -> float:
    """L(sigma) = rho / 2 * ||sigma||_2^2 for scalar sigma."""

    return 0.5 * rho * sigma * sigma


def demo_lmi_oracle(
    sigma: float,
    *,
    feasible_threshold: float,
    rho: float,
) -> LMIResult:


    margin = feasible_threshold - sigma
    feasible = margin >= -1e-12

    # Algorithm line 3 uses eta^T F. This demo uses the signed distance from
    # the feasible boundary so the ADA update moves sigma toward sigma_max.
    eta_t_f = sigma - feasible_threshold
    return LMIResult(
        feasible=feasible,
        eta_t_f=eta_t_f,
        target_value=regularization_l(sigma, rho),
        message="demo LMI oracle",
    )


def update_h(
    sigma: float,
    pi: float,
    ell: float,
    *,
    mode: str,
) -> float:
    """
    Compute h^{k+1}.

    The algorithm box writes an argmax of a positive quadratic. Over all real
    numbers this is unbounded, so a runnable threshold implementation uses the
    natural threshold domain h in [0, 1].
    """

    center = sigma - pi / ell

    if mode == "bounded-argmax":
        distance_to_zero = abs(0.0 - center)
        distance_to_one = abs(1.0 - center)
        return 0.0 if distance_to_zero >= distance_to_one else 1.0

    if mode == "unconstrained":
        raise ValueError(
            "Line 4 is unbounded without a constraint on h. Use bounded-argmax."
        )

    raise ValueError(f"Unknown h update mode: {mode}")


def improved_alternate_direction_algorithm(
    *,
    sigma0: float,
    ell: float,
    rho: float,
    max_iter: int,
    tol: float,
    h_update_mode: str,
    lmi_oracle: Callable[[float], LMIResult],
) -> tuple[float, List[IterationRecord]]:

    if not 0.0 <= sigma0 <= 1.0:
        raise ValueError("sigma0 must be inside [0, 1].")
    if ell <= 0.0:
        raise ValueError("ell must be positive.")
    if rho <= 0.0:
        raise ValueError("rho must be positive.")

    sigma = float(sigma0)
    h = float(sigma0)
    pi = 0.0
    sigma_max = np.nan
    records: List[IterationRecord] = []

    for k in range(max_iter):
        lmi = lmi_oracle(sigma)

        if lmi.feasible:
            sigma_max = sigma if np.isnan(sigma_max) else max(sigma_max, sigma)

        sigma_next = (ell * h + pi - lmi.eta_t_f) / ell
        sigma_next = float(np.clip(sigma_next, 0.0, 1.0))
        h_next = update_h(sigma_next, pi, ell, mode=h_update_mode)
        pi_next = float(pi + ell * (h_next - sigma_next))

        records.append(
            IterationRecord(
                k=k,
                sigma=sigma,
                h=h,
                pi=pi,
                sigma_next=sigma_next,
                h_next=h_next,
                pi_next=pi_next,
                feasible=lmi.feasible,
                target_value=lmi.target_value,
                sigma_max=float(sigma_max) if not np.isnan(sigma_max) else np.nan,
            )
        )

        primal_residual = abs(h_next - sigma_next)
        sigma_change = abs(sigma_next - sigma)
        if sigma_change <= tol and primal_residual <= tol:
            break

        sigma, h, pi = sigma_next, h_next, pi_next

    if np.isnan(sigma_max):
        raise RuntimeError("No feasible trigger threshold was found.")

    return float(sigma_max), records


def print_records(records: List[IterationRecord]) -> None:
    print("k | sigma_k   | h_k       | pi_k      | feasible | target L(sigma) | sigma_max")
    print("--+-----------+-----------+-----------+----------+-----------------+----------")
    for row in records:
        feasible = "yes" if row.feasible else "no"
        print(
            f"{row.k:1d} | "
            f"{row.sigma:9.6f} | "
            f"{row.h:9.6f} | "
            f"{row.pi:9.6f} | "
            f"{feasible:^8s} | "
            f"{row.target_value:15.8f} | "
            f"{row.sigma_max:8.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the improved ADA trigger-threshold optimizer."
    )
    parser.add_argument("--sigma0", type=float, default=0.2, help="Initial sigma in [0, 1].")
    parser.add_argument("--ell", type=float, default=1.0, help="Positive scalar ell.")
    parser.add_argument("--rho", type=float, default=1.0, help="L2 regularization weight.")
    parser.add_argument(
        "--feasible-threshold",
        type=float,
        default=0.7,
        help="Demo LMI maximum feasible sigma. Replace oracle for a real LMI solver.",
    )
    parser.add_argument("--max-iter", type=int, default=50, help="Maximum ADA iterations.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Stopping tolerance.")
    parser.add_argument(
        "--h-update-mode",
        choices=("bounded-argmax", "unconstrained"),
        default="bounded-argmax",
        help="How to solve the h update from line 4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    oracle = lambda sigma: demo_lmi_oracle(
        sigma,
        feasible_threshold=args.feasible_threshold,
        rho=args.rho,
    )

    sigma_max, records = improved_alternate_direction_algorithm(
        sigma0=args.sigma0,
        ell=args.ell,
        rho=args.rho,
        max_iter=args.max_iter,
        tol=args.tol,
        h_update_mode=args.h_update_mode,
        lmi_oracle=oracle,
    )

    print_records(records)
    print()
    print(f"Optimal trigger threshold sigma_max = {sigma_max:.8f}")
    print(f"Regularization L(sigma_max) = {regularization_l(sigma_max, args.rho):.8f}")


if __name__ == "__main__":
    main()
