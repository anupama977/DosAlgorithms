"""
Performance Analysis and Controller Matrix (Algorithm 2).

Algorithm 2 searches for the maximum acceptable time delay tau_max and then
computes the controller gain matrix K = Y X^{-1}.

The algorithm box defines the loop structure and the output rule, but the
actual LMI matrices are system-specific. This file implements the loop exactly
as a runnable Python structure and keeps the LMI feasibility calculation as a
replaceable oracle.

Run:
    python performance_controller_matrix.py

With the bundled Codex Python in this workspace:
    C:\\Users\\anupa\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe performance_controller_matrix.py
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from typing import Callable, List

import numpy as np


@dataclass
class Algorithm2Input:
    sigma_max: float
    h: float
    theta: float
    epsilon: float
    kappa: float


@dataclass
class LMIResult:
    feasible: bool
    matrix_block_positive: bool
    summation_negative: bool
    derivative_negative: bool
    y: np.ndarray
    x: np.ndarray
    message: str = ""


@dataclass
class SearchRecord:
    p: int
    eta_i: float
    tau_candidate: float
    feasible: bool
    tau_max: float


def demo_lmi_oracle(
    tau_candidate: float,
    inputs: Algorithm2Input,
    *,
    delay_limit: float,
) -> LMIResult:
    """
    Demonstration replacement for Algorithm 2 lines 5 to 7.

    The real version should build and solve:
        [M Z; * M] > 0
        sum(Lambda, Omega) = Xi + Pi_2.T @ Omega @ Pi_2 <= 0
        dV_it/dt <= sum(Lambda, Omega) <= 0

    This demo marks candidates as feasible until `delay_limit`.
    """

    feasible = tau_candidate <= delay_limit + 1e-12

    # Demo controller matrices. In a real LMI solver, Y and X are decision
    # variables returned by the feasible SDP solution.
    x = np.eye(4)
    y = np.array([[0.0195, 0.0052, -0.0062, -0.0244]])

    return LMIResult(
        feasible=feasible,
        matrix_block_positive=feasible,
        summation_negative=feasible,
        derivative_negative=feasible,
        y=y,
        x=x,
        message="demo LMI oracle",
    )


def compute_controller_gain(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute K = Y X^{-1}."""

    return y @ np.linalg.inv(x)


def performance_analysis_and_controller_matrix(
    inputs: Algorithm2Input,
    *,
    upper_limit_m: int,
    eta_step: float,
    eta_stop: float,
    lmi_oracle: Callable[[float, Algorithm2Input], LMIResult],
) -> tuple[float, np.ndarray, List[SearchRecord]]:
    """
    Run Algorithm 2 and return tau_max, K, and the search records.
    """

    if not 0.0 <= inputs.sigma_max <= 1.0:
        raise ValueError("sigma_max must be inside [0, 1].")
    if upper_limit_m <= 0:
        raise ValueError("upper_limit_m must be positive.")
    if eta_step <= 0.0:
        raise ValueError("eta_step must be positive.")
    if eta_stop <= 0.0:
        raise ValueError("eta_stop must be positive.")

    p = 0
    tau_i = 0.0
    tau_max = 0.0
    best_k: np.ndarray | None = None
    records: List[SearchRecord] = []

    eta_values = np.arange(0.0, eta_stop + eta_step / 2.0, eta_step)

    while p < upper_limit_m:
        advanced = False

        for eta_i in eta_values:
            tau_candidate = tau_i + float(eta_i)
            lmi = lmi_oracle(tau_candidate, inputs)

            records.append(
                SearchRecord(
                    p=p,
                    eta_i=float(eta_i),
                    tau_candidate=tau_candidate,
                    feasible=lmi.feasible,
                    tau_max=tau_max,
                )
            )

            if lmi.feasible:
                if eta_i <= 0.0:
                    continue

                if tau_candidate > tau_max:
                    tau_max = tau_candidate
                    tau_i = tau_candidate
                    best_k = compute_controller_gain(lmi.y, lmi.x)
                    p += 1
                    advanced = True
                break

            if best_k is None:
                raise RuntimeError("No feasible controller matrix was found.")

            return tau_max, best_k, records

        if not advanced:
            break

    if best_k is None:
        raise RuntimeError("No feasible controller matrix was found.")

    return tau_max, best_k, records


def print_records(records: List[SearchRecord], *, max_rows: int) -> None:
    print("p | eta_i    | tau_candidate | feasible | tau_max")
    print("--+----------+---------------+----------+----------")
    indexed_records = list(enumerate(records))
    if len(records) <= max_rows:
        rows_to_print = indexed_records
    else:
        head_count = max_rows // 2
        tail_count = max_rows - head_count
        rows_to_print = indexed_records[:head_count] + indexed_records[-tail_count:]

    previous_index = -1
    for original_index, row in rows_to_print:
        if previous_index != -1 and original_index != previous_index + 1:
            print(f"... {original_index - previous_index - 1} search rows omitted")
        previous_index = original_index
        feasible = "yes" if row.feasible else "no"
        print(
            f"{row.p:1d} | "
            f"{row.eta_i:8.4f} | "
            f"{row.tau_candidate:13.4f} | "
            f"{feasible:^8s} | "
            f"{row.tau_max:8.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Algorithm 2: performance analysis and controller matrix."
    )
    parser.add_argument("--sigma-max", type=float, default=0.7, help="Input sigma_max.")
    parser.add_argument("--h", type=float, default=0.5, help="Input h.")
    parser.add_argument("--theta", type=float, default=0.5, help="Input theta.")
    parser.add_argument("--epsilon", type=float, default=5.0, help="Input epsilon.")
    parser.add_argument("--kappa", type=float, default=1.0, help="Input kappa.")
    parser.add_argument("--upper-limit-m", type=int, default=50000, help="Loop upper limit m.")
    parser.add_argument(
        "--eta-step",
        type=float,
        default=0.0001,
        help="Step size for eta_i, matching 0:0.0001:1 in Algorithm 2.",
    )
    parser.add_argument("--eta-stop", type=float, default=1.0, help="End value for eta_i.")
    parser.add_argument(
        "--delay-limit",
        type=float,
        default=2.0173,
        help="Demo maximum feasible delay. Replace oracle for real LMI results.",
    )
    parser.add_argument(
        "--max-print-rows",
        type=int,
        default=25,
        help="Maximum rows printed from the search table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = Algorithm2Input(
        sigma_max=args.sigma_max,
        h=args.h,
        theta=args.theta,
        epsilon=args.epsilon,
        kappa=args.kappa,
    )

    oracle = lambda tau, alg_inputs: demo_lmi_oracle(
        tau,
        alg_inputs,
        delay_limit=args.delay_limit,
    )

    tau_max, k_matrix, records = performance_analysis_and_controller_matrix(
        inputs,
        upper_limit_m=args.upper_limit_m,
        eta_step=args.eta_step,
        eta_stop=args.eta_stop,
        lmi_oracle=oracle,
    )

    print_records(records, max_rows=args.max_print_rows)
    print()
    print(f"Maximum acceptable time delay tau_max = {tau_max:.4f}")
    print("Controller gain matrix K = Y X^-1:")
    print(np.array2string(k_matrix, precision=4, suppress_small=False))


if __name__ == "__main__":
    main()
