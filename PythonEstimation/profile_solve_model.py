from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

from PythonEstimation.model_functions import Parameters, solve_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile solve_model to identify computational bottlenecks.")
    parser.add_argument("--ns", type=int, default=120, help="Inventory grid size (smaller than production default for fast profiling).")
    parser.add_argument("--q", type=int, default=19, help="Gauss-Hermite quadrature nodes.")
    parser.add_argument("--q-omega", type=int, default=7, help="Tauchen omega grid size.")
    parser.add_argument("--maxiter", type=int, default=300, help="Max iterations for value-function iteration.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated solve_model runs.")
    parser.add_argument("--verbose", action="store_true", help="Enable per-iteration VFI progress prints.")
    parser.add_argument("--profile-lines", type=int, default=25, help="Number of profile rows to print.")
    return parser


def build_candidate_params(ns: int, q: int, q_omega: int) -> Parameters:
    # Candidate vector close to values used in the production run path.
    # Parameter order reference: (gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta)
    return Parameters(
        gamma=1.05,
        mu_eta=0.0,
        sigma_eta2=0.08,
        rho_omega=0.85,
        sigma_nu2=0.15,
        epsilon=2.5,
        delta=0.08,
        ns=ns,
        q=q,
        q_omega=q_omega,
    )


def profile_one_run(params: Parameters, maxiter: int, verbose: bool, profile_lines: int) -> tuple[float, bool]:
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    _, _, _, _, _, _, _, converged = solve_model(params, full=False, verbose=verbose, maxiter=maxiter)
    pr.disable()
    elapsed = time.perf_counter() - t0

    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream).sort_stats("cumtime")
    stats.print_stats(profile_lines)

    print("\n=== solve_model profile (top cumulative time) ===", flush=True)
    print(stream.getvalue(), flush=True)
    return elapsed, converged


def main() -> None:
    args = build_parser().parse_args()

    params = build_candidate_params(args.ns, args.q, args.q_omega)
    print("Candidate parameter vector for solve_model speed test:", flush=True)
    print(
        {
            "gamma": params.gamma,
            "mu_eta": params.mu_eta,
            "sigma_eta2": params.sigma_eta2,
            "rho_omega": params.rho_omega,
            "sigma_nu2": params.sigma_nu2,
            "epsilon": params.epsilon,
            "delta": params.delta,
            "ns": params.ns,
            "q": params.q,
            "q_omega": params.q_omega,
        },
        flush=True,
    )

    durations = []
    convergences = []
    for r in range(args.repeats):
        print(f"\n--- solve_model run {r + 1}/{args.repeats} ---", flush=True)
        elapsed, converged = profile_one_run(
            params=params,
            maxiter=args.maxiter,
            verbose=args.verbose,
            profile_lines=args.profile_lines,
        )
        durations.append(elapsed)
        convergences.append(converged)
        print(f"Run {r + 1}: elapsed={elapsed:.2f}s, converged={converged}", flush=True)

    avg = sum(durations) / len(durations)
    print("\n=== Summary ===", flush=True)
    print(f"Runs: {len(durations)}", flush=True)
    print(f"Average elapsed: {avg:.2f}s", flush=True)
    print(f"Min elapsed: {min(durations):.2f}s", flush=True)
    print(f"Max elapsed: {max(durations):.2f}s", flush=True)
    print(f"All converged: {all(convergences)}", flush=True)


if __name__ == "__main__":
    main()
