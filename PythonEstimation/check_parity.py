import subprocess
import sys
from pathlib import Path

import numpy as np

from model_functions import Parameters, solve_value_function


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "parity_reference"


def _load_vector(name):
    arr = np.loadtxt(REFERENCE_DIR / f"{name}.csv", delimiter=",")
    return np.atleast_1d(np.asarray(arr, dtype=np.float64))


def _load_matrix(name):
    arr = np.loadtxt(REFERENCE_DIR / f"{name}.csv", delimiter=",")
    return np.asarray(arr, dtype=np.float64)


def _max_abs_diff(a, b):
    return float(np.max(np.abs(a - b)))


def _max_rel_diff(a, b):
    scale = max(float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / scale)


def _print_comparison(name, py_arr, jl_arr):
    print(
        f"{name}: shape={py_arr.shape}, max_abs_diff={_max_abs_diff(py_arr, jl_arr):.6e}, max_rel_diff={_max_rel_diff(py_arr, jl_arr):.6e}"
    )


def run_julia_export():
    cmd = ["julia", str(ROOT / "export_julia_reference.jl")]
    subprocess.run(cmd, check=True)


def run_python_solution():
    ns = 200
    params = Parameters(
        c=1.0,
        fc=0.0,
        mu_eta=float(np.log(0.01)),
        sigma_eta2=0.05,
        rho_omega=0.1,
        gamma=0.9,
        delta=0.01,
        beta=0.95,
        epsilon=8.0,
        mu_nu=1.0,
        sigma_nu2=0.15,
        ns=ns,
        scale=1.0,
        size=100.0,
    )
    result = solve_value_function(params, tol=1e-4, maxiter=1000)
    return params, result


def load_julia_outputs():
    return {
        "sgrid": _load_vector("sgrid"),
        "omega_grid": _load_vector("omega_grid"),
        "pi_omega": _load_vector("pi_omega"),
        "p_omega": _load_matrix("p_omega"),
        "v": _load_vector("v"),
        "n_policy": _load_matrix("n_policy"),
        "p_policy": _load_matrix("p_policy"),
        "v_by_omega": _load_matrix("v_by_omega"),
        "converged": int(_load_vector("converged")[0]),
    }


def main():
    run_export = "--run-julia" in sys.argv
    if run_export:
        print("Running Julia export...")
        run_julia_export()

    params, py_result = run_python_solution()
    jl = load_julia_outputs()

    print("\nParity results")
    _print_comparison("sgrid", params.sgrid, jl["sgrid"])
    _print_comparison("omega_grid", params.omega_grid, jl["omega_grid"])
    _print_comparison("pi_omega", params.pi_omega, jl["pi_omega"])
    _print_comparison("p_omega", params.p_omega, jl["p_omega"])
    _print_comparison("v", py_result["v"], jl["v"])
    _print_comparison("n_policy", py_result["n_policy"], jl["n_policy"])
    _print_comparison("p_policy", py_result["p_policy"], jl["p_policy"])
    _print_comparison("v_by_omega", py_result["v_by_omega"], jl["v_by_omega"])

    py_conv = 1 if py_result["converged"] else 0
    print(f"converged: python={py_conv}, julia={jl['converged']}")


if __name__ == "__main__":
    main()
