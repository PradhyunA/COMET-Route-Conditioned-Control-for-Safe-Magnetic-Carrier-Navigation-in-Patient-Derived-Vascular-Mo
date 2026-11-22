"""Safe-success confidence intervals across replicate benchmark runs.

Each --replicate is a name and the run directories holding its paired_results.csv
files, as name=dir or name=dir1+dir2. Routes are the resampling unit, so a route
counted at three flow scales and several seeds is not treated as independent.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t as student_t


def _is_true(value):
    return str(value).strip().lower() in ("true", "1", "1.0")


def load_replicate(dirs, controllers, anatomies):
    """Return {(controller, anatomy, pair, flow): safe_success} for one replicate."""
    results = {}
    for directory in dirs:
        for path in Path(directory).rglob("paired_results.csv"):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    controller, anatomy = row["controller_key"], row["run_id"]
                    if controller not in controllers or (anatomies and anatomy not in anatomies):
                        continue
                    key = (controller, anatomy, row["pair_id"], float(row["flow_scale"]))
                    if key in results:
                        raise ValueError(f"Duplicate result for {key} under {directory}")
                    results[key] = (_is_true(row["reached_target"]) and _is_true(row["clearance_target_met"])
                                    and not _is_true(row["wall_exit"]))
    return results


def outcome_tensor(replicates, controllers):
    """Stack outcomes as (controller, replicate, case) plus each case's route index."""
    cases = sorted({key[1:] for results in replicates.values() for key in results})
    routes = sorted({case[:2] for case in cases})
    route_of_case = np.array([routes.index(case[:2]) for case in cases])
    outcomes = np.zeros((len(controllers), len(replicates), len(cases)))
    for r, results in enumerate(replicates.values()):
        for c, controller in enumerate(controllers):
            for i, case in enumerate(cases):
                if (controller, *case) not in results:
                    raise ValueError(f"Missing {controller} {case} in a replicate")
                outcomes[c, r, i] = results[(controller, *case)]
    return outcomes, route_of_case, cases


def route_bootstrap(outcomes, route_of_case, n_boot, seed):
    """Resample routes to get each controller's safe rate across all replicates."""
    rng = np.random.default_rng(seed)
    n_routes = int(route_of_case.max()) + 1
    per_route = np.stack([outcomes[:, :, route_of_case == i].mean(axis=(1, 2)) for i in range(n_routes)], axis=1)
    draws = rng.integers(0, n_routes, size=(n_boot, n_routes))
    return per_route[:, draws].mean(axis=2)


def replicate_ci(counts, level=0.95):
    """Mean and half-width of a t interval over replicate success counts."""
    counts = np.asarray(counts, dtype=float)
    if len(counts) < 2:
        return float(counts.mean()), float("nan")
    half = student_t.ppf(0.5 + level / 2, len(counts) - 1) * counts.std(ddof=1) / np.sqrt(len(counts))
    return float(counts.mean()), float(half)


def summarize(outcomes, route_of_case, controllers, reference, n_boot, seed):
    boot = route_bootstrap(outcomes, route_of_case, n_boot, seed)
    n_cases = outcomes.shape[2]
    ref = controllers.index(reference)
    rows = []
    for c, controller in enumerate(controllers):
        counts = outcomes[c].sum(axis=1)
        mean, half = replicate_ci(counts)
        row = {"controller": controller, "n_cases": n_cases, "replicate_safe_counts": " ".join(f"{x:g}" for x in counts),
               "safe_mean": mean, "safe_replicate_ci_halfwidth": half,
               "safe_rate": outcomes[c].mean(),
               "safe_rate_route_ci_low": np.percentile(boot[c], 2.5),
               "safe_rate_route_ci_high": np.percentile(boot[c], 97.5)}
        if c != ref:
            delta = boot[ref] - boot[c]
            only_ref = int(((outcomes[ref] == 1) & (outcomes[c] == 0)).sum())
            only_other = int(((outcomes[c] == 1) & (outcomes[ref] == 0)).sum())
            row.update({"rate_gain_of_reference": outcomes[ref].mean() - outcomes[c].mean(),
                        "rate_gain_route_ci_low": np.percentile(delta, 2.5),
                        "rate_gain_route_ci_high": np.percentile(delta, 97.5),
                        "reference_only": only_ref, "other_only": only_other})
            if outcomes.shape[1] == 1:
                total = only_ref + only_other
                row["mcnemar_p"] = binomtest(only_ref, total).pvalue if total else 1.0
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate", action="append", required=True, help="name=dir or name=dir1+dir2")
    parser.add_argument("--controllers", required=True, help="Comma-separated; the first is the reference.")
    parser.add_argument("--anatomies", default="", help="Comma-separated; empty keeps every anatomy found.")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional CSV path.")
    args = parser.parse_args()

    controllers = args.controllers.split(",")
    anatomies = set(filter(None, args.anatomies.split(",")))
    replicates = {}
    for spec in args.replicate:
        name, dirs = spec.split("=", 1)
        replicates[name] = load_replicate(dirs.split("+"), set(controllers), anatomies)
    outcomes, route_of_case, cases = outcome_tensor(replicates, controllers)
    rows = summarize(outcomes, route_of_case, controllers, controllers[0], args.n_boot, args.seed)

    print(f"{len(replicates)} replicate(s), {len(cases)} cases, {int(route_of_case.max()) + 1} routes")
    fields = {}
    for row in rows:
        fields.update(dict.fromkeys(row))
        line = (f"{row['controller']:14s} safe {row['safe_mean']:5.1f}/{row['n_cases']}"
                f" ±{row['safe_replicate_ci_halfwidth']:.1f} [{row['replicate_safe_counts']}]"
                f" rate {row['safe_rate']:.3f} route CI {row['safe_rate_route_ci_low']:.3f}-{row['safe_rate_route_ci_high']:.3f}")
        if "rate_gain_of_reference" in row:
            line += (f" | gain {row['rate_gain_of_reference']:+.3f} CI {row['rate_gain_route_ci_low']:+.3f}"
                     f" to {row['rate_gain_route_ci_high']:+.3f} ({row['reference_only']}/{row['other_only']})")
        if "mcnemar_p" in row:
            line += f" p={row['mcnemar_p']:.2g}"
        print(line)
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields), restval="")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
