#!/usr/bin/env python3
"""Compare certified corrections with a lifted SDP relaxation on Oxford data.

The comparator is deliberately independent of the scalar root solver.  For a
measured correspondence z0 and fundamental matrix F it writes the translated
epipolar constraint as

    p.T A p + 2 g.T p + c = 0,          z = z0 + p,

and applies the Shor lift Y = [p;1][p;1].T.  The resulting five-by-five SDP has
one epipolar equality, Y[4,4] = 1, and Y positive semidefinite.  An exactly
feasible rank-one SDP optimum recovers a globally optimal correction for the
original QCQP; the floating-point gates below provide numerical corroboration.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from cvxopt import matrix, solvers
except ImportError as error:  # pragma: no cover - exercised by runner gate
    raise SystemExit(
        "This comparator requires NumPy and CVXOPT; install "
        "scripts/requirements-sdp.txt or set TPAMI_SDP_PYTHON."
    ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--cameras", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pair-output", type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="evaluate every point row instead of cost-stratified sampling",
    )
    parser.add_argument(
        "--samples-per-pair",
        type=int,
        default=3,
        help="cost-stratified samples per nonempty view pair (default: 3)",
    )
    parser.add_argument("--expected-input-rows", type=int, default=27080)
    parser.add_argument("--expected-nonempty-pairs", type=int, default=364)
    parser.add_argument("--expected-samples", type=int, default=939)
    return parser.parse_args()


def load_cameras(path: Path) -> list[np.ndarray]:
    cameras: list[np.ndarray] = []
    with path.open("r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            expected_view = len(cameras) + 1
            if len(fields) != 13 or int(fields[0]) != expected_view:
                raise ValueError(f"invalid camera row {line_number}")
            values = np.asarray([float(value) for value in fields[1:]], dtype=float)
            cameras.append(values.reshape(3, 4))
    if len(cameras) != 36:
        raise ValueError(f"expected 36 cameras, found {len(cameras)}")
    return cameras


def load_tracks(path: Path) -> list[np.ndarray]:
    tracks: list[np.ndarray] = []
    with path.open("r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, start=1):
            values = np.asarray([float(value) for value in line.split()], dtype=float)
            if values.size != 72 or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid track row {line_number}")
            tracks.append(values)
    if not tracks:
        raise ValueError("track file is empty")
    return tracks


def load_point_rows(path: Path) -> list[dict[str, str]]:
    required = {
        "view1",
        "view2",
        "track",
        "status_name",
        "lott_finite",
        "lott_cost",
        "lott_normalized_residual",
    }
    with path.open("r", encoding="ascii", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("point CSV is missing required columns")
        rows = list(reader)
    if not rows:
        raise ValueError("point CSV is empty")
    return rows


def derive_fundamental(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    _, _, right = np.linalg.svd(first, full_matrices=True)
    center = right[-1]
    epipole = second @ center
    skew = np.asarray(
        [
            [0.0, -epipole[2], epipole[1]],
            [epipole[2], 0.0, -epipole[0]],
            [-epipole[1], epipole[0], 0.0],
        ]
    )
    raw = skew @ second @ np.linalg.pinv(first)
    left, singular_values, right = np.linalg.svd(raw)
    singular_values[-1] = 0.0
    fundamental = (left * singular_values) @ right
    return fundamental / np.linalg.norm(fundamental)


def translated_constraint(
    fundamental: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    quadratic = np.zeros((4, 4), dtype=float)
    quadratic[0, 2] = quadratic[2, 0] = fundamental[0, 0] / 2.0
    quadratic[1, 2] = quadratic[2, 1] = fundamental[0, 1] / 2.0
    quadratic[0, 3] = quadratic[3, 0] = fundamental[1, 0] / 2.0
    quadratic[1, 3] = quadratic[3, 1] = fundamental[1, 1] / 2.0
    linear = np.asarray(
        [fundamental[2, 0], fundamental[2, 1], fundamental[0, 2], fundamental[1, 2]]
    )
    constant = float(
        observed @ quadratic @ observed
        + linear @ observed
        + fundamental[2, 2]
    )
    shifted_linear_half = quadratic @ observed + linear / 2.0
    constraint = np.zeros((5, 5), dtype=float)
    constraint[:4, :4] = quadratic
    constraint[:4, 4] = shifted_linear_half
    constraint[4, :4] = shifted_linear_half
    constraint[4, 4] = constant
    norm = np.linalg.norm(constraint)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("invalid translated epipolar constraint")
    return constraint / norm


def symmetric_bases(order: int = 5) -> list[np.ndarray]:
    bases: list[np.ndarray] = []
    for row in range(order):
        for column in range(row + 1):
            basis = np.zeros((order, order), dtype=float)
            basis[row, column] = 1.0
            basis[column, row] = 1.0
            bases.append(basis)
    return bases


BASES = symmetric_bases()
OBJECTIVE = np.diag([1.0, 1.0, 1.0, 1.0, 0.0])
SDP_COST = matrix([float(np.sum(OBJECTIVE * basis)) for basis in BASES], tc="d")
SDP_CONE = matrix(
    np.column_stack([-basis.reshape(-1, order="F") for basis in BASES])
)
SDP_RIGHT_HAND_SIDE = [matrix(np.zeros((5, 5), dtype=float))]


def solve_relaxation(constraint: np.ndarray) -> dict[str, Any]:
    equalities = np.asarray(
        [
            [float(np.sum(constraint * basis)) for basis in BASES],
            [float(basis[4, 4]) for basis in BASES],
        ]
    )
    result = solvers.sdp(
        SDP_COST,
        Gs=[SDP_CONE],
        hs=SDP_RIGHT_HAND_SIDE,
        A=matrix(equalities),
        b=matrix([0.0, 1.0]),
    )
    if result["x"] is None:
        return {"status": result["status"]}
    coefficients = np.asarray(result["x"]).reshape(-1)
    lifted = sum(
        (coefficient * basis for coefficient, basis in zip(coefficients, BASES)),
        start=np.zeros((5, 5), dtype=float),
    )
    eigenvalues = np.linalg.eigvalsh(lifted)
    leading = max(abs(float(eigenvalues[-1])), np.finfo(float).tiny)
    tail_ratio = max(abs(float(value)) for value in eigenvalues[:-1]) / leading
    psd_violation_ratio = max(0.0, -float(eigenvalues[0])) / leading
    correction = lifted[:4, 4] / lifted[4, 4]
    return {
        "status": result["status"],
        "primal_objective": float(result["primal objective"]),
        "dual_objective": float(result["dual objective"]),
        "duality_gap": float(result["gap"]),
        "primal_infeasibility": float(result["primal infeasibility"]),
        "dual_infeasibility": float(result["dual infeasibility"]),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "tail_spectral_ratio": tail_ratio,
        "psd_violation_ratio": psd_violation_ratio,
        "extracted_cost": float(correction @ correction),
        "extracted_constraint_residual": abs(
            float(np.r_[correction, 1.0] @ constraint @ np.r_[correction, 1.0])
        ),
    }


def select_rows(
    rows: list[dict[str, str]], samples_per_pair: int, all_rows: bool
) -> list[dict[str, str]]:
    if all_rows:
        return list(rows)
    if samples_per_pair <= 0:
        raise ValueError("samples-per-pair must be positive")
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["view1"]), int(row["view2"]))].append(row)
    selected: list[dict[str, str]] = []
    for pair in sorted(grouped):
        ordered = sorted(grouped[pair], key=lambda row: (float(row["lott_cost"]), int(row["track"])))
        count = min(samples_per_pair, len(ordered))
        indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
        selected.extend(ordered[int(index)] for index in np.unique(indices))
    return selected


def finite_max(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return math.nan
    values = [float(record.get(key, math.inf)) for record in records]
    return max(values) if all(math.isfinite(value) for value in values) else math.inf


def main() -> int:
    args = parse_args()
    cameras = load_cameras(args.cameras)
    tracks = load_tracks(args.tracks)
    all_rows = load_point_rows(args.points)
    rows = select_rows(all_rows, args.samples_per_pair, args.all_rows)
    pair_fundamentals: dict[tuple[int, int], np.ndarray] = {}

    solvers.options.update(
        show_progress=False,
        abstol=1e-10,
        reltol=1e-9,
        feastol=1e-9,
        maxiters=200,
        refinement=3,
    )

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        first = int(row["view1"]) - 1
        second = int(row["view2"]) - 1
        track_index = int(row["track"]) - 1
        if not (0 <= first < second < len(cameras)):
            raise ValueError("point CSV contains an out-of-range or unordered view pair")
        if not (0 <= track_index < len(tracks)):
            raise ValueError("point CSV contains an out-of-range track index")
        pair = (first, second)
        if pair not in pair_fundamentals:
            pair_fundamentals[pair] = derive_fundamental(
                cameras[first], cameras[second]
            )
        track = tracks[track_index]
        observed = np.asarray(
            [
                track[2 * first],
                track[2 * first + 1],
                track[2 * second],
                track[2 * second + 1],
            ]
        )
        if np.any(observed == -1.0):
            raise ValueError("point CSV references a missing observation")
        constraint = translated_constraint(pair_fundamentals[pair], observed)
        record: dict[str, Any] = {
            "view1": first + 1,
            "view2": second + 1,
            "track": track_index + 1,
            "lott_status_name": row["status_name"],
            "lott_finite": int(row["lott_finite"]),
            "lott_cost": float(row["lott_cost"]),
            "lott_normalized_residual": float(row["lott_normalized_residual"]),
        }
        record.update(solve_relaxation(constraint))
        if record["status"] == "optimal":
            scale = max(1.0, abs(record["lott_cost"]))
            record["lott_primal_absolute_gap"] = abs(
                record["lott_cost"] - record["primal_objective"]
            )
            record["lott_primal_scaled_gap"] = (
                record["lott_primal_absolute_gap"] / scale
            )
            record["lott_extracted_absolute_gap"] = abs(
                record["lott_cost"] - record["extracted_cost"]
            )
            record["lott_extracted_scaled_gap"] = (
                record["lott_extracted_absolute_gap"] / scale
            )
            record["duality_gap_scaled"] = record["duality_gap"] / scale
            record["agreement_scaled_1e-8"] = int(
                record["lott_extracted_scaled_gap"] <= 1e-8
            )
            record["rank_one_1e-7"] = int(record["tail_spectral_ratio"] <= 1e-7)
        output_rows.append(record)

    fieldnames = [
        "view1",
        "view2",
        "track",
        "status",
        "lott_status_name",
        "lott_finite",
        "lott_cost",
        "lott_normalized_residual",
        "primal_objective",
        "dual_objective",
        "duality_gap",
        "primal_infeasibility",
        "dual_infeasibility",
        "minimum_eigenvalue",
        "tail_spectral_ratio",
        "psd_violation_ratio",
        "extracted_cost",
        "extracted_constraint_residual",
        "lott_primal_absolute_gap",
        "lott_primal_scaled_gap",
        "lott_extracted_absolute_gap",
        "lott_extracted_scaled_gap",
        "duality_gap_scaled",
        "agreement_scaled_1e-8",
        "rank_one_1e-7",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    optimal = [record for record in output_rows if record["status"] == "optimal"]
    input_keys = [
        (int(row["view1"]), int(row["view2"]), int(row["track"]))
        for row in all_rows
    ]
    input_keys_unique = len(set(input_keys))
    certified_statuses = {
        "already_feasible",
        "affine",
        "regular_interior",
        "boundary_psd_unique",
        "boundary_psd_nonunique",
    }
    diagnostic_keys = {
        "primal_objective",
        "dual_objective",
        "duality_gap",
        "primal_infeasibility",
        "dual_infeasibility",
        "minimum_eigenvalue",
        "tail_spectral_ratio",
        "psd_violation_ratio",
        "extracted_cost",
        "extracted_constraint_residual",
        "lott_primal_scaled_gap",
        "lott_extracted_scaled_gap",
        "duality_gap_scaled",
    }
    def lott_record_valid(record: dict[str, Any]) -> bool:
        return (
            record["lott_status_name"] in certified_statuses
            and record["lott_finite"] == 1
            and math.isfinite(record["lott_cost"])
            and record["lott_cost"] >= 0.0
            and math.isfinite(record["lott_normalized_residual"])
            and record["lott_normalized_residual"] <= 1e-12
        )

    def diagnostics_record_finite(record: dict[str, Any]) -> bool:
        return diagnostic_keys.issubset(record) and all(
            math.isfinite(float(record[key])) for key in diagnostic_keys
        )

    lott_input_valid = sum(int(lott_record_valid(record)) for record in output_rows)
    diagnostics_finite = sum(
        int(diagnostics_record_finite(record))
        for record in output_rows
    )
    rank_one = sum(int(record.get("rank_one_1e-7", 0)) for record in output_rows)
    agreement = sum(
        int(record.get("agreement_scaled_1e-8", 0)) for record in output_rows
    )
    pairs = {(record["view1"], record["view2"]) for record in output_rows}

    if args.pair_output is not None:
        records_by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for record in output_rows:
            records_by_pair[(record["view1"], record["view2"])].append(record)
        pair_fieldnames = [
            "view1",
            "view2",
            "rows",
            "lott_valid",
            "diagnostics_finite",
            "sdp_optimal",
            "rank_one_1e-7",
            "agreement_scaled_1e-8",
            "max_lott_primal_scaled_gap",
            "max_lott_extracted_scaled_gap",
            "max_tail_spectral_ratio",
            "max_psd_violation_ratio",
            "max_extracted_constraint_residual",
            "max_primal_infeasibility",
            "max_dual_infeasibility",
            "max_duality_gap_scaled",
        ]
        args.pair_output.parent.mkdir(parents=True, exist_ok=True)
        with args.pair_output.open("w", encoding="ascii", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=pair_fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            for pair in sorted(records_by_pair):
                records = records_by_pair[pair]
                pair_optimal = [
                    record for record in records if record["status"] == "optimal"
                ]
                writer.writerow(
                    {
                        "view1": pair[0],
                        "view2": pair[1],
                        "rows": len(records),
                        "lott_valid": sum(
                            int(lott_record_valid(record)) for record in records
                        ),
                        "diagnostics_finite": sum(
                            int(diagnostics_record_finite(record))
                            for record in records
                        ),
                        "sdp_optimal": len(pair_optimal),
                        "rank_one_1e-7": sum(
                            int(record.get("rank_one_1e-7", 0))
                            for record in records
                        ),
                        "agreement_scaled_1e-8": sum(
                            int(record.get("agreement_scaled_1e-8", 0))
                            for record in records
                        ),
                        "max_lott_primal_scaled_gap": finite_max(
                            pair_optimal, "lott_primal_scaled_gap"
                        ),
                        "max_lott_extracted_scaled_gap": finite_max(
                            pair_optimal, "lott_extracted_scaled_gap"
                        ),
                        "max_tail_spectral_ratio": finite_max(
                            pair_optimal, "tail_spectral_ratio"
                        ),
                        "max_psd_violation_ratio": finite_max(
                            pair_optimal, "psd_violation_ratio"
                        ),
                        "max_extracted_constraint_residual": finite_max(
                            pair_optimal, "extracted_constraint_residual"
                        ),
                        "max_primal_infeasibility": finite_max(
                            pair_optimal, "primal_infeasibility"
                        ),
                        "max_dual_infeasibility": finite_max(
                            pair_optimal, "dual_infeasibility"
                        ),
                        "max_duality_gap_scaled": finite_max(
                            pair_optimal, "duality_gap_scaled"
                        ),
                    }
                )

    summary = {
        "input_point_rows": len(all_rows),
        "input_unique_view_pair_track_keys": input_keys_unique,
        "nonempty_view_pairs_sampled": len(pairs),
        "selection_mode": "all_rows" if args.all_rows else "lott_cost_stratified",
        "samples_per_pair": "all" if args.all_rows else args.samples_per_pair,
        "sample_count": len(output_rows),
        "lott_input_certified_finite_feasible_1e-12": lott_input_valid,
        "sdp_required_diagnostics_finite": diagnostics_finite,
        "sdp_optimal": len(optimal),
        "rank_one_1e-7": rank_one,
        "lott_agreement_scaled_1e-8": agreement,
        "max_lott_primal_absolute_gap": finite_max(optimal, "lott_primal_absolute_gap"),
        "max_lott_primal_scaled_gap": finite_max(optimal, "lott_primal_scaled_gap"),
        "max_lott_extracted_absolute_gap": finite_max(
            optimal, "lott_extracted_absolute_gap"
        ),
        "max_lott_extracted_scaled_gap": finite_max(
            optimal, "lott_extracted_scaled_gap"
        ),
        "max_tail_spectral_ratio": finite_max(optimal, "tail_spectral_ratio"),
        "max_psd_violation_ratio": finite_max(optimal, "psd_violation_ratio"),
        "max_extracted_constraint_residual": finite_max(
            optimal, "extracted_constraint_residual"
        ),
        "max_primal_infeasibility": finite_max(optimal, "primal_infeasibility"),
        "max_dual_infeasibility": finite_max(optimal, "dual_infeasibility"),
        "max_duality_gap": finite_max(optimal, "duality_gap"),
        "max_duality_gap_scaled": finite_max(optimal, "duality_gap_scaled"),
    }
    passed = (
        len(all_rows) == args.expected_input_rows
        and input_keys_unique == len(all_rows)
        and len(pairs) == args.expected_nonempty_pairs
        and len(output_rows) == args.expected_samples
        and lott_input_valid == len(output_rows)
        and diagnostics_finite == len(output_rows)
        and len(optimal) == len(output_rows)
        and rank_one == len(output_rows)
        and agreement == len(output_rows)
        and summary["max_lott_primal_scaled_gap"] <= 1e-8
        and summary["max_extracted_constraint_residual"] <= 1e-7
        and summary["max_primal_infeasibility"] <= 1e-8
        and summary["max_dual_infeasibility"] <= 1e-8
        and summary["max_duality_gap_scaled"] <= 1e-8
        and summary["max_psd_violation_ratio"] <= 1e-7
    )
    summary["sdp_comparator"] = "PASS" if passed else "FAIL"
    lines = [
        f"{key}={value:.17g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in summary.items()
    ]
    args.summary.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
