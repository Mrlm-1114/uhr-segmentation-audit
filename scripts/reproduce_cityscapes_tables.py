#!/usr/bin/env python3
"""Reconstruct supplementary Tables S5/S6 from preserved numeric summaries.

Standard library only. This does not rerun inference, route selection or the
historical bootstrap. See records/cityscapes/README.md for the record scope.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys


FAMILIES = ("GlobalResize", "SparseRefine", "RCLC")
ORACLE_FIELDS = (
    "local_miou_percent", "matched_linear_global_miou_percent",
    "fitted_expected_mean_latency_ms", "maximum_margin_pp",
    "ci95_low_pp", "ci95_high_pp",
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct(root):
    checks = []

    def check(item, actual, expected):
        checks.append(dict(item=item, actual=actual, expected=expected,
                           status="PASS" if actual == expected else "FAIL"))

    def near(item, actual, expected):
        a, b = float(actual), float(expected)
        checks.append(dict(item=item, actual=a, expected=b,
                           status="PASS" if math.isfinite(a) and math.isfinite(b)
                           and abs(a - b) <= 1e-10 else "FAIL"))

    endpoints = read_csv(root / "shared_base_endpoints.csv")
    choices = read_csv(root / "shared_base_budget_choices.csv")
    sessions = read_csv(root / "shared_base_session_quantiles.csv")
    oracle = read_csv(root / "oracle_family_summaries.csv")
    expected = read_json(root / "expected_tables.json")
    protocol = read_json(root / "oracle_protocol.json")
    for name, rows, count in [("endpoints", endpoints, 29), ("choices", choices, 32),
                              ("sessions", sessions, 87), ("oracle", oracle, 54)]:
        check(name + "/row_count", len(rows), count)

    by_id = {row["endpoint_id"]: row for row in endpoints}
    check("endpoints/unique_ids", len(by_id), len(endpoints))
    check("sessions/endpoint_ids", sorted({r["endpoint_id"] for r in sessions}),
          sorted(by_id))
    for eid, endpoint in by_id.items():
        group = [r for r in sessions if r["endpoint_id"] == eid]
        check(eid + "/session_ids", sorted(int(r["session"]) for r in group), [0, 1, 2])
        for field in ("p50_ms", "p95_ms"):
            near(eid + "/median_session_" + field,
                 statistics.median(float(r[field]) for r in group), endpoint[field])
        for row in group:
            near(eid + "/session" + row["session"] + "/historical_quality",
                 row["historical_pooled_miou_percent"], endpoint["miou_percent"])
            check(eid + "/session" + row["session"] + "/quantile_order",
                  0 < float(row["p50_ms"]) <= float(row["p95_ms"]), True)

    budgets = sorted({row["budget_ms"] for row in choices}, key=float)
    check("choices/budget_count", len(budgets), 8)
    by_choice = {(r["budget_ms"], r["family"]): r for r in choices}
    check("choices/unique_budget_family", len(by_choice), len(choices))
    for budget in budgets:
        check("choices/" + budget + "/families",
              sorted(r["family"] for r in choices if r["budget_ms"] == budget),
              sorted((*FAMILIES, "FixedBudget")))
    for row in choices:
        item = "choice/" + row["budget_ms"] + "/" + row["family"]
        if row["status"] == "no_feasible_candidate":
            for field in ("endpoint_id", "calibration_miou", "development_p50_ms",
                          "evaluation_miou_percent", "evaluation_p50_ms",
                          "evaluation_p95_ms", "evaluation_within_frozen_budget"):
                check(item + "/" + field, row[field], "")
            continue
        check(item + "/status", row["status"], "selected")
        endpoint = by_id[row["endpoint_id"]]
        check(item + "/family", endpoint["family"], row["family"])
        for field in ("miou_percent", "p50_ms", "p95_ms"):
            near(item + "/" + field, row["evaluation_" + field], endpoint[field])
        check(item + "/calibration_fraction", 0 <= float(row["calibration_miou"]) <= 1, True)
        check(item + "/development_budget_feasible",
              0 < float(row["development_p50_ms"]) <= float(row["budget_ms"]), True)
        check(item + "/evaluation_budget_flag", row["evaluation_within_frozen_budget"],
              str(float(row["evaluation_p50_ms"]) <= float(row["budget_ms"])))

    # Use the frozen selections. Evaluation quality is never used for reselection.
    s5 = []
    for budget in budgets:
        out = {"budget_ms": f"{float(budget):.3f}"}
        for family in FAMILIES:
            row = by_choice[budget, family]
            out[family] = None if row["status"] == "no_feasible_candidate" else {
                "miou_percent": f"{float(by_id[row['endpoint_id']]['miou_percent']):.2f}",
                "p50_ms": f"{float(by_id[row['endpoint_id']]['p50_ms']):.3f}",
            }
        s5.append(out)

    by_oracle = {(int(r["head_seed"]), r["precision"], r["family"]): r for r in oracle}
    check("oracle/unique_seed_precision_family", len(by_oracle), len(oracle))
    for seed in protocol["head_seeds"]:
        for precision in ("fp32", "fp16"):
            check(f"oracle/{seed}/{precision}/families",
                  len({r["family"] for r in oracle
                       if int(r["head_seed"]) == seed and r["precision"] == precision}), 9)
    for (seed, precision, family), row in by_oracle.items():
        item = f"oracle/{seed}/{precision}/{family}"
        near(item + "/margin", float(row["local_miou_percent"])
             - float(row["matched_linear_global_miou_percent"]), row["maximum_margin_pp"])
        check(item + "/finite_fields", all(math.isfinite(float(row[f])) for f in ORACLE_FIELDS), True)
        check(item + "/interval_order", float(row["ci95_low_pp"]) <= float(row["ci95_high_pp"]), True)
        check(item + "/positive_latency", float(row["fitted_expected_mean_latency_ms"]) > 0, True)
        check(item + "/source_record_id", row["source_record_id"], f"oracle_seed{seed}_{precision}")
    s6 = []
    for precision in ("fp32", "fp16"):
        for seed in protocol["head_seeds"]:
            row = by_oracle[seed, precision, protocol["primary_family"]]
            out = dict(head_seed=seed, precision=precision.upper())
            out.update({field: f"{float(row[field]):.3f}" for field in ORACLE_FIELDS})
            s6.append(out)

    # Compare every printed value against a separate manuscript snapshot.
    def compare(item, actual, target):
        if isinstance(target, dict):
            check(item + "/keys", sorted(actual), sorted(target))
            for key in target:
                compare(item + "/" + key, actual.get(key), target[key])
        elif isinstance(target, list):
            check(item + "/rows", len(actual), len(target))
            for i, (a, b) in enumerate(zip(actual, target), 1):
                compare(item + f"/row{i}", a, b)
        else:
            check(item, actual, target)

    tables = {"S5": s5, "S6": s6}
    compare("manuscript", tables, expected)
    return tables, checks


def csv_tables(tables):
    s5 = []
    for row in tables["S5"]:
        out = {"budget_ms": row["budget_ms"]}
        for family in FAMILIES:
            for field in ("miou_percent", "p50_ms"):
                out[family + "_" + field] = row[family][field] if row[family] else ""
        s5.append(out)
    return {"S5": s5, "S6": tables["S6"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path,
                        default=Path(__file__).resolve().parents[1] / "records" / "cityscapes")
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--write-tables", type=Path,
                        help="Write table_S5.csv and table_S6.csv here, only after all checks pass.")
    args = parser.parse_args()
    try:
        tables, checks = reconstruct(args.records)
        exports = csv_tables(tables)
        # Check any committed display CSVs as well as the independent snapshot.
        for name, rows in exports.items():
            path = args.records / ("table_" + name + ".csv")
            if path.exists():
                existing = read_csv(path)
                rendered = [{key: str(value) for key, value in row.items()} for row in rows]
                checks.append(dict(item=name + "/committed_csv", actual=existing, expected=rendered,
                                   status="PASS" if existing == rendered else "FAIL"))
        failures = [row for row in checks if row["status"] != "PASS"]
        if args.format == "json":
            print(json.dumps(dict(status="FAIL" if failures else "PASS", tables=tables,
                                  checks=checks), indent=2))
        else:
            writer = csv.DictWriter(sys.stdout, ["item", "actual", "expected", "status"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(checks)
        print(f"{len(checks)} checks; {len(failures)} mismatches.", file=sys.stderr)
        if failures:
            return 1
        if args.write_tables:
            args.write_tables.mkdir(parents=True, exist_ok=True)
            for name, rows in exports.items():
                with (args.write_tables / ("table_" + name + ".csv")).open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
        return 0
    except (OSError, ValueError, KeyError, TypeError, statistics.StatisticsError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
