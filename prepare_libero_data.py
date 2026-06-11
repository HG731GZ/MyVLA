#!/usr/bin/env python
"""Validate LIBERO data and refresh stale derived metadata/norm files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from compute_libero_norm_stats import compute_norm_stats
from create_libero_meta import create_libero_meta
from libero_dataset_utils import (
    normalize_subsets,
    scan_libero_dataset,
    validate_official_counts,
)


def _load_json(path: Path) -> tuple[Dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "file does not exist"
    try:
        with path.open(encoding="utf-8") as input_file:
            data = json.load(input_file)
    except Exception as error:
        return None, f"cannot read JSON: {error}"
    if not isinstance(data, dict):
        return None, "JSON root is not an object"
    return data, None


def _common_mismatch_reasons(
    metadata: Dict[str, Any],
    snapshot: Dict[str, Any],
    *,
    subsets_key: str,
) -> List[str]:
    reasons = []
    if metadata.get("dataset_fingerprint") != snapshot["dataset_fingerprint"]:
        reasons.append("dataset fingerprint changed or is missing")
    if metadata.get("dataset_fingerprint_version") != snapshot["fingerprint_version"]:
        reasons.append("dataset fingerprint version changed or is missing")
    if normalize_subsets(metadata.get(subsets_key) or []) != snapshot["requested_subsets"]:
        reasons.append("requested subsets changed")
    if sorted(metadata.get("exclude_patterns") or []) != snapshot["exclude_patterns"]:
        reasons.append("excluded file patterns changed")
    if metadata.get("num_demos", metadata.get("num_episodes")) != snapshot["num_demos"]:
        reasons.append("demo count changed")
    if metadata.get("num_steps") != snapshot["num_steps"]:
        reasons.append("timestep count changed")
    return reasons


def _metadata_mismatch_reasons(
    metadata_path: Path,
    snapshot: Dict[str, Any],
) -> List[str]:
    data, load_error = _load_json(metadata_path)
    if load_error:
        return [load_error]
    assert data is not None

    reasons = _common_mismatch_reasons(
        data,
        snapshot,
        subsets_key="requested_subsets",
    )
    expected_paths = [item["path"] for item in snapshot["files"]]
    actual_paths = [
        str(Path(item.get("path", "")).expanduser().resolve())
        for item in data.get("datalist", [])
        if isinstance(item, dict)
    ]
    if actual_paths != expected_paths:
        reasons.append("metadata datalist changed or is stale")
    if data.get("num_files") != snapshot["num_files"]:
        reasons.append("HDF5 file count changed")
    return reasons


def _norm_mismatch_reasons(
    norm_path: Path,
    snapshot: Dict[str, Any],
    state_orientation_format: str,
) -> List[str]:
    data, load_error = _load_json(norm_path)
    if load_error:
        return [load_error]
    assert data is not None

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return ["norm stats metadata is missing"]
    reasons = _common_mismatch_reasons(
        metadata,
        snapshot,
        subsets_key="subsets",
    )
    if metadata.get("state_orientation_format") != state_orientation_format:
        reasons.append("state orientation format changed")
    if metadata.get("state_dim") != 8:
        reasons.append("state dimension is not 8")
    if metadata.get("action_dim") != 7:
        reasons.append("action dimension is not 7")

    norm_stats = data.get("norm_stats")
    if not isinstance(norm_stats, dict):
        reasons.append("norm_stats object is missing")
    elif "state" not in norm_stats or "actions" not in norm_stats:
        reasons.append("state/actions normalization statistics are incomplete")
    return reasons


def prepare_libero_data(
    *,
    data_dir: str,
    subsets: List[str],
    metadata_path: str,
    norm_stats_path: str,
    state_orientation_format: str = "axis_angle",
    allow_incomplete: bool = False,
    skip_bad_files: bool = False,
    force_rebuild: bool = False,
    exclude_patterns: List[str] | None = None,
) -> Dict[str, Any]:
    """Refresh derived files when they no longer describe the current dataset."""
    snapshot = scan_libero_dataset(
        data_dir,
        subsets,
        skip_bad_files=skip_bad_files,
        exclude_patterns=exclude_patterns or (),
    )
    validate_official_counts(
        snapshot,
        allow_incomplete=allow_incomplete or skip_bad_files,
    )

    print("LIBERO dataset snapshot")
    for subset, stats in snapshot["subset_stats"].items():
        print(
            f"   {subset}: {stats['num_files']} files, "
            f"{stats['num_demos']} demos, {stats['num_steps']} steps"
        )
    print(f"   fingerprint: {snapshot['dataset_fingerprint']}")
    if snapshot["excluded_files"]:
        print("   excluded files:")
        for item in snapshot["excluded_files"]:
            print(f"      - {item['relative_path']} ({item['task']})")
    print(
        "   completeness validation: "
        f"{'relaxed' if allow_incomplete else 'strict'}"
    )

    metadata_destination = Path(metadata_path)
    norm_destination = Path(norm_stats_path)
    metadata_reasons = (
        ["forced rebuild"]
        if force_rebuild
        else _metadata_mismatch_reasons(metadata_destination, snapshot)
    )
    norm_reasons = (
        ["forced rebuild"]
        if force_rebuild
        else _norm_mismatch_reasons(
            norm_destination,
            snapshot,
            state_orientation_format,
        )
    )

    if metadata_reasons:
        print(
            f"Refreshing metadata config {metadata_destination}: "
            + "; ".join(metadata_reasons)
        )
        create_libero_meta(
            data_dir=data_dir,
            subsets=subsets,
            output_path=str(metadata_destination),
            skip_bad_files=skip_bad_files,
            allow_incomplete=allow_incomplete,
            dataset_snapshot=snapshot,
            exclude_patterns=exclude_patterns,
        )
    else:
        print(f"Metadata config is current: {metadata_destination}")

    if norm_reasons:
        print(
            f"Refreshing normalization stats {norm_destination}: "
            + "; ".join(norm_reasons)
        )
        compute_norm_stats(
            data_dir=data_dir,
            subsets=subsets,
            output_path=str(norm_destination),
            state_orientation_format=state_orientation_format,
            skip_bad_files=skip_bad_files,
            allow_incomplete=allow_incomplete,
            dataset_snapshot=snapshot,
            exclude_patterns=exclude_patterns,
        )
    else:
        print(f"Normalization stats are current: {norm_destination}")

    return {
        "snapshot": snapshot,
        "metadata_rebuilt": bool(metadata_reasons),
        "norm_stats_rebuilt": bool(norm_reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate selected LIBERO subsets and automatically rebuild stale "
            "training metadata and normalization statistics."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--subsets", nargs="+", required=True)
    parser.add_argument("--metadata_output", required=True)
    parser.add_argument("--norm_stats_output", required=True)
    parser.add_argument(
        "--state_orientation_format",
        choices=["axis_angle", "euler"],
        default="axis_angle",
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Allow intentional missing task files or demonstrations",
    )
    parser.add_argument(
        "--skip_bad_files",
        action="store_true",
        help="Skip structurally invalid HDF5 files instead of failing",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Rebuild metadata and normalization statistics unconditionally",
    )
    parser.add_argument(
        "--exclude_file",
        action="append",
        default=[],
        help="Exclude an HDF5 basename/relative-path glob; may be repeated",
    )
    args = parser.parse_args()

    prepare_libero_data(
        data_dir=args.data_dir,
        subsets=args.subsets,
        metadata_path=args.metadata_output,
        norm_stats_path=args.norm_stats_output,
        state_orientation_format=args.state_orientation_format,
        allow_incomplete=args.allow_incomplete,
        skip_bad_files=args.skip_bad_files,
        force_rebuild=args.force_rebuild,
        exclude_patterns=args.exclude_file,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
