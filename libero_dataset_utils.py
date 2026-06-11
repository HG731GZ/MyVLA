"""Shared LIBERO dataset scanning, validation, and fingerprint helpers."""

from __future__ import annotations

import glob
import hashlib
import fnmatch
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import h5py


DATASET_FINGERPRINT_VERSION = 1
EXPECTED_SUBSET_FILES = {
    "libero_10": 10,
    "libero_goal": 10,
    "libero_object": 10,
    "libero_spatial": 10,
    "libero_90": 90,
}
EXPECTED_DEMOS_PER_FILE = 50
EXPECTED_FULL_LIBERO_STEPS = 1007618
REQUIRED_DATASET_KEYS = (
    "actions",
    "obs/agentview_rgb",
    "obs/eye_in_hand_rgb",
    "obs/ee_pos",
    "obs/ee_ori",
    "obs/gripper_states",
)


def normalize_subsets(subsets: Sequence[str]) -> List[str]:
    return [subset.strip("/").replace("\\", "/") for subset in subsets]


def resolve_subset_dirs(
    data_dir: str,
    subsets: Sequence[str],
) -> tuple[List[tuple[str, str]], List[str]]:
    """Resolve requested subset names for flat and grouped LIBERO layouts."""
    resolved: List[tuple[str, str]] = []
    missing: List[str] = []

    def add_subset(label: str, rel_path: str) -> None:
        subset_dir = os.path.join(data_dir, rel_path)
        if os.path.isdir(subset_dir):
            resolved.append((label, subset_dir))
        else:
            missing.append(rel_path)

    for normalized in normalize_subsets(subsets):
        if normalized == "libero_100":
            add_subset("libero_10", "libero_100/libero_10")
            add_subset("libero_90", "libero_100/libero_90")
        elif normalized in {"libero_10", "libero_90"}:
            flat_dir = os.path.join(data_dir, normalized)
            if os.path.isdir(flat_dir):
                resolved.append((normalized, flat_dir))
            else:
                add_subset(normalized, f"libero_100/{normalized}")
        else:
            add_subset(os.path.basename(normalized), normalized)

    return resolved, missing


def parse_task_from_filename(filepath: str) -> str:
    """Parse the canonical LIBERO task text from an HDF5 filename."""
    task = re.sub(r"_demo\.hdf5$", "", os.path.basename(filepath))
    scene_match = re.search(r"SCENE\d+_", task)
    if scene_match:
        task = task[scene_match.end():]
    return task.replace("_", " ")


def inspect_h5(h5_path: str) -> tuple[int, int]:
    """Validate one training HDF5 and return its demo and timestep counts."""
    num_demos = 0
    num_steps = 0

    with h5py.File(h5_path, "r") as h5_file:
        if "data" not in h5_file:
            raise ValueError("missing data group")

        demo_keys = sorted(
            key for key in h5_file["data"].keys() if key.startswith("demo")
        )
        if not demo_keys:
            raise ValueError("no demo_* groups found")

        for demo_key in demo_keys:
            demo = h5_file["data"][demo_key]
            for required_key in REQUIRED_DATASET_KEYS:
                if required_key not in demo:
                    raise ValueError(f"{demo_key} missing {required_key}")

            lengths = [len(demo[key]) for key in REQUIRED_DATASET_KEYS]
            timesteps = min(lengths)
            if timesteps <= 0:
                raise ValueError(f"{demo_key} has empty required data")
            num_demos += 1
            num_steps += timesteps

    return num_demos, num_steps


def _format_h5_errors(errors: List[Dict[str, str]]) -> str:
    lines = ["Unreadable or invalid HDF5 files:"]
    for item in errors:
        lines.append(f"  - [{item['subset']}] {item['path']}: {item['error']}")
    return "\n".join(lines)


def _fingerprint_payload(
    requested_subsets: Sequence[str],
    exclude_patterns: Sequence[str],
    files: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "version": DATASET_FINGERPRINT_VERSION,
        "requested_subsets": normalize_subsets(requested_subsets),
        "exclude_patterns": sorted(exclude_patterns),
        "files": [
            {
                "subset": item["subset"],
                "relative_path": item["relative_path"],
                "size": item["size"],
                "mtime_ns": item["mtime_ns"],
                "num_demos": item["num_demos"],
                "num_steps": item["num_steps"],
            }
            for item in files
        ],
    }


def scan_libero_dataset(
    data_dir: str,
    subsets: Sequence[str],
    *,
    skip_bad_files: bool = False,
    exclude_patterns: Sequence[str] = (),
) -> Dict[str, Any]:
    """Scan selected subsets and return a deterministic dataset snapshot."""
    data_root = Path(data_dir).expanduser().resolve()
    resolved_subsets, missing_subsets = resolve_subset_dirs(
        str(data_root),
        subsets,
    )
    files: List[Dict[str, Any]] = []
    subset_stats: Dict[str, Dict[str, int]] = {}
    bad_files: List[Dict[str, str]] = []
    excluded_files: List[Dict[str, str]] = []
    normalized_exclude_patterns = sorted(
        pattern for pattern in exclude_patterns if pattern
    )
    matched_exclude_patterns = set()

    for subset, subset_dir in resolved_subsets:
        subset_files = 0
        subset_demos = 0
        subset_steps = 0

        for h5_path in sorted(glob.glob(os.path.join(subset_dir, "*.hdf5"))):
            relative_path = Path(
                os.path.relpath(os.path.abspath(h5_path), data_root)
            ).as_posix()
            basename = os.path.basename(h5_path)
            matching_patterns = [
                pattern
                for pattern in normalized_exclude_patterns
                if fnmatch.fnmatch(basename, pattern)
                or fnmatch.fnmatch(relative_path, pattern)
            ]
            if matching_patterns:
                matched_exclude_patterns.update(matching_patterns)
                excluded_files.append(
                    {
                        "subset": subset,
                        "relative_path": relative_path,
                        "task": parse_task_from_filename(h5_path),
                    }
                )
                continue

            try:
                num_demos, num_steps = inspect_h5(h5_path)
            except Exception as error:
                bad_files.append(
                    {
                        "subset": subset,
                        "path": h5_path,
                        "error": str(error),
                    }
                )
                if skip_bad_files:
                    print(
                        "Warning: Skipping unreadable or invalid file: "
                        f"{h5_path}: {error}"
                    )
                continue

            stat = os.stat(h5_path)
            absolute_path = str(Path(h5_path).resolve())
            files.append(
                {
                    "subset": subset,
                    "path": absolute_path,
                    "relative_path": relative_path,
                    "task": parse_task_from_filename(h5_path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "num_demos": num_demos,
                    "num_steps": num_steps,
                }
            )
            subset_files += 1
            subset_demos += num_demos
            subset_steps += num_steps

        subset_stats[subset] = {
            "num_files": subset_files,
            "num_demos": subset_demos,
            "num_steps": subset_steps,
        }

    unmatched_patterns = sorted(
        set(normalized_exclude_patterns) - matched_exclude_patterns
    )
    if unmatched_patterns:
        raise RuntimeError(
            "The following LIBERO_EXCLUDE_FILES/--exclude_file patterns did "
            "not match any HDF5 file:\n"
            + "\n".join(f"  - {pattern}" for pattern in unmatched_patterns)
        )
    if bad_files and not skip_bad_files:
        raise RuntimeError(
            f"Found {len(bad_files)} bad HDF5 file(s).\n"
            f"{_format_h5_errors(bad_files)}"
        )
    if not files:
        raise RuntimeError(
            f"No valid LIBERO HDF5 files found under {data_root} "
            f"for subsets {list(subsets)}."
        )

    files.sort(key=lambda item: (item["subset"], item["relative_path"]))
    payload = _fingerprint_payload(subsets, normalized_exclude_patterns, files)
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "fingerprint_version": DATASET_FINGERPRINT_VERSION,
        "dataset_fingerprint": fingerprint,
        "data_dir": str(data_root),
        "requested_subsets": normalize_subsets(subsets),
        "exclude_patterns": normalized_exclude_patterns,
        "excluded_files": excluded_files,
        "resolved_subsets": [
            {"name": name, "path": str(Path(path).resolve())}
            for name, path in resolved_subsets
        ],
        "missing_subsets": missing_subsets,
        "files": files,
        "subset_stats": subset_stats,
        "num_files": len(files),
        "num_demos": sum(item["num_demos"] for item in files),
        "num_steps": sum(item["num_steps"] for item in files),
    }


def validate_official_counts(
    snapshot: Dict[str, Any],
    *,
    allow_incomplete: bool,
) -> None:
    """Enforce official LIBERO counts unless leave-out experiments are enabled."""
    if allow_incomplete:
        return

    problems = []
    if snapshot["missing_subsets"]:
        problems.extend(
            f"missing requested subset directory: {subset}"
            for subset in snapshot["missing_subsets"]
        )

    subset_stats = snapshot["subset_stats"]
    for subset, stats in subset_stats.items():
        expected_files = EXPECTED_SUBSET_FILES.get(subset)
        if expected_files is None:
            continue
        expected_demos = expected_files * EXPECTED_DEMOS_PER_FILE
        if stats["num_files"] != expected_files:
            problems.append(
                f"{subset}: {stats['num_files']} files, expected {expected_files}"
            )
        if stats["num_demos"] != expected_demos:
            problems.append(
                f"{subset}: {stats['num_demos']} demos, expected {expected_demos}"
            )

    if set(subset_stats) == set(EXPECTED_SUBSET_FILES):
        if snapshot["num_steps"] != EXPECTED_FULL_LIBERO_STEPS:
            problems.append(
                "full LIBERO: "
                f"{snapshot['num_steps']} steps, expected {EXPECTED_FULL_LIBERO_STEPS}"
            )

    if problems:
        detail = "\n".join(f"  - {problem}" for problem in problems)
        raise RuntimeError(
            "LIBERO dataset does not match the official complete split. "
            "Set LIBERO_STRICT_VALIDATION=0 or pass --allow_incomplete for "
            "intentional leave-one-task/demo-out experiments.\n"
            f"{detail}"
        )


def snapshot_manifest(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the portable file inventory stored in generated JSON files."""
    return _fingerprint_payload(
        snapshot["requested_subsets"],
        snapshot["exclude_patterns"],
        snapshot["files"],
    )["files"]


def atomic_json_dump(data: Dict[str, Any], output_path: str) -> None:
    """Atomically replace a JSON file after fully writing it."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
            json.dump(data, output_file, indent=2)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temp_path, destination)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise
