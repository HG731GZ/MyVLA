#!/usr/bin/env python
"""
Create LIBERO Dataset Training Metadata Configuration

Usage:
    python create_libero_meta.py \\
        --data_dir /datasets/metas \\
        --output ./datasets/metas/libero_train.json

This will scan the LIBERO dataset directory and generate a metadata file 
containing all HDF5 file paths and task descriptions.

Note: Each LIBERO HDF5 file contains 50 demos (episodes).
"""

import argparse
from typing import Any, Dict, List

from libero_dataset_utils import (
    atomic_json_dump,
    scan_libero_dataset,
    snapshot_manifest,
    validate_official_counts,
)


def create_libero_meta(
    data_dir: str,
    subsets: List[str] = None,
    output_path: str = None,
    skip_bad_files: bool = False,
    allow_incomplete: bool = False,
    dataset_snapshot: Dict[str, Any] | None = None,
    exclude_patterns: List[str] | None = None,
) -> Dict:
    """
    Create LIBERO dataset meta configuration.
    
    Args:
        data_dir: LIBERO dataset root directory
        subsets: List of subsets to include
        output_path: Output JSON path
        
    Returns:
        meta dictionary
    """
    if subsets is None:
        # Default 4 subsets (excluding libero_90)
        subsets = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    
    print(f"Scanning LIBERO dataset: {data_dir}")
    snapshot = dataset_snapshot or scan_libero_dataset(
        data_dir,
        subsets,
        skip_bad_files=skip_bad_files,
        exclude_patterns=exclude_patterns or (),
    )
    validate_official_counts(
        snapshot,
        allow_incomplete=allow_incomplete or skip_bad_files,
    )

    for subset, stats in snapshot["subset_stats"].items():
        print(
            f"   {subset}: {stats['num_files']} files, "
            f"{stats['num_demos']} demos, {stats['num_steps']} steps"
        )

    datalist = [
        {
            "path": item["path"],
            "task": item["task"],
            "subset": item["subset"],
            "num_demos": item["num_demos"],
            "num_steps": item["num_steps"],
        }
        for item in snapshot["files"]
    ]
    
    meta = {
        "dataset_name": "libero_hdf5",
        "data_dir": snapshot["data_dir"],
        "datalist": datalist,
        "num_files": snapshot["num_files"],
        "num_episodes": snapshot["num_demos"],
        "num_steps": snapshot["num_steps"],
        "subsets": list(snapshot["subset_stats"].keys()),
        "requested_subsets": snapshot["requested_subsets"],
        "exclude_patterns": snapshot["exclude_patterns"],
        "excluded_files": snapshot["excluded_files"],
        "subset_stats": snapshot["subset_stats"],
        "dataset_fingerprint": snapshot["dataset_fingerprint"],
        "dataset_fingerprint_version": snapshot["fingerprint_version"],
        "dataset_manifest": snapshot_manifest(snapshot),
        "observation_key": ["obs/agentview_rgb", "obs/eye_in_hand_rgb"],
        "action_key": "actions",
        "state_dim": 8,
        "action_dim": 7,
        "fps": 10,
    }
    
    print(
        f"\nFound {snapshot['num_files']} HDF5 files, "
        f"{snapshot['num_demos']} demos, {snapshot['num_steps']} steps"
    )
    print(f"Dataset fingerprint: {snapshot['dataset_fingerprint']}")
    
    if output_path:
        atomic_json_dump(meta, output_path)
        print(f"Saved to: {output_path}")
    
    return meta


def main():
    parser = argparse.ArgumentParser(description="Create LIBERO training metadata")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="LIBERO dataset root directory")
    parser.add_argument("--subsets", type=str, nargs="+",
                        default=["libero_10", "libero_goal", "libero_object", "libero_spatial"],
                        help="Subsets to include (default 4 subsets, excluding libero_90)")
    parser.add_argument("--output", type=str,
                        default="./datasets/metas/libero_train.json",
                        help="Output file path")
    parser.add_argument("--validate_only", action="store_true",
                        help="Validate the dataset without writing metadata")
    parser.add_argument("--skip_bad_files", action="store_true",
                        help="Skip unreadable files instead of failing (debug only)")
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="Do not enforce official LIBERO file/demo counts")
    parser.add_argument(
        "--exclude_file",
        action="append",
        default=[],
        help="Exclude an HDF5 basename/relative-path glob; may be repeated",
    )
    
    args = parser.parse_args()
    
    create_libero_meta(
        data_dir=args.data_dir,
        subsets=args.subsets,
        output_path=None if args.validate_only else args.output,
        skip_bad_files=args.skip_bad_files,
        allow_incomplete=args.allow_incomplete,
        exclude_patterns=args.exclude_file,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(str(e)) from None
