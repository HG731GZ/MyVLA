#!/usr/bin/env python3
"""Portable LIBERO-PRO path resolution and runtime configuration."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_LIBERO_PRO_ROOT = PROJECT_ROOT / "third_party" / "LIBERO-PRO"
DEFAULT_LIBERO_PRO_DATA_ROOT = PROJECT_ROOT / "third_party" / "LIBERO-PRO-data"


@dataclass(frozen=True)
class LiberoProPaths:
    source_root: Path
    data_root: Path
    benchmark_root: Path
    datasets_root: Path


def add_libero_pro_path_arguments(parser) -> None:
    parser.add_argument(
        "--libero_pro_root",
        default=os.environ.get("LIBERO_PRO_ROOT", str(DEFAULT_LIBERO_PRO_ROOT)),
        help=(
            "Official LIBERO-PRO repository root. Can also be set with "
            "LIBERO_PRO_ROOT."
        ),
    )
    parser.add_argument(
        "--libero_pro_data_root",
        "--data_root",
        dest="libero_pro_data_root",
        default=os.environ.get(
            "LIBERO_PRO_DATA_ROOT",
            str(DEFAULT_LIBERO_PRO_DATA_ROOT),
        ),
        help=(
            "Directory containing LIBERO-PRO bddl_files/ and init_files/. "
            "Can also be set with LIBERO_PRO_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--libero_datasets",
        default=os.environ.get("LIBERO_DATASETS"),
        help=(
            "Optional LIBERO datasets directory written to the runtime config. "
            "It is not used by policy rollout."
        ),
    )


def resolve_libero_pro_paths(args) -> LiberoProPaths:
    source_root = Path(args.libero_pro_root).expanduser().resolve()
    data_root = Path(args.libero_pro_data_root).expanduser().resolve()
    benchmark_root = source_root / "libero" / "libero"
    datasets_root = (
        Path(args.libero_datasets).expanduser().resolve()
        if args.libero_datasets
        else PROJECT_ROOT / "datasets"
    )

    source_marker = benchmark_root / "benchmark" / "__init__.py"
    if not source_marker.is_file():
        raise FileNotFoundError(
            f"Invalid LIBERO-PRO source root: {source_root}\n"
            f"Expected: {source_marker}\n"
            "Pass --libero_pro_root or set LIBERO_PRO_ROOT."
        )

    missing_data_dirs = [
        data_root / name
        for name in ("bddl_files", "init_files")
        if not (data_root / name).is_dir()
    ]
    if missing_data_dirs:
        missing = "\n".join(f"  - {path}" for path in missing_data_dirs)
        raise FileNotFoundError(
            f"Invalid LIBERO-PRO data root: {data_root}\n"
            f"Missing directories:\n{missing}\n"
            "Pass --libero_pro_data_root or set LIBERO_PRO_DATA_ROOT."
        )

    return LiberoProPaths(
        source_root=source_root,
        data_root=data_root,
        benchmark_root=benchmark_root,
        datasets_root=datasets_root,
    )


def _write_runtime_config(config_dir: Path, paths: LiberoProPaths) -> None:
    config = {
        "assets": paths.benchmark_root / "assets",
        "bddl_files": paths.data_root / "bddl_files",
        "benchmark_root": paths.benchmark_root,
        "datasets": paths.datasets_root,
        "init_states": paths.data_root / "init_files",
    }
    contents = "".join(
        f"{key}: {json.dumps(str(value))}\n"
        for key, value in config.items()
    )
    (config_dir / "config.yaml").write_text(contents, encoding="utf-8")


@contextmanager
def configured_libero_pro_environment(paths: LiberoProPaths) -> Iterator[Path]:
    """Configure LIBERO-PRO without modifying ~/.libero or a source checkout."""
    managed_env = (
        "LIBERO_ROOT",
        "LIBERO_PRO_ROOT",
        "LIBERO_PRO_DATA_ROOT",
        "LIBERO_CONFIG_PATH",
        "SIMVLA_AUTO_LIBERO_CONFIG",
        "MUJOCO_GL",
        "PYTHONPATH",
    )
    previous_env = {name: os.environ.get(name) for name in managed_env}
    source_path = str(paths.source_root)
    old_pythonpath = os.environ.get("PYTHONPATH")

    with tempfile.TemporaryDirectory(prefix="myvla-libero-pro-config-") as config_dir:
        config_path = Path(config_dir)
        _write_runtime_config(config_path, paths)

        os.environ["LIBERO_ROOT"] = source_path
        os.environ["LIBERO_PRO_ROOT"] = source_path
        os.environ["LIBERO_PRO_DATA_ROOT"] = str(paths.data_root)
        os.environ["LIBERO_CONFIG_PATH"] = str(config_path)
        os.environ["SIMVLA_AUTO_LIBERO_CONFIG"] = "0"
        os.environ["PYTHONPATH"] = (
            source_path
            if not old_pythonpath
            else source_path + os.pathsep + old_pythonpath
        )
        os.environ.setdefault("MUJOCO_GL", "egl")

        inserted = source_path not in sys.path
        if inserted:
            sys.path.insert(0, source_path)

        try:
            yield config_path
        finally:
            if inserted:
                try:
                    sys.path.remove(source_path)
                except ValueError:
                    pass
            for name, value in previous_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
