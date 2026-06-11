#!/usr/bin/env python3
"""Generate isolated LIBERO-PRO environment-perturbation BDDL/init files."""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

from libero_pro_runtime import (
    add_libero_pro_path_arguments,
    configured_libero_pro_environment,
    resolve_libero_pro_paths,
)


BASE_SUITES = ("libero_goal", "libero_spatial", "libero_10", "libero_object")
EXPECTED_TASKS = 10


def is_complete(directory: Path, pattern: str) -> bool:
    return directory.is_dir() and len(list(directory.glob(pattern))) >= EXPECTED_TASKS


def prepare_suite(
    suite: str,
    paths,
    bddl_parser,
    environment_perturbator,
    num_inits: int,
    seed: int,
    force: bool,
) -> None:
    source_dir = paths.benchmark_root / "bddl_files" / suite
    bddl_output_dir = paths.data_root / "bddl_files" / f"{suite}_env"
    init_output_dir = paths.data_root / "init_files" / f"{suite}_env"
    environment_config = paths.source_root / "libero_ood" / "ood_environment.yaml"
    init_generator = paths.source_root / "notebooks" / "generate_init_states.py"

    if (
        not force
        and is_complete(bddl_output_dir, "*.bddl")
        and is_complete(init_output_dir, "*.pruned_init")
    ):
        print(f"{suite}_env is already complete; skipping.")
        return

    source_files = sorted(source_dir.glob("*.bddl"))
    if len(source_files) != EXPECTED_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TASKS} source BDDL files in {source_dir}, "
            f"found {len(source_files)}."
        )
    if not environment_config.is_file():
        raise FileNotFoundError(f"Missing environment config: {environment_config}")
    if not init_generator.is_file():
        raise FileNotFoundError(f"Missing init-state generator: {init_generator}")

    bddl_output_dir.mkdir(parents=True, exist_ok=True)
    init_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating perturbed BDDL files for {suite}_env")
    for source_path in source_files:
        content = source_path.read_text(encoding="utf-8")
        parser = bddl_parser(content)
        perturbator = environment_perturbator(parser, str(environment_config))
        output = perturbator.perturb(
            task_suite_name=suite,
            task_name=source_path.stem,
            seed=seed,
        )
        (bddl_output_dir / source_path.name).write_text(output, encoding="utf-8")

    command = [
        sys.executable,
        str(init_generator),
        "--bddl_base_dir",
        str(bddl_output_dir),
        "--output_dir",
        str(init_output_dir),
        "--num_inits",
        str(num_inits),
    ]
    print(f"Generating init states for {suite}_env")
    subprocess.run(command, check=True)

    if not is_complete(init_output_dir, "*.pruned_init"):
        raise RuntimeError(f"Init-state generation did not complete for {suite}_env.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_libero_pro_path_arguments(parser)
    parser.add_argument("--suite", choices=(*BASE_SUITES, "all"), required=True)
    parser.add_argument("--num_inits", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_inits < 1:
        raise ValueError("--num_inits must be at least 1")

    paths = resolve_libero_pro_paths(args)
    suites = BASE_SUITES if args.suite == "all" else (args.suite,)

    with configured_libero_pro_environment(paths):
        perturbation = importlib.import_module("perturbation")
        for suite in suites:
            prepare_suite(
                suite=suite,
                paths=paths,
                bddl_parser=perturbation.BDDLParser,
                environment_perturbator=perturbation.EnvironmentReplacePerturbator,
                num_inits=args.num_inits,
                seed=args.seed,
                force=args.force,
            )


if __name__ == "__main__":
    main()
