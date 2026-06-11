#!/usr/bin/env python3
"""Evaluate a MyVLA policy server on LIBERO-PRO suites."""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

from libero_pro_runtime import (
    add_libero_pro_path_arguments,
    configured_libero_pro_environment,
    resolve_libero_pro_paths,
)


SCRIPT_DIR = Path(__file__).resolve().parent
LIBERO_CLIENT_DIR = SCRIPT_DIR.parent / "libero"

BASE_SUITES = ("libero_goal", "libero_spatial", "libero_10", "libero_object")
CANONICAL_PERTURBATIONS = ("object", "position", "semantic", "task", "environment")
PERTURBATION_SUFFIXES = {
    "object": "object",
    "position": "swap",
    "swap": "swap",
    "semantic": "lan",
    "language": "lan",
    "lan": "lan",
    "task": "task",
    "environment": "env",
    "env": "env",
}
PRO_MAX_STEPS = {
    "libero_goal": 300,
    "libero_spatial": 220,
    "libero_10": 520,
    "libero_object": 280,
}
LANGUAGE_PATTERN = re.compile(r"\(:language\s+(.*?)\)", re.DOTALL)


def parse_bddl_language(bddl_path: Path, fallback: str) -> str:
    """Read the prompt from BDDL so semantic and task changes are preserved."""
    content = bddl_path.read_text(encoding="utf-8")
    match = LANGUAGE_PATTERN.search(content)
    if not match:
        print(f"[warning] No (:language ...) block in {bddl_path}; using task name.")
        return fallback
    return " ".join(match.group(1).split())


def make_libero_pro_env_factory(base):
    def get_libero_pro_env(task, resolution: int, seed: int):
        bddl_path = (
            Path(base.get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env_args = {
            "bddl_file_name": str(bddl_path),
            "camera_heights": resolution,
            "camera_widths": resolution,
        }
        env = base.OffScreenRenderEnv(**env_args)
        env.seed(seed)
        return env, parse_bddl_language(bddl_path, task.language)

    return get_libero_pro_env


def actual_suite_name(base_suite: str, perturbation: str) -> str:
    return f"{base_suite}_{PERTURBATION_SUFFIXES[perturbation]}"


def validate_suite(base, suite_name: str) -> None:
    if suite_name not in base.benchmark_dict:
        raise RuntimeError(
            f"LIBERO-PRO suite {suite_name!r} is not registered. Check that "
            "--libero_pro_root points to the official LIBERO-PRO repository."
        )

    bddl_dir = Path(base.get_libero_path("bddl_files")) / suite_name
    init_dir = Path(base.get_libero_path("init_states")) / suite_name
    if not bddl_dir.is_dir() or not init_dir.is_dir():
        hint = ""
        if suite_name.endswith("_env"):
            hint = " Generate it with prepare_environment_suite.sh first."
        raise FileNotFoundError(
            f"Missing LIBERO-PRO data for {suite_name}: expected "
            f"{bddl_dir} and {init_dir}.{hint}"
        )

    task_suite = base.benchmark_dict[suite_name]()
    missing = []
    for task in task_suite.tasks:
        bddl_path = bddl_dir / task.bddl_file
        init_path = init_dir / task.init_states_file
        if not bddl_path.is_file():
            missing.append(str(bddl_path))
        if not init_path.is_file():
            missing.append(str(init_path))
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:8])
        raise FileNotFoundError(
            f"Incomplete LIBERO-PRO data for {suite_name}. Missing files:\n{preview}"
        )


def parse_task_ids(raw_task_ids: str | None) -> List[int] | None:
    if not raw_task_ids:
        return None
    return [
        int(value)
        for value in raw_task_ids.replace(" ", "").split(",")
        if value
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("MyVLA LIBERO-PRO Evaluation Client")
    add_libero_pro_path_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--connection_info", default=None)
    parser.add_argument("--client_type", choices=["websocket", "http"], default="websocket")
    parser.add_argument("--task_suite", choices=BASE_SUITES, default="libero_object")
    parser.add_argument(
        "--perturbation",
        choices=tuple(PERTURBATION_SUFFIXES),
        default="position",
        help="LIBERO-PRO generalization dimension.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all four base suites across all five perturbations.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate source/data paths and suite registration without connecting to a server.",
    )
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--video_out", default="./eval_results_libero_pro")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--task_ids", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no_clip_actions", action="store_true")
    return parser


def load_connection_info(args: argparse.Namespace) -> None:
    if not args.connection_info:
        return
    info_path = Path(args.connection_info)
    print(f"Loading connection info from: {info_path}")
    while not info_path.exists():
        sys.stdout.write("\rWaiting for server...")
        sys.stdout.flush()
        time.sleep(0.5)
    print()
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    args.host = info["host"]
    args.port = info["port"]


def create_client(base, args: argparse.Namespace):
    if args.client_type == "websocket":
        return base.WebSocketClient(
            args.host,
            args.port,
            replan_steps=args.replan_steps,
            clip_actions=not args.no_clip_actions,
        )
    return base.HTTPClient(
        args.host,
        args.port,
        replan_steps=args.replan_steps,
        clip_actions=not args.no_clip_actions,
    )


def requested_runs(args: argparse.Namespace) -> List[tuple[str, str]]:
    if args.all:
        return [
            (suite, perturbation)
            for suite in BASE_SUITES
            for perturbation in CANONICAL_PERTURBATIONS
        ]
    return [(args.task_suite, args.perturbation)]


def _import_base_client():
    client_path = str(LIBERO_CLIENT_DIR)
    if client_path not in sys.path:
        sys.path.insert(0, client_path)
    return importlib.import_module("libero_client")


def main() -> None:
    args = build_parser().parse_args()
    if args.num_trials < 1:
        raise ValueError("--num_trials must be at least 1")
    if args.replan_steps < 1:
        raise ValueError("--replan_steps must be at least 1")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max_steps must be at least 1")

    paths = resolve_libero_pro_paths(args)
    runs = requested_runs(args)
    suite_names = [
        actual_suite_name(suite, perturbation)
        for suite, perturbation in runs
    ]

    with configured_libero_pro_environment(paths) as config_dir:
        base = _import_base_client()
        for suite_name in suite_names:
            validate_suite(base, suite_name)

        print(f"LIBERO-PRO source: {paths.source_root}")
        print(f"LIBERO-PRO data: {paths.data_root}")
        print(f"Runtime config: {config_dir / 'config.yaml'}")

        if args.validate_only:
            print(f"Validated suites: {', '.join(suite_names)}")
            return

        load_connection_info(args)
        protocol = "ws" if args.client_type == "websocket" else "http"
        print("Starting LIBERO-PRO evaluation client")
        print(f"   Client type: {args.client_type}")
        print(f"   Server: {protocol}://{args.host}:{args.port}")
        print(f"   Runs: {len(runs)}")
        print(f"   Replan steps: {args.replan_steps}")
        print(f"   Clip actions: {not args.no_clip_actions}")
        print()

        base.get_libero_env = make_libero_pro_env_factory(base)
        client = create_client(base, args)
        task_ids = parse_task_ids(args.task_ids)
        results: Dict[str, float] = {}

        for (base_suite, perturbation), suite_name in zip(runs, suite_names):
            max_steps = (
                args.max_steps
                if args.max_steps is not None
                else PRO_MAX_STEPS[base_suite]
            )
            print("=" * 72)
            print(f"LIBERO-PRO run: {suite_name} ({perturbation})")
            print("=" * 72)
            results[suite_name] = base.eval_libero(
                client=client,
                task_suite_name=suite_name,
                num_trials=args.num_trials,
                seed=args.seed,
                video_out_path=str(Path(args.video_out) / suite_name),
                save_video=not args.no_video,
                task_ids=task_ids,
                max_steps_override=max_steps,
            )

        print("\nLIBERO-PRO summary")
        print("=" * 72)
        for suite_name, success_rate in results.items():
            print(f"{suite_name}: {success_rate:.4f}")


if __name__ == "__main__":
    main()
