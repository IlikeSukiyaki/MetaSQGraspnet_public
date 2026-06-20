"""Shared path helpers for SuperSQ debug inspection tools."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "supersq_grasp-v0"
DEBUG_ROOT = REPO_ROOT / "logs" / "sq_debug" / TASK_NAME
ATTEMPT_PATTERN = re.compile(
    r"^env(?P<env_id>\d+)_epi(?P<episode>\d+)_attempt(?P<attempt>\d+)_obj(?P<obj_id>.+)$"
)


def list_runs() -> list[Path]:
    if not DEBUG_ROOT.exists():
        return []
    return sorted(
        (path for path in DEBUG_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )


def resolve_run_dir(run: str) -> Path:
    candidate = Path(run).expanduser()
    if candidate.exists():
        return candidate.resolve()

    run_dir = DEBUG_ROOT / run
    if run_dir.exists():
        return run_dir.resolve()

    matches = sorted(DEBUG_ROOT.glob(f"*{run}*"))
    matches = [path for path in matches if path.is_dir()]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise FileNotFoundError(f"Run '{run}' is ambiguous. Matches: {names}")

    raise FileNotFoundError(
        f"Could not find debug run '{run}' under {DEBUG_ROOT}. "
        "Use --list-runs to see available timestamps."
    )


def latest_run_dir() -> Path:
    runs = list_runs()
    if not runs:
        raise FileNotFoundError(f"No debug runs found under {DEBUG_ROOT}.")
    return runs[0]


def list_attempt_dirs(run_dir: Path) -> list[Path]:
    attempts = [path for path in run_dir.iterdir() if path.is_dir() and ATTEMPT_PATTERN.match(path.name)]
    return sorted(attempts, key=lambda path: path.name)


def resolve_attempt_dir(
    run_dir: Path,
    *,
    env_id: int = 0,
    episode: int = 0,
    attempt: int = 0,
    obj_id: str | None = None,
) -> Path:
    if obj_id is not None:
        candidate = run_dir / f"env{env_id}_epi{episode:06d}_attempt{attempt}_obj{obj_id}"
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Missing debug attempt folder: {candidate}")

    pattern = f"env{env_id}_epi{episode:06d}_attempt{attempt}_obj*"
    matches = sorted(run_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"No debug attempt folders matching {pattern} under {run_dir}. "
            "Use --list-episodes to inspect this run."
        )
    names = ", ".join(path.name for path in matches)
    raise FileNotFoundError(
        f"Multiple attempt folders match {pattern}: {names}. Pass --obj-id explicitly."
    )


def resolve_debug_dir(
    *,
    debug_dir: Path | None = None,
    run: str | None = None,
    latest: bool = False,
    episode: int = 0,
    attempt: int = 0,
    env_id: int = 0,
    obj_id: str | None = None,
) -> Path:
    if debug_dir is not None:
        path = debug_dir.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Debug folder not found: {path}")
        return path

    if latest:
        run_dir = latest_run_dir()
    elif run:
        run_dir = resolve_run_dir(run)
    else:
        raise ValueError("Provide --dir, --run, or --latest.")

    return resolve_attempt_dir(
        run_dir,
        env_id=env_id,
        episode=episode,
        attempt=attempt,
        obj_id=obj_id,
    )


def add_debug_dir_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("debug target")
    group.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Full path to one env*_epi*_attempt*_obj* folder.",
    )
    group.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run timestamp or folder name, e.g. 2026-06-09_19-01-25.",
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest run under logs/sq_debug/supersq_grasp-v0/.",
    )
    group.add_argument("--episode", type=int, default=0, help="Episode index within the run.")
    group.add_argument("--attempt", type=int, default=0, help="Grasp retry attempt index.")
    group.add_argument("--env-id", type=int, default=0, help="Environment index.")
    group.add_argument("--obj-id", type=str, default=None, help="Object ID if the pattern is ambiguous.")
    group.add_argument(
        "--list-runs",
        action="store_true",
        help="List available debug run folders and exit.",
    )
    group.add_argument(
        "--list-episodes",
        action="store_true",
        help="List attempt folders for --run or --latest and exit.",
    )


def resolve_debug_dir_from_args(args: argparse.Namespace) -> Path:
    if args.list_runs:
        runs = list_runs()
        if not runs:
            print(f"No runs under {DEBUG_ROOT}")
        else:
            for run_dir in runs:
                print(run_dir.name)
        sys.exit(0)

    if args.list_episodes:
        if args.dir:
            run_dir = args.dir.expanduser().resolve()
        elif args.latest:
            run_dir = latest_run_dir()
        elif args.run:
            run_dir = resolve_run_dir(args.run)
        else:
            raise SystemExit("--list-episodes requires --run, --latest, or --dir pointing to a run folder.")

        attempts = list_attempt_dirs(run_dir)
        if not attempts:
            print(f"No attempt folders under {run_dir}")
        else:
            for attempt_dir in attempts:
                print(attempt_dir.name)
        sys.exit(0)

    selected = sum(
        1 for value in (args.dir is not None, args.run is not None, args.latest) if value
    )
    if selected != 1:
        raise SystemExit("Provide exactly one of --dir, --run, or --latest.")

    return resolve_debug_dir(
        debug_dir=args.dir,
        run=args.run,
        latest=args.latest,
        episode=args.episode,
        attempt=args.attempt,
        env_id=args.env_id,
        obj_id=args.obj_id,
    )
