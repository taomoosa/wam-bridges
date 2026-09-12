# SPDX-FileCopyrightText: Copyright (c) 2026 WAM Bridges contributors
# SPDX-License-Identifier: MIT
"""Prepare a local LeRobot SFT run or launch the existing Cosmos trainer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cosmos_framework.data.generator.action.utils.embodiment_profile import (
    digest,
    load_bundle,
    load_profiles,
    write_json,
)
from cosmos_framework.data.generator.action.utils.embodiment_sample import split_episodes
from tools.compute_action_stats import compute_stats


def prepare(args) -> Path:
    profiles = load_profiles(args.profiles)
    if args.embodiments:
        requested = set(args.embodiments)
        if not requested <= {p.name for p in profiles}:
            raise ValueError("unknown selected embodiment")
        profiles = [p for p in profiles if p.name in requested]
    if len({p.horizon for p in profiles}) != 1:
        raise ValueError("this small recipe uses one shared horizon per run")
    from cosmos_framework.data.generator.action.datasets import configurable_lerobot_dataset as adapter

    for path in (args.base_checkpoint, args.vae):
        if not path or not Path(path).exists():
            raise FileNotFoundError("--base-checkpoint and --vae must be staged locally")
    entries = []
    for p in profiles:
        reader = adapter.open_reader(p)
        p.validate_metadata(reader.meta.info)
        ids = [int(e["episode_index"]) for e in reader.meta.episodes]
        splits = split_episodes(ids, args.seed)
        stats = compute_stats(adapter.audit_rows(p, reader, splits["train"]), seed=args.seed)
        counts = {}
        for split, selected in splits.items():
            dataset = adapter.ConfigurableLeRobotDataset(p, selected, reader=reader)
            dataset[0]  # Decode a preview; every consumed training window is validated too.
            counts[split] = len(dataset)
        entries.append(
            {
                "profile": p.to_dict(),
                "stats": stats,
                "splits": splits,
                "window_counts": counts,
                "metadata_sha256": digest(reader.meta.info),
            }
        )
    body = {
        "schema_version": 2,
        "backend": "cosmos",
        "model_family": "nano",
        "seed": args.seed,
        "max_action_dim": 64,
        "num_embodiment_domains": max(p.domain_id for p in profiles) + 1,
        "provenance": {
            "base_revision": args.base_revision,
            "base_checkpoint": str(args.base_checkpoint),
            "vae": str(args.vae),
        },
        "entries": entries,
    }
    bundle = {**body, "bundle_id": digest(body)}
    run = Path(args.run_dir).resolve()
    # Never silently rewrite the contract of a previous run.
    run.mkdir(parents=True, exist_ok=False)
    bundle_path = run / "profiles.json"
    write_json(bundle_path, bundle)
    template = Path(__file__).with_name("sft.toml").read_text()
    for token, value in {
        "__VAE__": str(Path(args.vae).resolve()),
        "__BASE__": str(Path(args.base_checkpoint).resolve()),
        "__BUNDLE__": str(bundle_path),
        "__ID__": bundle["bundle_id"],
    }.items():
        template = template.replace(json.dumps(token), json.dumps(value))
    template = template.replace("__HORIZON__", str(profiles[0].horizon)).replace("__SEED__", str(args.seed))
    template = template.replace("__NUM_DOMAINS__", str(body["num_embodiment_domains"]))
    (run / "sft.toml").write_text(template)
    return run


def training_command(run: Path, gpus: int, *, dryrun: bool = False) -> list[str]:
    if gpus < 1:
        raise ValueError("--gpus must be positive")
    with (run / "sft.toml").open("rb") as f:
        recipe = tomllib.load(f)
    c = recipe["custom"]
    bundle = load_bundle(c["bundle_path"])
    if bundle["bundle_id"] != c["bundle_id"]:
        raise ValueError("recipe/bundle mismatch")
    if c["seed"] != bundle["seed"] or any(e["profile"]["horizon"] != c["horizon"] for e in bundle["entries"]):
        raise ValueError("recipe seed/horizon differs from the prepared contract")
    if c["num_domains"] != bundle["num_embodiment_domains"]:
        raise ValueError("recipe domain count differs from the prepared contract")
    if any(e["window_counts"]["train"] < gpus for e in bundle["entries"]):
        raise ValueError("each mechanism needs at least one train window per GPU")
    # No shell interpolation: paths, spaces and metacharacters remain arguments.
    command = [sys.executable]
    if not dryrun:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={gpus}"]
    command += ["-m", "cosmos_framework.scripts.train", f"--sft-toml={run / 'sft.toml'}"]
    if dryrun:
        command.append("--dryrun")
    target = "dataloader_train.dataloader.datasets.robots.dataset"
    command += [
        "--",
        f"{target}.bundle_path={json.dumps(c['bundle_path'])}",
        f"{target}.bundle_id={c['bundle_id']}",
        f"{target}.expected_horizon={c['horizon']}",
        f"model.config.tokenizer.encode_exact_durations=[{c['horizon'] + 1}]",
        f"model.config.num_embodiment_domains={c['num_domains']}",
        f"trainer.seed={c['seed']}",
    ]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--dryrun", action="store_true")
    parser.add_argument("--profiles", type=Path, default=Path(__file__).with_name("profiles.json"))
    parser.add_argument("--embodiments", nargs="+")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-checkpoint", default=os.getenv("BASE_CHECKPOINT_PATH"))
    parser.add_argument("--vae", default=os.getenv("WAN_VAE_PATH"))
    parser.add_argument(
        "--base-revision", default="unspecified", help="record the exact revision used for base conversion"
    )
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args))
    else:
        run = Path(args.run_dir).resolve()
        command = training_command(run, args.gpus, dryrun=args.dryrun)
        env = dict(os.environ, IMAGINAIRE_OUTPUT_ROOT=str(run / "training"))
        subprocess.run(command, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
