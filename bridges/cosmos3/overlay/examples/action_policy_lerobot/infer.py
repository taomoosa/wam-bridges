# SPDX-FileCopyrightText: Copyright (c) 2026 WAM Bridges contributors
# SPDX-License-Identifier: MIT
"""Return an action chunk from a JSON observation; no robot SDK is invoked."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cosmos_framework.data.generator.action.utils.embodiment_profile import (
    read_json,
    write_json,
)
from cosmos_framework.inference.lerobot_policy import LeRobotPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument(
        "--live", action="store_true", help="require a recent Unix-time observation; offline replay is default"
    )
    args = parser.parse_args()
    policy = LeRobotPolicy.from_checkpoint(args.checkpoint, args.config, args.profiles)
    obs = read_json(args.input)
    from PIL import Image

    with Image.open(args.input.resolve().parent / obs.pop("image_path")) as image:
        if image.mode != "RGB":
            raise ValueError("input image must be RGB")
        obs["image"] = np.array(image)
    output = policy.infer(
        obs, seed=args.seed, steps=args.steps, guidance=args.guidance, shift=args.shift, live=args.live
    )
    output["checkpoint"] = str(Path(args.checkpoint).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)
    print(f"{output['backend']}: {len(output['actions'])} x {len(output['joint_names'])} -> {args.output}")


if __name__ == "__main__":
    main()
