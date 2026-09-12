# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""One-observation policy API for single-arm and multi-arm joint targets."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cosmos_framework.data.generator.action.utils.embodiment_profile import EmbodimentProfile, load_bundle
from cosmos_framework.data.generator.action.utils.embodiment_sample import (
    CosmosSampleTransform,
    build_sample,
    collate_inference,
    external_actions,
)


def validate_checkpoint_contract(saved, bundle: dict) -> None:
    # config.custom is dynamically attached by the TOML loader and is not an
    # attrs field: the training YAML serializer may omit it. The declared
    # dataloader tree does survive serialization, so bind using its factory args.
    try:
        dataset = saved["dataloader_train"]["dataloader"]["datasets"]["robots"]["dataset"]
        model = saved["model"]["config"]
        if dataset["bundle_id"] != bundle["bundle_id"]:
            raise ValueError("checkpoint config and profile/stats bundle differ")
        if model["num_embodiment_domains"] != bundle["num_embodiment_domains"] or model["max_action_dim"] != 64:
            raise ValueError("checkpoint architecture differs from the profile bundle")
        horizon = dataset["expected_horizon"]
        if any(e["profile"]["horizon"] != horizon for e in bundle["entries"]):
            raise ValueError("checkpoint/profile horizon mismatch")
        if list(model["tokenizer"]["encode_exact_durations"]) != [horizon + 1]:
            raise ValueError("checkpoint tokenizer duration mismatch")
    except KeyError as error:
        raise ValueError("missing trained policy contract in config.yaml") from error


class LeRobotPolicy:
    def __init__(self, bundle: dict, model, transform_factory=CosmosSampleTransform, collator=collate_inference):
        self.bundle, self.model, self.collator = bundle, model, collator
        self.entries = {e["profile"]["name"]: e for e in bundle["entries"]}
        self.transforms = {
            name: transform_factory(e["stats"], len(e["profile"]["joint_names"])) for name, e in self.entries.items()
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: str, config: str, profiles: str):
        """Load a local trained DCP and its own frozen config, never a public DROID alias."""
        bundle = load_bundle(profiles)
        checkpoint_path = Path(checkpoint).resolve()
        config_path = Path(config).resolve()
        if not checkpoint_path.is_dir() or not config_path.is_file():
            raise FileNotFoundError("local trained DCP and config.yaml are required")
        if checkpoint_path.parent.name != "checkpoints" or config_path != checkpoint_path.parent.parent / "config.yaml":
            raise ValueError("use config.yaml from the checkpoint's own run directory")
        import torch
        from omegaconf import OmegaConf

        saved = OmegaConf.load(config_path)
        validate_checkpoint_contract(saved, bundle)
        if not torch.cuda.is_available():
            raise RuntimeError("Cosmos inference requires CUDA")

        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.inference import OmniInference
        from cosmos_framework.scripts.action_policy_server_utils import (
            disable_runtime_ema_for_frozen_config,
            maybe_init_distributed,
        )

        maybe_init_distributed()
        setup = OmniSetupOverrides.model_validate(
            {
                "checkpoint_path": str(checkpoint_path),
                "config_file": str(config_path),
                "output_dir": checkpoint_path.parent.parent / "policy_inference",
                "sampler": "unipc",
                "parallelism_preset": "throughput",
            }
        ).build_setup()
        pipe = OmniInference.create(disable_runtime_ema_for_frozen_config(setup))
        pipe.model.eval()
        policy = cls(bundle, pipe.model)
        policy.pipe = pipe  # Keep tokenizers and model ownership alive.
        return policy

    def infer(
        self,
        observation: dict,
        *,
        seed: int = 42,
        steps: int = 30,
        guidance: float = 1.0,
        shift: float = 5.0,
        live: bool = False,
        now: float | None = None,
    ) -> dict:
        name = observation["embodiment_id"]
        if name not in self.entries:
            raise ValueError(f"unknown embodiment: {name}")
        if any(key in observation for key in ("action", "actions", "future_images")):
            raise ValueError("inference observations must not contain future teachers")
        if type(seed) is not int or type(steps) is not int or steps < 1:
            raise ValueError("integer seed and positive steps are required")
        if not np.isfinite([guidance, shift]).all() or guidance < 0 or shift <= 0:
            raise ValueError("invalid guidance/shift")
        p = EmbodimentProfile(**self.entries[name]["profile"])
        timestamp = float(observation["timestamp"])
        if not np.isfinite(timestamp):
            raise ValueError("nonfinite observation timestamp")
        if live:
            age = (time.time() if now is None else now) - timestamp
            if age < 0 or age > p.max_age_seconds:
                raise ValueError("stale/future observation timestamp")
        raw = build_sample(p, observation["task"], observation["state"], observation["image"])
        sample = self.transforms[name](raw, p.resolution)
        batch = self.collator(sample)
        import torch

        with torch.inference_mode():
            generated = self.model.generate_samples_from_batch(
                batch,
                seed=[seed],
                num_steps=steps,
                guidance=guidance,
                shift=shift,
            )
        actions = external_actions(p, generated["action"][0])
        if live:
            age = (time.time() if now is None else now) - timestamp
            if age < 0 or age > p.max_age_seconds:
                raise ValueError("observation expired during inference")
            grippers = actions[:, p.gripper_indices]
            if np.any((grippers < 0) | (grippers > 1)):
                raise ValueError("generated gripper command exceeds its calibrated range")
        return {
            "backend": self.bundle["backend"],
            "bundle_id": self.bundle["bundle_id"],
            "embodiment_id": name,
            "actions": actions.tolist(),
            "joint_names": p.joint_names,
            "units": p.units,
            "fps": p.fps,
            "observation_timestamp": timestamp,
            "seed": seed,
        }
