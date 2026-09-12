# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared observation/window logic; torch is imported only at the Cosmos boundary."""

from __future__ import annotations

import json

import numpy as np

from cosmos_framework.data.generator.action.utils.embodiment_profile import EmbodimentProfile, validate_stats


def as_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def split_episodes(episode_ids: list[int], seed: int) -> dict[str, list[int]]:
    if len(episode_ids) < 3 or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("at least three distinct episodes are required for train/val/test")
    ids = np.random.default_rng(seed).permutation(sorted(episode_ids)).tolist()
    n = max(1, len(ids) // 10)
    return {"train": sorted(ids[2 * n :]), "val": sorted(ids[:n]), "test": sorted(ids[n : 2 * n])}


def valid_windows(episodes: list[dict], selected: list[int], horizon: int) -> list[int]:
    by_id = {int(ep["episode_index"]): ep for ep in episodes}
    if len(by_id) != len(episodes) or not set(selected) <= set(by_id):
        raise ValueError("invalid episode manifest")
    ordered = sorted(episodes, key=lambda ep: int(ep["dataset_from_index"]))
    for left, right in zip(ordered, ordered[1:]):
        if int(left["dataset_to_index"]) > int(right["dataset_from_index"]):
            raise ValueError("overlapping episode offsets")
    indices = []
    for ep_id in selected:
        ep = by_id[ep_id]
        start, stop = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        if start < 0 or stop - start != int(ep["length"]):
            raise ValueError("invalid episode offsets")
        indices.extend(range(start, max(start, stop - horizon)))
    return indices


def balanced_epoch_size(lengths: list[int], world: int = 1) -> int:
    if world < 1 or not lengths or any(n < world for n in lengths):
        raise ValueError("each selected dataset needs at least one window per rank")
    return len(lengths) * max((n + world - 1) // world for n in lengths)


def balanced_schedule(lengths: list[int], seed: int, epoch: int, rank: int = 0, world: int = 1):
    """Equal mixture, shuffled disjoint windows per worker; oversample smaller sources."""
    size = balanced_epoch_size(lengths, world)
    if not 0 <= rank < world:
        raise ValueError("invalid shard rank")
    rng = np.random.default_rng(seed + epoch)
    orders = [rng.permutation(n)[rank::world] for n in lengths]
    for i in range(size // len(lengths)):
        for source, order in enumerate(orders):
            yield source, int(order[i % len(order)])


def build_sample(profile: EmbodimentProfile, task: str, state, image, *, actions=None, future_images=None) -> dict:
    """Input state uses source feature order; image is explicit RGB HWC uint8."""
    p = profile
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be nonempty text")
    image = as_numpy(image)
    if image.dtype != np.uint8 or image.shape != (p.image_height, p.image_width, 3):
        raise ValueError("image must match profile dimensions and RGB HWC uint8")
    current = p.canonical(state, state=True)
    if current.shape != (p.dim,):
        raise ValueError("one current state vector is required")
    video = np.zeros((p.horizon + 1, p.image_height, p.image_width, 3), dtype=np.uint8)
    video[0] = image
    action = np.zeros((p.horizon + 1, p.dim), dtype=np.float32)
    action[0] = current
    if (actions is None) != (future_images is None):
        raise ValueError("training needs both future images and target actions")
    if actions is not None:
        targets = p.canonical(actions, state=False)
        future = as_numpy(future_images)
        if targets.shape != (p.horizon, p.dim) or future.shape != video[1:].shape or future.dtype != np.uint8:
            raise ValueError("teacher window shape/dtype mismatch")
        action[1:] = targets
        video[1:] = future
    return {
        "ai_caption": task.strip(),
        "video": video.transpose(3, 0, 1, 2).copy(),
        "action": action,
        "mode": "wam",
        "domain_id": p.domain_id,
        "viewpoint": "third_person_view",
        "conditioning_fps": p.fps,
    }


def sample_from_lerobot(profile: EmbodimentProfile, sample: dict) -> dict:
    p = profile
    for key, value in sample.items():
        if key.endswith("_is_pad") and as_numpy(value).any():
            raise ValueError(f"episode-end padding in {key}")
    timestamps = as_numpy(sample["timestamp"]).reshape(-1)
    if timestamps.shape != (p.horizon + 1,) or not np.allclose(np.diff(timestamps), 1 / p.fps, atol=1e-4):
        raise ValueError("timestamp window does not match control fps")
    ep = as_numpy(sample["episode_index"]).reshape(-1)
    if ep.size != p.horizon + 1 or np.any(ep != ep[0]):
        raise ValueError("window crosses episodes")
    images = as_numpy(sample[p.image_key])
    expected = (p.horizon + 1, 3, p.image_height, p.image_width)
    if images.shape != expected or not np.issubdtype(images.dtype, np.floating):
        raise ValueError("LeRobot camera must be TCHW float in [0,1]")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("LeRobot camera values must be finite and in [0,1]")
    images = np.rint(images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    state = as_numpy(sample[p.state_key])
    if state.ndim != 2 or state.shape[0] != 1:
        raise ValueError("expected only the current state")
    return build_sample(p, sample["task"], state[0], images[0], actions=sample[p.action_key], future_images=images[1:])


class CosmosSampleTransform:
    def __init__(self, stats: dict, dim: int, tokenizer_config=None, cfg_dropout_rate: float = 0.0):
        import torch

        from cosmos_framework.data.generator.action.utils.action_processing import ActionAffineNormalization
        from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline

        mean, std = validate_stats(stats, dim)
        self.normalizer = ActionAffineNormalization(offset=torch.from_numpy(mean), scale=torch.from_numpy(std))
        self.transform = ActionTransformPipeline(
            max_action_dim=64,
            action_channel_masking=True,
            format_prompt_as_json=True,
            append_idle_frames=False,
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
        )

    def __call__(self, sample: dict, resolution: str) -> dict:
        import torch

        data = {k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for k, v in sample.items()}
        for key in ("domain_id", "conditioning_fps"):
            data[key] = torch.tensor(data[key], dtype=torch.long)
        return self.transform(data, resolution, action_normalizer=self.normalizer)


def collate_inference(sample: dict) -> dict:
    """Match the action server's one-sample, multi-item batch without importing a trainer."""
    import torch

    batch = {}
    for key, value in sample.items():
        if key in ("text_token_ids", "images", "video", "action", "action_raw", "sound"):
            batch[key] = [[value]]
        elif isinstance(value, torch.Tensor):
            batch[key] = [value.unsqueeze(0)]
        else:
            batch[key] = [json.dumps(value) if key == "ai_caption" and isinstance(value, dict) else value]
    return batch


def external_actions(profile: EmbodimentProfile, generated) -> np.ndarray:
    """Cosmos already unpads/denormalizes; only remove the conditioned state row."""
    actions = as_numpy(generated).astype(np.float32)
    if actions.shape != (profile.horizon + 1, profile.dim) or not np.isfinite(actions).all():
        raise ValueError("invalid generated external action shape or nonfinite output")
    return actions[1:].copy()
