# SPDX-FileCopyrightText: Copyright (c) 2026 WAM Bridges contributors
# SPDX-License-Identifier: MIT
"""Explicit LeRobot v3 joint-position adapter for the small multi-robot example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cosmos_framework.data.generator.action.utils.embodiment_profile import EmbodimentProfile, digest, load_bundle
from cosmos_framework.data.generator.action.utils.embodiment_sample import (
    CosmosSampleTransform,
    as_numpy,
    balanced_epoch_size,
    balanced_schedule,
    sample_from_lerobot,
    valid_windows,
)


def open_reader(profile: EmbodimentProfile):
    if not (Path(profile.root) / "meta/info.json").is_file():
        raise FileNotFoundError(f"local LeRobot metadata missing: {profile.root}")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    p = profile
    observations = [i / p.fps for i in range(p.horizon + 1)]
    return LeRobotDataset(
        repo_id="local",
        root=p.root,
        revision="local",
        download_videos=False,
        delta_timestamps={
            p.image_key: observations,
            p.state_key: [0.0],
            p.action_key: observations[:-1],
        },
        tolerance_s=1e-4,
    )


class ConfigurableLeRobotDataset:
    """Map-style complete windows from a full LeRobot dataset."""

    def __init__(self, profile: EmbodimentProfile, episode_ids: list[int], reader=None):
        self.profile = profile
        self.reader = open_reader(profile) if reader is None else reader
        if getattr(self.reader, "episodes", None) is not None:
            raise ValueError("reader must contain the full dataset; select episodes through the adapter")
        profile.validate_metadata(self.reader.meta.info)
        episodes = [dict(ep) for ep in self.reader.meta.episodes]
        self.episode_ids = set(episode_ids)
        self.indices = valid_windows(episodes, episode_ids, profile.horizon)
        if not self.indices:
            raise ValueError(f"{profile.name}: split has no complete windows")
        # Read only scalar metadata columns, without decoding embedded images.
        # LeRobot needs a scalar timestamp inside __getitem__ for video queries.
        self.window_metadata = self.reader.hf_dataset.select_columns(["index", "timestamp", "episode_index"])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start = self.indices[index]
        sample = dict(self.reader[start])
        if int(as_numpy(sample["episode_index"]).item()) not in self.episode_ids:
            raise ValueError("episode metadata points outside the selected split")
        stop = start + self.profile.horizon + 1
        window = self.window_metadata[start:stop]
        if not np.array_equal(as_numpy(window["index"]).reshape(-1), np.arange(start, stop)):
            raise ValueError("metadata window must contain consecutive absolute row indices")
        for key in ("index", "timestamp", "episode_index"):
            if as_numpy(sample[key]).item() != as_numpy(window[key]).reshape(-1)[0]:
                raise ValueError(f"reader and metadata window disagree on {key}")
        # Validate the actual H+1 row values, never synthesized times or padded
        # episode IDs. Keep the existing timestamp/episode/padding checks.
        sample.update({key: window[key] for key in ("timestamp", "episode_index")})
        return sample_from_lerobot(self.profile, sample)


def audit_rows(profile: EmbodimentProfile, reader, train_ids: list[int]):
    """Validate raw tabular rows once; yield train actions without window duplication.

    hf_dataset bypasses video decoding. Actual decoded windows are checked by
    sample_from_lerobot when consumed (and a preview per split during prepare).
    """
    p = profile
    previous = {}
    for row in reader.hf_dataset:
        episode = int(as_numpy(row["episode_index"]).item())
        timestamp = float(as_numpy(row["timestamp"]).item())
        if not np.isfinite(timestamp):
            raise ValueError("nonfinite timestamp")
        if episode in previous and not np.isclose(timestamp - previous[episode], 1 / p.fps, atol=1e-4):
            raise ValueError("nonmonotonic or irregular timestamps; resample explicitly")
        previous[episode] = timestamp
        state = p.canonical(row[p.state_key], state=True)
        action = p.canonical(row[p.action_key], state=False)
        if state.ndim != 1 or action.ndim != 1:
            raise ValueError("raw LeRobot rows must contain vectors")
        if episode in train_ids:
            yield action


def get_configurable_lerobot_sft_dataset(
    *,
    bundle_path: str,
    bundle_id: str,
    expected_horizon: int = 32,
    tokenizer_config=None,
    cfg_dropout_rate: float = 0.1,
):
    import torch
    from torch.utils.data import IterableDataset

    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset

    bundle = load_bundle(bundle_path)
    if bundle["bundle_id"] != bundle_id:
        raise ValueError("training config and profile bundle differ")
    datasets = []
    for entry in bundle["entries"]:
        p = EmbodimentProfile(**entry["profile"])
        if p.horizon != expected_horizon:
            raise ValueError("dataset horizon and tokenizer duration disagree")
        raw = ConfigurableLeRobotDataset(p, entry["splits"]["train"])
        if digest(raw.reader.meta.info) != entry["metadata_sha256"]:
            raise ValueError("dataset metadata changed since preparation")
        transform = CosmosSampleTransform(entry["stats"], p.dim, tokenizer_config, cfg_dropout_rate)
        datasets.append(ActionSFTDataset(raw, transform, p.resolution))

    class BalancedStream(IterableDataset):
        # This example deliberately uses num_workers=0 in its recipe.
        shard_world_size = 1
        shard_rank = 0

        def __len__(self):
            return balanced_epoch_size([len(d) for d in datasets], self.shard_world_size)

        def __iter__(self):
            if torch.utils.data.get_worker_info() is not None:
                raise ValueError("this example requires dataloader num_workers=0")
            epoch = 0
            while True:
                for source, index in balanced_schedule(
                    [len(d) for d in datasets], bundle["seed"], epoch, self.shard_rank, self.shard_world_size
                ):
                    yield datasets[source][index]
                epoch += 1

    return BalancedStream()
