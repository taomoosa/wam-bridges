# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small custom-joint policy recipe; launch via examples/action_policy_lerobot/train.py."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.data.generator.action.datasets.configurable_lerobot_dataset import (
    get_configurable_lerobot_sft_dataset,
)
from cosmos_framework.utils.lazy_config import LazyCall as L

action_policy_lerobot_nano = copy.deepcopy(action_policy_droid_nano)
cfg = action_policy_lerobot_nano
cfg.job.name = "action_policy_lerobot_nano"
cfg.job.wandb_mode = "disabled"
# train.py overrides this with max(selected domain_id) + 1.
cfg.model.config.num_embodiment_domains = 34
cfg.model.config.compile.enabled = False
cfg.model.config.ema.enabled = False
cfg.model.config.parallelism.data_parallel_shard_degree = -1
cfg.model.config.parallelism.data_parallel_replicate_degree = 1
cfg.dataloader_train.dataset_name = "action_lerobot"
cfg.dataloader_train.max_samples_per_batch = 1
loader = cfg.dataloader_train.dataloader
loader.batch_size = 1
loader.num_workers = 0
loader.persistent_workers = False
loader.prefetch_factor = None
loader.datasets = dict(
    robots=dict(
        ratio=1,
        dataset=L(get_configurable_lerobot_sft_dataset)(
            bundle_path="???",
            bundle_id="???",
            expected_horizon=32,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            cfg_dropout_rate=0.1,
        ),
    ),
)
# The inherited warm-start skip list resets action heads; same-job resume does not.
ConfigStore.instance().store(group="experiment", package="_global_", name="action_policy_lerobot_nano", node=cfg)
