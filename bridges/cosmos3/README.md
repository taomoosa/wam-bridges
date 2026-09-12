# Cosmos3: new robot embodiments

Adapt Cosmos3-Nano to absolute joint-position targets using LeRobot v3 datasets.
The example supports separate or mixed SFT for 6/7-joint arms, dual arms, and
other mechanisms with **1–64 total channels**, including any number of scalar
grippers. Changing an embodiment within this contract requires only dataset and
profile changes. Model training and robot deployment still require validation on
the target hardware; no trained custom policy is included.

## Setup

Use `cosmos-framework` revision `2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce` and its
[training environment](https://github.com/NVIDIA/cosmos-framework/blob/2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce/docs/setup.md).
Its lock selects LeRobot `1a4316c6845330bc552fb982dbc44bdb4f66f2f1`.
No new packages or lock changes are required. Python 3.11+ is recommended.

Manually copy the nine files under [overlay/](overlay/) to matching paths in the
framework checkout. Preserve existing local edits. Add this line to the experiment
imports inside `make_config()` in `cosmos_framework/configs/base/config.py`:

```python
    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_lerobot_nano  # noqa: F401
```

If redistributing the modified framework, also include this repository's
`LICENSE` and `NOTICE` under distinct names alongside its existing licenses.

Run the following commands from the framework root. Stage a Nano base checkpoint
in DCP format and `Wan2.2_VAE.pth` using the
[upstream SFT instructions](https://github.com/NVIDIA/cosmos/blob/a9aa3bc3986910892d3af1575a486babfb731275/cookbooks/cosmos3/generator/action/finetune/README.md).
The `cosmos` and `wam-bridges` checkouts are not runtime dependencies.

## Dataset and profile

Use one LeRobot root per embodiment. Each row contains synchronized RGB,
measured joint/gripper state, issued absolute joint/gripper targets, timestamp,
episode/index metadata, and a task instruction. The defaults are
`observation.images.front`, `observation.state`, and `action`. Both image and video
storage are supported. Use fixed fps, explicit feature `names`, at least three
episodes, and complete H+1-frame windows (default H=32 at 15 Hz). Keep the full
root; the adapter selects train/validation/test episodes.

Edit [profiles.json](overlay/examples/action_policy_lerobot/profiles.json):

| Setting | Meaning |
| --- | --- |
| `name`, `root` | Embodiment identifier and dataset path relative to the profile JSON |
| `domain_id` | Unique ID per embodiment, starting at 32; use consecutive IDs |
| `joint_names` | Canonical order of **all** channels, including every gripper |
| `state_indices`, `action_indices` | Source-vector positions in canonical order |
| `gripper_indices` | Gripper positions in canonical order; `[]` for none; omission defaults to the final channel |
| `units` | Optional per-channel `rad`, `m`, or `open_fraction`; defaults to radians and designated gripper fractions |
| `gripper_convention` | `open_fraction` (default) or `closed_fraction`, applied to all input grippers |
| `fps`, `horizon` | Positive integer fps and positive H divisible by four; H must match within a mixed run |
| `image_height`, `image_width` | Dataset image dimensions, default 256 x 256 |

The supplied `dual_arm7` profile has 16 channels:
`[left_q1..q7, left_gripper, right_q1..q7, right_gripper]`, with
`gripper_indices=[7,15]`. Record both arms in the same row, with aligned timestamps
and a front image covering the task. A dual six-joint arm uses 14 channels and
`gripper_indices=[6,13]`. More joints or additional grippers use the same mapping;
no Python edit is needed within the 64-channel limit.

Convert logs to the declared units before training; `units` labels do not convert
degrees to radians or millimeters to meters. Grippers must already be calibrated
fractions with one common input convention. Output grippers always use 0=closed,
1=open. Channels outside `gripper_indices` are never flipped or constrained to
[0,1]. Adding channels changes the learned contract and requires a new SFT run.

## Train

```bash
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Nano-DCP
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
python examples/action_policy_lerobot/train.py --prepare \
  --profiles examples/action_policy_lerobot/profiles.json \
  --embodiments dual_arm7 --run-dir outputs/dual_arm --seed 42 \
  --base-revision YOUR_EXACT_MODEL_REVISION
python examples/action_policy_lerobot/train.py --dryrun --run-dir outputs/dual_arm --gpus 1
python examples/action_policy_lerobot/train.py --launch --run-dir outputs/dual_arm --gpus 1
```

Omit `--embodiments` to mix all configured datasets, or list the desired names.
Use a new run directory for each preparation. Statistics are computed from original
training rows per embodiment; the frozen bundle records profiles, splits, and
statistics. The wrapper sets the model's domain count to the highest selected ID
plus one. Keep this bundle with its generated TOML and checkpoints. Re-prepare old
bundles after upgrading to the multi-gripper version (bundle schema 2).

The TOML defaults to 10 iterations, one sample per batch, disabled EMA/compilation,
and no online W&B logging. Adjust it for actual training. Keep `num_workers=0`.
Mixed sampling gives each embodiment equal weight and repeats shorter sources;
distributed ranks share epoch lengths. Each source needs at least one training
window per GPU. Repeat the same launch command/output location to use framework
checkpoint resume.

## Infer

Supply an observation JSON with `embodiment_id`, `task`, `state`, `image_path`, and
`timestamp`. State uses the same source order and units as the dataset; image paths
resolve relative to the JSON. The supplied `dual_arm7` profile uses a 16-value
state; a dual six-joint arm with two grippers uses 14 values. Do not supply future
actions or images.

```bash
export POLICY_JOB=outputs/dual_arm/training/cosmos3_action/lerobot/demo
python examples/action_policy_lerobot/infer.py \
  --profiles outputs/dual_arm/profiles.json \
  --checkpoint "$POLICY_JOB/checkpoints/iter_000000010" \
  --config "$POLICY_JOB/config.yaml" \
  --input observation.json --output actions.json --seed 42
```

Choose an actual saved iteration and its own `config.yaml`. Output is H x D
absolute targets in `joint_names` order with per-channel `units`; normalization
is already reversed. Python callers can use
`LeRobotPolicy.from_checkpoint(checkpoint, config, profiles).infer(observation)`,
passing an HWC uint8 RGB array as `image` instead of `image_path`.

Offline replay permits timestamp 0. `--live` requires a recent Unix timestamp and
checks freshness before/after generation plus every gripper's output range.
The robot controller must handle timing, limits, collisions, and stale chunks.
The example does not execute a robot. Held-out evaluation and full SFT/checkpoint
round-trip validation remain required; metadata hashes do not replace immutable
dataset/model snapshots.

## Alternatives and sources

**Cartesian targets:** not implemented. Absolute `[xyz, rotation6d, gripper]`
requires frame/TCP calibration, a separate state/action normalization contract,
and Cartesian control or IK. Compare task success, FK pose error, IK continuity,
and constraints against joint targets on the same demonstrations.

**Edge:** the implementation uses Nano. Public cards list
[Nano at 16B](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) and
[Edge at 4B](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID). BF16 weights
alone are roughly 32/8 GB, not full memory requirements. Evaluate Edge first for
memory-limited deployment; it needs a separate Edge recipe/checkpoint. Measure
full memory and latency on the target hardware. At 15 Hz, K actions span K/15
seconds; replanning and control overhead must fit within that budget.

The example reuses the upstream DROID recipe, `ActionSFTDataset`,
`ActionTransformPipeline`, action postprocessing, trainer and `OmniInference`.
The profile, channel mapping and run preparation are bridge additions.
See the [license table](../../README.md#licenses) and [file-level attribution](../../NOTICE).
