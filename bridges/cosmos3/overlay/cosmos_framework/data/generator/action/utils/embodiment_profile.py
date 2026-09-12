# SPDX-FileCopyrightText: Copyright (c) 2026 WAM Bridges contributors
# SPDX-License-Identifier: MIT
"""Small, explicit joint-position contract shared by data preparation and serving."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@dataclass(frozen=True)
class EmbodimentProfile:
    name: str
    domain_id: int
    joint_names: list[str]  # Canonical order of all joints and grippers.
    root: str
    state_indices: list[int]
    action_indices: list[int]
    gripper_indices: list[int] | None = None  # Canonical indices; None preserves the single-gripper default.
    units: list[str] | None = None  # Canonical output units: rad, m, or open_fraction.
    state_key: str = "observation.state"
    action_key: str = "action"
    image_key: str = "observation.images.front"
    action_space: str = "absolute_joint_position"
    gripper_convention: str = "open_fraction"
    fps: int = 15
    horizon: int = 32
    image_height: int = 256
    image_width: int = 256
    resolution: str = "256"
    max_age_seconds: float = 0.5

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError("name must be a lowercase identifier")
        if type(self.domain_id) is not int or self.domain_id < 32:
            raise ValueError("custom domain IDs start at 32; use a distinct ID per embodiment")
        if self.action_space != "absolute_joint_position":
            raise ValueError("only absolute_joint_position is implemented; Cartesian control needs a separate contract")
        if not 1 <= self.dim <= 64 or len(set(self.joint_names)) != self.dim:
            raise ValueError("joint_names must be unique and fit max_action_dim=64")
        if any(not isinstance(n, str) or not n for n in self.joint_names):
            raise ValueError("joint names must be nonempty strings")
        for order in (self.state_indices, self.action_indices):
            if len(order) != self.dim or len(set(order)) != self.dim:
                raise ValueError("state/action indices must map every canonical channel exactly once")
            if any(type(i) is not int or i < 0 for i in order):
                raise ValueError("indices must be nonnegative integers")
        if self.gripper_convention not in ("open_fraction", "closed_fraction"):
            raise ValueError("gripper convention must be explicit")
        grippers = [self.dim - 1] if self.gripper_indices is None else self.gripper_indices
        if any(type(i) is not int or not 0 <= i < self.dim for i in grippers) or len(set(grippers)) != len(grippers):
            raise ValueError("gripper_indices must be distinct canonical channel indices")
        units = (
            self.units
            if self.units is not None
            else ["open_fraction" if i in grippers else "rad" for i in range(self.dim)]
        )
        if len(units) != self.dim or any(
            unit not in (("open_fraction",) if i in grippers else ("rad", "m")) for i, unit in enumerate(units)
        ):
            raise ValueError("units must use open_fraction for grippers and rad or m for other joints")
        object.__setattr__(self, "gripper_indices", list(grippers))
        object.__setattr__(self, "units", list(units))
        if type(self.fps) is not int or self.fps <= 0:
            raise ValueError("fps must be a positive integer")
        if type(self.horizon) is not int or self.horizon <= 0 or self.horizon % 4:
            raise ValueError("horizon must be a positive multiple of four")
        if any(type(n) is not int or n < 16 for n in (self.image_height, self.image_width)):
            raise ValueError("image dimensions must be integers >=16")
        if self.resolution not in ("256", "480"):
            raise ValueError("sample supports resolution tiers 256 and 480")
        if not np.isfinite(self.max_age_seconds) or self.max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive and finite")

    @property
    def dim(self) -> int:
        return len(self.joint_names)

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical(self, value, *, state: bool) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        order = self.state_indices if state else self.action_indices
        if value.ndim not in (1, 2) or value.shape[-1] <= max(order) or not np.isfinite(value).all():
            raise ValueError("invalid/nonfinite source state or action")
        result = value[..., order].copy()
        if self.gripper_convention == "closed_fraction":
            result[..., self.gripper_indices] = 1 - result[..., self.gripper_indices]
        grippers = result[..., self.gripper_indices]
        if np.any((grippers < 0) | (grippers > 1)):
            raise ValueError("gripper must be a calibrated fraction in [0,1]")
        return result

    def validate_metadata(self, info: dict) -> None:
        if info.get("codebase_version") != "v3.0" or info.get("fps") != self.fps:
            raise ValueError("expected LeRobot v3.0 and matching fps; resample explicitly before training")
        for key, order in ((self.state_key, self.state_indices), (self.action_key, self.action_indices)):
            feature = info["features"][key]
            names = feature.get("names")
            if not isinstance(names, list) or len(feature["shape"]) != 1:
                raise ValueError(f"{key}: vector features with explicit names are required")
            if feature["shape"][0] != len(names) or max(order) >= len(names):
                raise ValueError(f"{key}: feature shape/index mismatch")
            if [names[i] for i in order] != self.joint_names:
                raise ValueError(f"{key}: joint order differs from the profile")
        image = info["features"][self.image_key]
        if image["dtype"] not in ("image", "video"):
            raise ValueError("camera feature must be an image or video")
        shape = list(image["shape"])
        if shape not in ([self.image_height, self.image_width, 3], [3, self.image_height, self.image_width]):
            raise ValueError("camera dimensions differ from profile")


def load_profiles(path: str | Path) -> list[EmbodimentProfile]:
    path = Path(path).resolve()
    doc = read_json(path)
    if doc.get("schema_version") != 1:
        raise ValueError("unsupported profile schema")
    profiles = []
    for data in doc["profiles"]:
        data = dict(data)
        data["root"] = str((path.parent / data["root"]).resolve())
        profiles.append(EmbodimentProfile(**data))
    if not profiles or len({p.name for p in profiles}) != len(profiles):
        raise ValueError("empty or duplicate profiles")
    if len({p.domain_id for p in profiles}) != len(profiles):
        raise ValueError("different mechanisms must not share a domain")
    return profiles


def validate_stats(stats: dict, dim: int) -> tuple[np.ndarray, np.ndarray]:
    mean, std = (np.asarray(stats[k], dtype=np.float32) for k in ("mean", "std"))
    if mean.shape != (dim,) or std.shape != (dim,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("normalization stats must match the real action width and be finite")
    if np.any(std < 0):
        raise ValueError("negative standard deviation")
    return mean, np.maximum(std, 1e-8)


def load_bundle(path: str | Path) -> dict:
    bundle = read_json(path)
    body = {k: v for k, v in bundle.items() if k != "bundle_id"}
    if bundle.get("bundle_id") != digest(body):
        raise ValueError("profile/stats/split bundle hash mismatch")
    if bundle.get("schema_version") != 2 or bundle.get("model_family") != "nano":
        raise ValueError("unsupported policy bundle; prepare a new Nano run")
    domains = bundle.get("num_embodiment_domains")
    if bundle.get("max_action_dim") != 64 or type(domains) is not int or domains < 33:
        raise ValueError("bundle must specify 64 action channels and custom domains starting at 32")
    if bundle.get("backend") != "cosmos":
        raise ValueError("a Cosmos training bundle is required")
    profiles = [EmbodimentProfile(**entry["profile"]) for entry in bundle["entries"]]
    if (
        not profiles
        or len({p.domain_id for p in profiles}) != len(profiles)
        or len({p.name for p in profiles}) != len(profiles)
    ):
        raise ValueError("empty bundle or duplicate domains")
    for p, entry in zip(profiles, bundle["entries"]):
        if p.domain_id >= domains:
            raise ValueError("profile domain exceeds the model domain count")
        validate_stats(entry["stats"], p.dim)
    return bundle
