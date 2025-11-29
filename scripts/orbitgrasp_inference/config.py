from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_checkpoint(root_dir: Path, load_epoch: int, ckpt_prefix: str = "orbitgrasp-ckpt-") -> str:
    files = os.listdir(root_dir)
    matching_files = [
        f for f in files
        if f"{ckpt_prefix}{load_epoch}-" in f and f.endswith(".pt")
    ]
    if not matching_files:
        raise ValueError(
            f"No checkpoints found with prefix '{ckpt_prefix}{load_epoch}-' in '{root_dir}'"
        )
    matching_files.sort(reverse=True)
    return matching_files[0]


class OrbitGraspConfig:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config: Dict[str, Any] = load_config(config_path)
        self.base_path = config_path.parent

    @property
    def device(self) -> str:
        return self.config["orbit_grasper"]["device"]

    @property
    def num_intervals(self) -> int:
        return self.config["orbit_grasper"]["num_channel"]

    @property
    def harmonics_lmax(self) -> int:
        if "train_dataset" in self.config:
            return self.config["train_dataset"]["lmax"]
        return self.config.get("lmax", 3)

    @property
    def orbit_lmax(self) -> int:
        return self.config["orbit_grasper"].get("lmax", 3)

    @property
    def orbit_mmax(self) -> int:
        return self.config["orbit_grasper"].get("mmax", 2)

    @property
    def load_epoch(self) -> int:
        return self.config["test"]["load_epoch"]

    @property
    def ckpt_root(self) -> Path:
        return self.base_path / self.config["test"]["root_dir"]

    @property
    def ckpt_name(self) -> str:
        return find_checkpoint(
            self.ckpt_root,
            self.load_epoch,
            self.config["test"].get("ckpt_prefix", "orbitgrasp-ckpt-"),
        )

    @property
    def seed(self) -> int:
        return int(self.config.get("seed", 12345))
