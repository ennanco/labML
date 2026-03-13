"""Partition generation and split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
import pandas as pd
import typer
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)


@dataclass
class PartitionConfig:
    """Controls CV split generation."""

    mode: str = "random"
    n_splits: int = 10
    shuffle: bool = True
    random_state: int | None = 42
    group_column: str | None = None


class SplitStrategy(Protocol):
    """Strategy interface to generate train/test split indexes."""

    def split(
        self, data: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Return train/test indexes for each fold."""


class RandomSplitStrategy:
    """Random split strategy with task-aware splitter selection."""

    def __init__(self, task: str, config: PartitionConfig) -> None:
        self.task = task
        self.config = config

    def split(
        self, data: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        del groups
        if self.task == "classification":
            splitter = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
            return splitter.split(data, y)

        splitter = KFold(
            n_splits=self.config.n_splits,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state,
        )
        return splitter.split(data, y)


class GroupSplitStrategy:
    """Group-aware split strategy that avoids leakage by individual."""

    def __init__(self, task: str, config: PartitionConfig) -> None:
        self.task = task
        self.config = config

    def split(
        self, data: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise typer.BadParameter("Group split requires groups")

        if self.task == "classification":
            splitter = StratifiedGroupKFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
            try:
                return splitter.split(data, y, groups=groups)
            except ValueError:
                fallback = GroupKFold(n_splits=self.config.n_splits)
                return fallback.split(data, y, groups=groups)

        splitter = GroupKFold(n_splits=self.config.n_splits)
        return splitter.split(data, y, groups=groups)


def make_split_strategy(task: str, partition: PartitionConfig) -> SplitStrategy:
    """Factory method returning a split strategy implementation."""
    if partition.mode == "group":
        return GroupSplitStrategy(task=task, config=partition)
    return RandomSplitStrategy(task=task, config=partition)


def build_fold_ids(
    data: pd.DataFrame,
    y: pd.Series,
    task: str,
    partition: PartitionConfig,
) -> pd.Series:
    """Generate one fold_id per row according to the selected strategy."""
    fold_ids = pd.Series(np.full(len(data), -1, dtype=int))

    groups: pd.Series | None = None
    if partition.mode == "group":
        if not partition.group_column or partition.group_column not in data.columns:
            raise typer.BadParameter(
                "Partition mode 'group' requires [partition].group_column present in data"
            )
        groups = data[partition.group_column]

    strategy = make_split_strategy(task=task, partition=partition)
    splits = strategy.split(data, y, groups)

    for fold_id, (_, test_idx) in enumerate(splits):
        fold_ids.iloc[test_idx] = fold_id

    return fold_ids


def fold_ids_to_splits(fold_ids: pd.Series) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Convert fold ids into sklearn-compatible train/test index tuples."""
    fold_array = fold_ids.to_numpy()
    unique_folds = sorted(pd.unique(fold_array))
    for fold in unique_folds:
        test_idx = np.where(fold_array == fold)[0]
        train_idx = np.where(fold_array != fold)[0]
        yield train_idx, test_idx
