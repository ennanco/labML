import pandas as pd

from labml.core.partitioning import PartitionConfig, build_fold_ids


def test_group_partition_keeps_individual_together() -> None:
    data = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
            "individual_id": ["a", "a", "b", "b", "c", "c"],
        }
    )
    cfg = PartitionConfig(mode="group", n_splits=3, group_column="individual_id")
    fold_ids = build_fold_ids(
        data, y=data["target"], task="classification", partition=cfg
    )

    grouped = (
        data.assign(fold_id=fold_ids).groupby("individual_id")["fold_id"].nunique()
    )
    assert grouped.eq(1).all()
