from labml.core.benchmark_progress import _append_fifo


def test_append_fifo_keeps_latest_ten_items() -> None:
    items: list[str] = []
    for idx in range(10):
        removed = _append_fifo(items, f"item-{idx}", 10)
        assert removed is None

    removed = _append_fifo(items, "item-10", 10)

    assert removed == "item-0"
    assert len(items) == 10
    assert items[0] == "item-1"
    assert items[-1] == "item-10"


def test_append_fifo_preserves_insertion_order_after_evictions() -> None:
    items = ["a", "b", "c"]
    removed = _append_fifo(items, "d", 3)

    assert removed == "a"
    assert items == ["b", "c", "d"]
