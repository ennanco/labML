from labml.core.ranges import expand_param_grid, expand_value


def test_expand_value_range_inclusive() -> None:
    values = expand_value("1:0.5:2")
    assert values == [1.0, 1.5, 2.0]


def test_expand_param_grid_cartesian() -> None:
    grid = expand_param_grid({"a": [1, 2], "b": ["0:1:2"]})
    assert len(grid) == 6
    assert {row["a"] for row in grid} == {1, 2}
    assert {row["b"] for row in grid} == {0, 1, 2}


def test_expand_list_keeps_nested_lists() -> None:
    values = expand_value([[100], [100, 50]])
    assert values == [[100], [100, 50]]
