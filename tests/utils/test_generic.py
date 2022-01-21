import pytest

from ertk.utils.generic import filter_kwargs, itmap, ordered_intersect


@pytest.mark.parametrize(
    ["kwargs", "expected"],
    [
        ({"a": 1, "b": 2, "c": 3, "d": 4}, {"a": 1, "b": 2, "c": 3}),
        ({"a": 1, "b": 2, "d": 4}, {"a": 1, "b": 2}),
        ({"d": 4, "e": 5}, {}),
    ],
)
def test_filter_kwargs(kwargs, expected):
    def f(a, b, c):
        pass

    filtered = filter_kwargs(kwargs, f)
    assert filtered == expected


@pytest.mark.parametrize(
    "kwargs",
    [{"a": 1, "b": 2, "c": 3, "d": 4}, {"a": 1, "b": 2, "d": 4}, {"d": 4, "e": 5}],
)
def test_filter_kwargs_posonly(kwargs):
    filtered = filter_kwargs(kwargs, sum)
    assert filtered == {}


@pytest.mark.parametrize(
    "kwargs",
    [{"a": 1, "b": 2, "c": 3, "d": 4}, {"a": 1, "b": 2, "d": 4}, {"d": 4, "e": 5}],
)
def test_filter_kwargs_kwargs(kwargs):
    def f(*args, **kwargs):
        pass

    filtered = filter_kwargs(kwargs, f)
    assert filtered == {}


def test_itmap():
    f = itmap(lambda x: x + 1)
    assert f(1) == 2
    assert f([1, 2]) == [2, 3]
    assert type(f([1, 2])) is list
    assert f((1, 2)) == (2, 3)
    assert type(f((1, 2))) is tuple


@pytest.mark.parametrize(
    ["a", "b", "expected"],
    [
        ([5, 2, 5, 6], {5, 2}, [5, 2, 5]),
        (range(10), {5, 2}, [2, 5]),
    ],
)
def test_ordered_intersect(a, b, expected):
    assert ordered_intersect(a, b) == expected
