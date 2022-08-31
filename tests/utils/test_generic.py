import pytest

from ertk.utils.generic import batch_iterable, filter_kwargs, itmap, ordered_intersect


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


def test_batch_iterable():
    iterable = [0] * 100
    batches = list(batch_iterable(iterable, batch_size=10, return_last=True))
    assert len(batches) == 10
    assert len(batches[0]) == 10
    assert len(batches[-1]) == 10

    batches = list(batch_iterable(iterable, batch_size=12, return_last=True))
    assert len(batches) == 9
    assert len(batches[0]) == 12
    assert len(batches[-1]) == 4

    batches = list(batch_iterable(iterable, batch_size=12, return_last=False))
    assert len(batches) == 8
    assert len(batches[0]) == 12
    assert len(batches[-1]) == 12
