from itertools import chain, combinations, permutations, zip_longest
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    overload,
)

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")


def itmap(s: Callable[[T1], T2]):
    """Returns a new map function that additionally maps tuples to
    tuples and lists to lists.

    Parameters
    ----------
    s: callable
        A callable that converts objects of type T to type S.

    Returns
    -------
    callable
        A function that converts objects of type T to type S, lists of T
        to lists of S, and tuples of T to tuples of S.
    """

    @overload
    def _map(x: T1) -> T2:
        ...

    @overload
    def _map(x: List[T1]) -> List[T2]:
        ...

    @overload
    def _map(x: Tuple[T1, ...]) -> Tuple[T2, ...]:
        ...

    def _map(x):
        if isinstance(x, (list, tuple)):
            return type(x)(s(y) for y in x)
        else:
            return s(x)

    return _map


def ordered_intersect(a: Iterable, b: Container) -> List:
    """Returns a list of the intersection of `a` and `b`, in the order
    elements appear in `a`.

    Parameters
    ----------
    a: iterable
        Iterable with the order of elements.
    b: container
        Container to test inclusion of elements in `a`.

    Returns
    -------
    list
        The resulting elements, in order.
    """
    return [x for x in a if x in b]


def filter_kwargs(kwargs: Dict[str, Any], method: Callable) -> Dict[str, Any]:
    """Removes incompatible keyword arguments. This ignores any
    `**kwargs` catchall in method signature, and only returns args
    specifically present as keyhwords in the method signature which are
    also not positional only.

    Parameters
    ----------
    params: dict
        Keyword arguments to pass to method.
    method: callable
        The method for which to check valid parameters.

    Returns
    -------
    dict
        Filtered keyword arguments.
    """
    import inspect

    meth_params = inspect.signature(method).parameters
    kwargs = kwargs.copy()
    for key in set(kwargs.keys()):
        if (
            key not in meth_params
            or meth_params[key].kind == inspect.Parameter.POSITIONAL_ONLY
        ):
            del kwargs[key]
    return kwargs


_BatchSentinel = object()


def batch_iterable(
    it: Iterable[T],
    batch_size: int,
    return_last: bool = True,
) -> Iterable[Tuple[T, ...]]:
    """Batches an iterable into chunks of size `batch_size`.

    Parameters
    ----------
    it: iterable
        The iterable to batch.
    batch_size: int
        The size of each batch/chunk.
    return_last: bool
        Whether to yield the last batch if it isn't full (i.e.
        `batch_size` doesn't divide the iterable length.)

    Yields
    -------
    tuple
        A tuple of length `batch_size` with successive elements from the
        original iterable. If `return_last == False` then the last batch
        is dropped if it contains less than `batch_size` items.
    """
    # From the itertools recipes, to chunk an iterator
    iterables = [iter(it)] * batch_size
    if return_last:
        for batch in zip_longest(*iterables, fillvalue=_BatchSentinel):
            yield tuple(filter(lambda x: x is not _BatchSentinel, batch))
    else:
        yield from zip(*iterables)


def subsets(it: Iterable[T], max_size: Optional[int] = None) -> Iterable[Tuple[T, ...]]:
    """Iterate over all subsets of the iterable `it`, up to a given
    maximum size. This will generate subsets in size order and then
    index-sorted order (i.e. the order items appear in `it`).

    Parameters
    ----------
    it: iterable
        The iterable from which to generate subsets.
    max_size: int, optional
        The maximum size of generated subsets. If not given, the size of
        `it` is determined by creating a list of `it`'s elements.

    Yields
    ------
    tuple
        The next generated subset.
    """
    if max_size is None:
        it = list(it)
        max_size = len(it)
    yield from chain(*(combinations(it, i) for i in range(max_size + 1)))


def ordered_subsets(
    it: Iterable[T], max_size: Optional[int] = None
) -> Iterable[Tuple[T, ...]]:
    """Iterate over all ordered subsets of the iterable `it`, up to a
    given maximum size. This will generate subsets in size order and
    then index-sorted order (i.e. the order items appear in `it`).

    Parameters
    ----------
    it: iterable
        The iterable from which to generate ordered subsets.
    max_size: int, optional
        The maximum size of generated subsets. If not given, the size of
        `it` is determined by creating a list of `it`'s elements.

    Yields
    ------
    tuple
        The next generated ordered subset.
    """
    if max_size is None:
        it = list(it)
        max_size = len(it)
    yield from chain(*(permutations(it, i) for i in range(max_size + 1)))
