from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    Tuple,
    TypeVar,
    overload,
)

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def itmap(s: Callable[[T1], T2]):
    """Returns a new map function that additionally maps tuples to
    tuples and lists to lists.
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
    """
    return [x for x in a if x in b]


def filter_kwargs(kwargs: Dict[str, Any], method: Callable) -> Dict[str, Any]:
    """Removes incompatible keyword arguments. This ignores any **kwargs
    catchall in method signature, and only returns args specifically
    present as keyhwords in the method signature which are also not
    positional only.

    Args:
    -----
    params: dict
        Keyword arguments to pass to method.
    method: callable
        The method for which to check valid parameters.

    Returns:
    --------
    params: dict
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
