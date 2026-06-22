"""Compatibility helpers for code moved behind stable module facades."""

from types import FunctionType
from typing import Any, Callable, Dict


def rebind_function_to_module(
    function: Callable[..., Any],
    module_globals: Dict[str, Any],
) -> Callable[..., Any]:
    """Copy a function so global lookups and metadata use its legacy module."""

    rebound = FunctionType(
        function.__code__,
        module_globals,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    rebound.__kwdefaults__ = function.__kwdefaults__
    rebound.__annotations__ = dict(function.__annotations__)
    rebound.__dict__.update(function.__dict__)
    rebound.__doc__ = function.__doc__
    rebound.__module__ = module_globals["__name__"]
    rebound.__qualname__ = function.__qualname__
    return rebound
