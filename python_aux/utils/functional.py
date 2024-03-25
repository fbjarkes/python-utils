from functools import reduce
from typing import Callable


def pipe(*functions: Callable) -> Callable:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)
