def pipe(*functions: Callable) -> Callable:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)
