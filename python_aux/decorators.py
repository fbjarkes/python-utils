from functools import wraps
from time import perf_counter
from typing import Callable


def try_except(func):
    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)  # TODO: user logger

    return handler


def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        results = func(*args, **kwargs)
        end = perf_counter()
        run_time = end - start
        print(f"{func.__name__} ran in {run_time:.4f} seconds")
        return results, run_time

    return wrapper