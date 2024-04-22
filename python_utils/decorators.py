from functools import wraps
from time import perf_counter
from typing import Callable
import logging
import traceback

logger = logging.getLogger(__name__)

def try_except(func):
    @wraps(func)
    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"{func.__name__}: {e}")
            #if stack_trace:
            traceback.print_exc()
    return handler


def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        results = func(*args, **kwargs)
        end = perf_counter()
        run_time = end - start
        print(f"{func.__name__} ran in {run_time:.4f} seconds")
        return results

    return wrapper