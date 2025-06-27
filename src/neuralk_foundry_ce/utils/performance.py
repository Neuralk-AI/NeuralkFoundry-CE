import time
from memory_profiler import memory_usage


def profile_function(func, *args, **kwargs):
    """
    Profile a function's execution time and memory usage.
     
    Returns
    -------
    result : any
        The return value of the function.
    memory_usage : list
        List of memory usage measurements (in MB) during function execution.
    execution_time : float
        Total execution time in seconds.
        
    Examples
    --------
    >>> def expensive_function(n):
    ...     return sum(range(n))
    ... 
    >>> result, mem_usage, exec_time = profile_function(expensive_function, 1000000)
    >>> print(f"Result: {result}")
    >>> print(f"Max memory: {max(mem_usage):.2f} MB")
    >>> print(f"Execution time: {exec_time:.4f}s")
    """
    start_time = time.perf_counter()
    mem_usage, result = memory_usage((func, args, kwargs), retval=True)
    end_time = time.perf_counter()
    return result, mem_usage, end_time - start_time

