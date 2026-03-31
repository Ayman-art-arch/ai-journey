import time 

def average_time(func, *args, repeat=10000, **kwargs):
    start = time.perf_counter()
    for _ in range(repeat):
        func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) / repeat