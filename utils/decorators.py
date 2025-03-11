from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time

        from loguru import logger

        logger.info(f"Running task '{func.__name__}'...")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.success(f"Task '{func.__name__}' ended successfully in: {end - start}s")
        return result

    return wrapper
