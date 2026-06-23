__all__ = ["Worker"]


def __getattr__(name):
    if name == "Worker":
        from .worker import Worker

        return Worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
