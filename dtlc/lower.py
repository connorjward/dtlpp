import dtlc.backends


def lower(expr, *, backend, **kwargs):
    """Compile ``expr`` to an executable.

    Returns
    -------
    something that accepts TensorVariables as arguments and performs the computation
    """
    if backend == dtlc.backends.Backend.PYTHON:
        return dtlc.backends.python.lower(expr, **kwargs)
    else:
        raise ValueError
