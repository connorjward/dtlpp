import dtlc.backends


def compile(expr, *, backend=None, **kwargs):
    """Compile ``expr`` to an executable.

    Returns
    -------
    something that accepts TensorVariables as arguments and performs the computation
    """
    if backend == dtlc.backends.Backend.PYTHON:
        raise NotImplementedError
    else:
        raise ValueError
