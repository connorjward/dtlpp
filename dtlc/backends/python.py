import functools
import itertools
from typing import Dict

import dtl
import dtlutils
import numpy as np


def lower(expr):
    return functools.partial(evaluate, expr)


BINOPS = {dtl.AddBinOp: np.add, dtl.MulBinOp: np.multiply}


class IndexIterator:
    def __init__(self, index, space: dtl.VectorSpace):
        self._index = index
        self._counter = iter(range(space.dim))

    def __iter__(self):
        return self

    def __next__(self):
        """Return a 2-tuple of the index and its current value."""
        return self._index, next(self._counter)


class MultiindexIterator:
    def __init__(self, index_spaces):
        iters = [IndexIterator(index, space) for index, space in index_spaces.items()]
        self._iterator = itertools.product(*iters)

    def __iter__(self):
        return self

    def __next__(self):
        """Return a dictionary mapping each index to its current value."""
        return dict(next(self._iterator))


@functools.singledispatch
def evaluate(expr, **kwargs):
    """Apply a postorder traversal to the DAG, computing a result as you go."""
    raise NotImplementedError


@evaluate.register
def _(expr: dtl.Lambda, **kwargs):
    return evaluate(expr.sub, **kwargs)


@evaluate.register
def _(unindex: dtl.deIndex, **kwargs):
    scalar_expr = evaluate(unindex.scalar_expr, **kwargs)

    result = np.zeros(unindex.space.shape, dtype=int)

    for index_map in MultiindexIterator(scalar_expr.index_spaces):
        value = compute_scalar_expr(scalar_expr, index_map, kwargs["var_map"])
        result[tuple(index_map[i] for i in unindex.indices)] += value
    return result


@evaluate.register
def _(expr: dtl.BinOp, **kwargs):
    return type(expr)(evaluate(expr.lhs, **kwargs), evaluate(expr.rhs, **kwargs))


@evaluate.register
def _(expr: dtl.IndexedTensor, **kwargs):
    # remember this is a scalar
    return type(expr)(evaluate(expr.tensor_expr, **kwargs), expr.indices)


@evaluate.register
def _(expr: dtl.TensorVariable, **kwargs):
    # do nothing for actual tensor variables. These sit as the base of the tree
    return expr


def compute_scalar_expr(scalar_expr, index_map, var_map):
    """Return a scalar from a scalar expression and a given map between
    abstract indices and their values.

    This function is recursive in order to handle a sequence of binary operations.
    """
    if isinstance(scalar_expr, dtl.IndexedTensor):
        # base case
        return var_map[scalar_expr.tensor_expr][
            tuple(index_map[i] for i in scalar_expr.indices)
        ]
    else:
        assert isinstance(scalar_expr, dtl.BinOp)
        lhs = compute_scalar_expr(scalar_expr.lhs, index_map, var_map)
        rhs = compute_scalar_expr(scalar_expr.rhs, index_map, var_map)
        return BINOPS[type(scalar_expr)](lhs, rhs)


def _stringify(indices):
    """Turn an iterable of indices into a string suitable for passing to einsum."""
    return "".join(str(i) for i in indices)
