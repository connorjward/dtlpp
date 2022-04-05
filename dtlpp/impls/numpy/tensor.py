import functools
import itertools

import dtl
import numpy as np


BINOPS = {dtl.AddBinOp: np.add, dtl.MulBinOp: np.multiply}


class NameGenerator:
    def __init__(self, prefix="", suffix=""):
        if not (prefix or suffix):
            raise ValueError

        self._prefix = prefix
        self._suffix = suffix
        self._counter = itertools.count()

    def __iter__(self):
        return self

    def __next__(self):
        return f"{self._prefix}{next(self._counter)}{self._suffix}"


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


# TODO don't actually pass this in to expr, instead pass a mapping at compute-time
class Tensor(dtl.TensorVariable):

    _name_generator = NameGenerator(prefix="T")

    def __init__(self, space, name=None, data=None):
        if name is None:
            name = next(self._name_generator)
        super().__init__(space, name)
        self.data = data

    def copy(self, *, name=None, space=None, data=None):
        name = name if name is not None else self.name
        space = space if space is not None else self.space
        data = data if data is not None else self.data
        return type(self)(name, space, data)


@functools.singledispatch
def apply(expr):
    """Apply a postorder traversal to the DAG, computing a result as you go."""
    raise NotImplementedError


@apply.register
def _(expr: dtl.Lambda):
    return apply(expr.sub)


@apply.register
def _(unindex: dtl.deIndex):
    scalar_expr = apply(unindex.scalar_expr)

    result = Tensor(unindex.space, data=np.zeros(unindex.space.shape, dtype=int))
    indexed_result = result[unindex.indices]

    for index_map in MultiindexIterator(scalar_expr.index_spaces):
        # We have the index map to look up the values of i, j, k etc so we can index
        # into the numpy arrays.
        value = compute_scalar_expr(scalar_expr, index_map)
        # warning: modifies in place!
        inc_scalar(indexed_result, value, index_map)
    return result


@apply.register
def _(expr: dtl.BinOp):
    return type(expr)(apply(expr.lhs), apply(expr.rhs))


@apply.register
def apply_indexed_tensor(expr: dtl.IndexedTensor):
    # remember this is a scalar
    return type(expr)(apply(expr.tensor_expr), expr.indices)


@apply.register
def apply_tensor_variable(expr: dtl.TensorVariable):
    # do nothing for actual tensor variables. These sit as the base of the tree
    return expr


def compute_scalar_expr(scalar_expr, index_map):
    """Return a scalar from a scalar expression and a given map between
    abstract indices and their values.

    This function is recursive in order to handle a sequence of binary operations.
    """
    if isinstance(scalar_expr, dtl.IndexedTensor):
        # base case
        return get_scalar(scalar_expr, index_map)
    else:
        assert isinstance(scalar_expr, dtl.BinOp)
        lhs = compute_scalar_expr(scalar_expr.lhs, index_map)
        rhs = compute_scalar_expr(scalar_expr.rhs, index_map)
        return BINOPS[type(scalar_expr)](lhs, rhs)


def get_scalar(expr: dtl.ScalarExpr, index_map):
    """Retrieve a scalar from an expression given an index map."""
    return expr.tensor_expr.data[tuple(index_map[i] for i in expr.indices)]


def inc_scalar(expr, value, index_map):
    """Increment an entry in a scalar expression."""
    expr.tensor_expr.data[tuple(index_map[i] for i in expr.indices)] += value


def iterate_index(index):
    return ((index, d) for d in range(index.space.dim))


def _stringify(indices):
    """Turn an iterable of indices into a string suitable for passing to einsum."""
    return "".join(str(i) for i in indices)
