import functools
import itertools

import dtl
import numpy as np


BINOPS = {
    dtl.MulBinOp: np.multiply
}


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


class Tensor(dtl.TensorVariable):

    _name_generator = NameGenerator(prefix="T")

    def __init__(self, space, name=None, data=None):
        if name is None:
            name = next(self._name_generator)
        super().__init__(name, space)
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
def apply_lambda(expr: dtl.Lambda):
    return apply(expr.sub)


@apply.register
def apply_unindex(unindex: dtl.deIndex):
    indexed = apply(unindex.tensor)

    input_indices = _stringify(indexed.indices)
    output_indices = _stringify(unindex.indices)
    return indexed.tensor.copy(data=np.einsum(f"{input_indices}->{output_indices}", indexed.tensor.data))


@apply.register
def _(expr: dtl.BinOp):
    # I think that binops only make sense if the lhs and rhs have the same indices
    # That doesn't work because matmul breaks...
    # Should only apply to matching indices, but then what is the output shape?
    assert expr.lhs.indices == expr.rhs.indices

    op = BINOPS[type(expr)]
    lhs, rhs = apply(expr.lhs), apply(expr.rhs)
    tensor = Tensor(expr.lhs.tensor.space, data=op(lhs.tensor.data, rhs.tensor.data))
    return dtl.IndexedTensor(tensor, lhs.indices)


@apply.register
def apply_indexed_tensor(expr: dtl.IndexedTensor):
    # remember this is a scalar
    return type(expr)(apply(expr.tensor), expr.indices)


@apply.register
def apply_tensor_variable(expr: dtl.TensorVariable):
    # do nothing for actual tensor variables. These sit as the base of the tree
    return expr


def _stringify(indices):
    """Turn an iterable of indices into a string suitable for passing to einsum."""
    return "".join(str(i) for i in indices)
