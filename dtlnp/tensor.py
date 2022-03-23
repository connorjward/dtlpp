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


# TODO don't actually pass this in to expr, a mapping at the end instead
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


class TensorSum(dtl.TensorExpression):

    def __init__(self, lhs, rhs, lhs_indices, rhs_indices, out_indices):
        self.lhs = lhs
        self.rhs = rhs
        self.lhs_indices = lhs_indices
        self.rhs_indices = rhs_indices
        self.out_indices = out_indices


@functools.singledispatch
def replace_deindexes(expr):
    """Replace deindex -> add with TensorSum nodes. This lets us do our
    evaluations at the right level with singledispatch."""
    raise AssertionError


@replace_deindexes.register
def _(expr: dtl.deIndex):
    if not isinstance(expr.scalar_expr, dtl.BinOp):
        return expr  # need to postorder here too

    lhs = expr.scalar_expr.lhs
    rhs = expr.scalar_expr.rhs

    assert isinstance(lhs, dtl.IndexedTensor)
    assert isinstance(rhs, dtl.IndexedTensor)
    
    # postorder traversal
    lhs_tensor_expr = replace_deindexes(lhs.tensor_expr)
    rhs_tensor_expr = replace_deindexes(rhs.tensor_expr)

    if isinstance(expr.scalar_expr, dtl.AddBinOp):
        return TensorSum(lhs_tensor_expr, rhs_tensor_expr, lhs.indices, rhs.indices, expr.indices)
    else:
        raise NotImplementedError


def evaluate(expr):
    expr = replace_deindexes(expr)
    return apply(expr)

@functools.singledispatch
def apply(expr):
    """Apply a postorder traversal to the DAG, computing a result as you go."""
    raise NotImplementedError


@apply.register
def _(expr: dtl.Lambda):
    return apply(expr.sub)


@apply.register
def _(unindex: dtl.deIndex):
    # scalar_expr = apply(unindex.scalar_expr)
    #
    # input_indices = _stringify(scalar_expr.indices)
    # output_indices = _stringify(unindex.indices)
    # return scalar_expr.tensor.copy(data=np.einsum(f"{input_indices}->{output_indices}", scalar_expr.tensor.data))

    if isinstance(unindex.scalar_expr, dtl.BinOp):
        lhs, rhs = 

    lhs, rhs = apply(unindex.scalar_expr)

    for i, j in unindex.indices:
        output[i, j] = apply(unindex.scalar_expr, [i, j])
        output[i, j] = lhs[i] * rhs[j]


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
