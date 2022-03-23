import functools
import numpy as np

import dtl


class Tensor(dtl.TensorVariable):

    def __init__(self, name, space, data):
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
def apply_unindex(expr: dtl.deIndex):
    indexed = apply(expr.tensor)
    # horrible and very fragile
    input_indices = ",".join("".join(str(j) for j in i) for i in indexed.indices)
    output_indices = "".join(str(i) for i in expr.indices)
    return indexed.tensor.copy(data=np.einsum(f"{input_indices}->{output_indices}", *indexed.tensor.data))


@apply.register
def apply_binop(expr: dtl.BinOp):
    if type(expr) is not dtl.MulBinOp:
        raise NotImplementedError

    lhs, rhs = apply(expr.lhs), apply(expr.rhs)
    tensor = Tensor(f"{lhs.tensor.name}{rhs.tensor.name}", (lhs.tensor.space, rhs.tensor.space), (lhs.tensor.data, rhs.tensor.data))
    return dtl.IndexedTensor(tensor, (lhs.indices, rhs.indices))


@apply.register
def apply_indexed_tensor(expr: dtl.IndexedTensor):
    # remember this is a scalar
    return type(expr)(apply(expr.tensor), expr.indices)


@apply.register
def apply_tensor_variable(expr: dtl.TensorVariable):
    # do nothing for actual tensor variables. These sit as the base of the tree
    return expr
