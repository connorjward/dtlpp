import dtl
import numpy as np
import pytest

from dtlnp.tensor import Tensor, apply


@pytest.fixture
def space():
    return dtl.RealVectorSpace(10)


def test_matmul():
    A = Tensor("A", [dtl.RealVectorSpace(10)]*2, np.arange(100, dtype=int).reshape((10, 10)))
    x = Tensor("x", [dtl.RealVectorSpace(10)], np.arange(10, dtype=int))
    i = dtl.Index("i")
    j = dtl.Index("j")
    expr = dtl.Lambda([A, x], (A[i, j]*x[j]).forall(i))

    res = apply(expr)
    assert (res.data == A.data@x.data).all()


def test_trace():
    data = np.arange(100, dtype=int).reshape((10, 10))
    A = Tensor([dtl.RealVectorSpace(10)]*2, "A", data.copy())
    i = dtl.Index("i")
    expr = dtl.Lambda([A], A[i, i].forall())

    tensor = apply(expr)
    assert (tensor.data == np.trace(data)).all()


def test_transpose():
    data = np.arange(100, dtype=int).reshape((10, 10))
    A = Tensor([dtl.RealVectorSpace(10)]*2, "A", data.copy())
    i = dtl.Index("i")
    j = dtl.Index("j")
    expr = dtl.Lambda([A], A[i, j].forall(j, i))

    tensor = apply(expr)
    assert (tensor.data == data.T).all()


def test_vecdot():
    x = Tensor([dtl.RealVectorSpace(10)], "x", np.arange(10, dtype=int))
    y = Tensor([dtl.RealVectorSpace(10)], "y", np.arange(10, dtype=int))

    i = dtl.Index("i")

    expr = dtl.Lambda([x, y], (x[i]*y[i]).forall())

    assert apply(expr).data == np.dot(np.arange(10, dtype=int), np.arange(10, dtype=int))
