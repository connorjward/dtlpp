import dtl
import numpy as np
import pytest

import dtlpp
from dtlpp.impls.numpy.tensor import Tensor, apply


@pytest.fixture
def x():
    return Tensor([dtlpp.RealVectorSpace(10)], "x", np.arange(10, dtype=int))


@pytest.fixture
def y():
    return Tensor([dtlpp.RealVectorSpace(10)], "y", np.arange(10, dtype=int))


@pytest.fixture
def z():
    return Tensor([dtlpp.RealVectorSpace(10)], "z", np.arange(10, dtype=int))


def test_matvec():
    A = Tensor(
        dtlpp.RealVectorSpace(10) ** 2, "A", np.arange(100, dtype=int).reshape((10, 10))
    )
    x = Tensor(dtlpp.RealVectorSpace(10), "x", np.arange(10, dtype=int))
    i, j = dtl.indices("i", "j")
    expr = dtl.Lambda([A, x], (A[i, j] * x[j]).forall(i))

    res = apply(expr)
    assert (res.data == A.data @ x.data).all()


def test_trace():
    data = np.arange(100, dtype=int).reshape((10, 10))
    A = Tensor([dtlpp.RealVectorSpace(10)] * 2, "A", data.copy())
    i = dtl.Index("i")
    expr = dtl.Lambda([A], A[i, i].forall())

    tensor = apply(expr)
    assert (tensor.data == np.trace(data)).all()


def test_transpose():
    data = np.arange(100, dtype=int).reshape((10, 10))
    A = Tensor([dtlpp.RealVectorSpace(10)] * 2, "A", data.copy())
    i, j = dtl.indices("i", "j")
    expr = dtl.Lambda([A], A[i, j].forall(j, i))

    tensor = apply(expr)
    assert (tensor.data == data.T).all()


def test_vecdot():
    x = Tensor([dtlpp.RealVectorSpace(10)], "x", np.arange(10, dtype=int))
    y = Tensor([dtlpp.RealVectorSpace(10)], "y", np.arange(10, dtype=int))

    i = dtl.Index("i")

    expr = dtl.Lambda([x, y], (x[i] * y[i]).forall())

    assert apply(expr).data == np.dot(
        np.arange(10, dtype=int), np.arange(10, dtype=int)
    )


def test_pointwise_vec_add(x, y):
    i = dtl.Index("i")
    expr = dtl.Lambda([x, y], (x[i] + y[i]).forall(i))
    assert all(apply(expr).data == x.data + y.data)


def test_pointwise_vec_mul(x, y):
    i = dtl.Index("i")
    expr = dtl.Lambda([x, y], (x[i] * y[i]).forall(i))
    assert all(apply(expr).data == x.data * y.data)


def test_ternary_add(x, y, z):
    i = dtl.Index("i")
    expr = dtl.Lambda([x, y, z], (x[i] + y[i] + z[i]).forall(i))
    assert all(apply(expr).data == x.data + y.data + z.data)
