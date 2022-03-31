import dtl
import numpy as np
import pytest

from dtlpp.impls.numpy.tensor import Tensor, apply


@pytest.fixture
def x():
    return Tensor([dtl.RealVectorSpace(10)], "x", np.arange(10, dtype=int))


@pytest.fixture
def y():
    return Tensor([dtl.RealVectorSpace(10)], "y", np.arange(10, dtype=int))


@pytest.fixture
def z():
    return Tensor([dtl.RealVectorSpace(10)], "z", np.arange(10, dtype=int))


def test_matvec():
    A = Tensor("A", [dtl.RealVectorSpace(10)]*2, np.arange(100, dtype=int).reshape((10, 10)))
    x = Tensor("x", [dtl.RealVectorSpace(10)], np.arange(10, dtype=int))
    i = dtl.Index("i", dtl.RealVectorSpace(10))
    j = dtl.Index("j", dtl.RealVectorSpace(10))
    expr = dtl.Lambda([A, x], (A[i, j]*x[j]).forall(i))

    res = apply(expr)
    assert (res.data == A.data@x.data).all()


def test_trace():
    data = np.arange(100, dtype=int).reshape((10, 10))
    A = Tensor([dtl.RealVectorSpace(10)]*2, "A", data.copy())
    i = dtl.Index("i", dtl.RealVectorSpace(10))
    expr = dtl.Lambda([A], A[i, i].forall())

    tensor = apply(expr)
    assert (tensor.data == np.trace(data)).all()


def test_transpose():
    data = np.arange(100, dtype=int).reshape((10, 10))
    A = Tensor([dtl.RealVectorSpace(10)]*2, "A", data.copy())
    i = dtl.Index("i", dtl.RealVectorSpace(10))
    j = dtl.Index("j", dtl.RealVectorSpace(10))
    expr = dtl.Lambda([A], A[i, j].forall(j, i))

    tensor = apply(expr)
    assert (tensor.data == data.T).all()


def test_vecdot():
    x = Tensor([dtl.RealVectorSpace(10)], "x", np.arange(10, dtype=int))
    y = Tensor([dtl.RealVectorSpace(10)], "y", np.arange(10, dtype=int))

    i = dtl.Index("i", dtl.RealVectorSpace(10))

    expr = dtl.Lambda([x, y], (x[i]*y[i]).forall())

    assert apply(expr).data == np.dot(np.arange(10, dtype=int), np.arange(10, dtype=int))


def test_pointwise_vec_add(x, y):
    i = dtl.Index("i", dtl.RealVectorSpace(10))
    expr = dtl.Lambda([x, y], (x[i]+y[i]).forall(i))
    assert all(apply(expr).data == x.data + y.data)


def test_pointwise_vec_mul(x, y):
    i = dtl.Index("i", dtl.RealVectorSpace(10))
    expr = dtl.Lambda([x, y], (x[i]*y[i]).forall(i))
    assert all(apply(expr).data == x.data * y.data)


def test_ternary_add(x, y, z):
    i = dtl.Index("i", dtl.RealVectorSpace(10))
    expr = dtl.Lambda([x, y, z], (x[i]+y[i]+z[i]).forall(i))
    assert all(apply(expr).data == x.data + y.data + z.data)
