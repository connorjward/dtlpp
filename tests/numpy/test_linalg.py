import dtl
import dtlc
import dtlpp
import numpy as np
import pytest


@pytest.fixture
def A():
    return dtl.TensorVariable(dtlpp.RealVectorSpace(10) ** 2, "A")


@pytest.fixture
def matdata():
    return np.arange(100, dtype=int).reshape((10, 10))


@pytest.fixture
def x():
    return dtl.TensorVariable(dtlpp.RealVectorSpace(10), "x")


@pytest.fixture
def y():
    return dtl.TensorVariable(dtlpp.RealVectorSpace(10), "y")


@pytest.fixture
def z():
    return dtl.TensorVariable(dtlpp.RealVectorSpace(10), "z")


@pytest.fixture
def vecdata():
    return np.arange(10, dtype=int)


def test_matvec(A, x, matdata, vecdata):
    i, j = dtl.indices("i", "j")
    expr = dtl.Lambda([A, x], (A[i, j] * x[j]).forall(i))

    func = dtlc.lower(expr, backend=dtlc.backends.Backend.PYTHON)
    result = func(var_map={A: matdata, x: vecdata})

    assert (result == matdata @ vecdata).all()


def test_trace(A, matdata):
    i = dtl.Index("i")
    expr = dtl.Lambda([A], A[i, i].forall())

    func = dtlc.lower(expr, backend=dtlc.backends.Backend.PYTHON)
    result = func(var_map={A: matdata})

    assert (result == np.trace(matdata)).all()


def test_transpose(A, matdata):
    i, j = dtl.indices("i", "j")
    expr = dtl.Lambda([A], A[i, j].forall(j, i))

    func = dtlc.lower(expr, backend=dtlc.backends.Backend.PYTHON)
    result = func(var_map={A: matdata})

    assert (result == matdata.T).all()


def test_vecdot(x, y, vecdata):
    i = dtl.Index("i")
    expr = dtl.Lambda([x, y], (x[i] * y[i]).forall())

    func = dtlc.lower(expr, backend=dtlc.backends.Backend.PYTHON)
    result = func(var_map={x: vecdata.copy(), y: vecdata.copy()})

    assert result == vecdata @ vecdata


def test_pointwise_vec_add(x, y, vecdata):
    i = dtl.Index("i")
    expr = dtl.Lambda([x, y], (x[i] + y[i]).forall(i))

    func = dtlc.lower(expr, backend=dtlc.backends.Backend.PYTHON)
    result = func(var_map={x: vecdata.copy(), y: vecdata.copy()})

    assert all(result == 2 * vecdata)


def test_pointwise_vec_mul(x, y, vecdata):
    i = dtl.Index("i")
    expr = dtl.Lambda([x, y], (x[i] * y[i]).forall(i))

    func = dtlc.lower(expr, backend=dtlc.backends.Backend.PYTHON)
    result = func(var_map={x: vecdata.copy(), y: vecdata.copy()})

    assert all(result == vecdata**2)


def test_ternary_add(x, y, z, vecdata):
    i = dtl.Index("i")
    expr = dtl.Lambda([x, y, z], (x[i] + y[i] + z[i]).forall(i))

    func = dtlc.lower(expr, backend=dtlc.backends.Backend.PYTHON)
    result = func(var_map={x: vecdata.copy(), y: vecdata.copy(), z: vecdata.copy()})

    assert all(result == 3 * vecdata)
