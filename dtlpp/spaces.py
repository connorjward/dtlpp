import dtl


class RealVectorSpace(dtl.VectorSpace):

    symbol = "ℝ"


class UnitVectorSpace(dtl.VectorSpace):
    """Vector space where all entries are zero apart from (optionally) one."""
