import abc
import enum
from typing import Iterable

import dtl


class Monad(dtl.Node, abc.ABC):
    ...



class MonadExpression(Monad):
    # note that this is technically a functor subclass (it permits a map operation)
    def __init__(self, monad_expr):
        """
        function is an expression that evaluates to a monad (this obj)
        """
        self.monad_expr = monad_expr

    @abc.abstractmethod
    def bind(self, other):
        ...

    @property
    def is_bound(self):
        return bool(self.monad_expr)


class MonadVariable(Monad):

    def __init__(self, value, monoid):
        self.value = value
        self.monoid = monoid

    @property
    def operands(self):
        return self.value, self.monoid


class State(dtl.Node):
    """This is NOT a monad. This is what gets passed between them."""

    def __init__(self, *operands):
        self.operands = operands


class StateMonad(MonadExpression, abc.ABC):
    """A monad is a function that takes some operands and returns a result + a new state.
    """


class AccessDescriptor(enum.Enum):

    READ: enum.auto()
    WRITE: enum.auto()
    INC: enum.auto()


class FunctionCallArgument:

    def __init__(self, tensor: dtl.TensorExpr, access: AccessDescriptor):
        self.tensor = tensor
        self.access = access


# it inherits from StateMonad because that is the output.
class FunctionCall(StateMonad):
    """
    A function call should act on a state monad and output another one with no arguments.

    It inherits from StateMonad because that is what it chucks back out.
    """

    def __init__(self, function: dtl.Lambda, arguments, monad_expr: StateMonad=None):
        self.function = function
        self.arguments = arguments
        self.monad_expr = monad_expr

    def __str__(self):
        return f"{self.function}({', '.join(map(str, self.arguments))})"

    @property
    def operands(self):
        ops = self.function, *self.arguments
        if self.monad_expr:
            ops += self.monad_expr,
        return ops

    def bind(self, monad_expr):
        assert not self.is_bound
        return type(self)(self.function, self.arguments, monad_expr)
