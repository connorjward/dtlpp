import dtl


class LeftFold(dtl.Node):

    def __init__(self, function, initializer, iterable):
        self.function = function
        self.initializer = initializer
        self.iterable = iterable

    @property
    def operands(self):
        ops = self.function, self.iterable
        if self.initializer:
            ops += self.initializer,
        return ops
