import numpy as np
from core import *


class DataSet:
    methods = ("gpr", "kde", "classic")

    def replace_data(self, x=None, fx=None):
        if x is not None:
            self.x = exactly_2d(x)
        if fx is not None:
            self.fx = exactly_2d(fx)
        assert len(self.x) == len(self.fx), "x and fx must be the same length after reshaping into a matrix."
        self.n = len(self.x)
        self.dim = self.x.shape[1]
        self.output_dim = self.fx.shape[1]

    def __init__(self, x, fx):
        self.x = None
        self.fx = None
        self.n = None
        self.dim = None
        self.output_dim = None
        self.replace_data(x=x, fx=fx)

    def __getitem__(self, item):
        return DataSet(self.x[item], self.fx[item])

    def __str__(self):
        return "x:\n" + self.x.__str__() + "\n fx:\n" + self.fx.__str__()

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.n:
            self.i += 1
            return self[self.i - 1]
        else:
            raise StopIteration

    def __len__(self):
        return self.n

    def extend(self, x, fx):
        new_x = np.concatenate((self.x, x), axis=0)
        new_fx = np.concatenate((self.fx, fx), axis=0)
        self.replace_data(new_x, new_fx)
