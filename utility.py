
import numpy as np

class myException(Exception):
    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg


class Data:
    def __init__(self):
        ## User defined parameters
        self.xlow = None
        self.xup = None
        self.objfunction = None
        self.dim = None
        self.const = None

