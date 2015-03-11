import numpy as np
import jsonpickle


class NumpyHandler(jsonpickle.handlers.BaseHandler):
    """
    Automatic conversion of numpy float  to python floats
    Required for jsonpickle to work correctly
    """
    def flatten(self, obj, data):
        data['repr'] = repr(obj)
        return data

    def restore(self, data):
        return eval('np.' + data['repr'])


def register_handlers():
    print '!!'
    jsonpickle.handlers.registry.register(np.ndarray, NumpyHandler)
    jsonpickle.handlers.registry.register(np.matrix, NumpyHandler)
    return True
