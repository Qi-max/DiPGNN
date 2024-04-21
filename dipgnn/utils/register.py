import logging


class Register:
    """
    This code is extracted from delta (https://github.com/Delta-ML/delta).
    """
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, cls, name=None):
        def decorator(key, value):
            self[key] = value
            return value

        if callable(cls):
            return decorator(name, cls)

        return lambda x: decorator(cls, x)

    def get_class(self, key):
        return self.__getitem__(key)

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as e:
            logging.error(f"module {key} not found: {e}")
            raise e

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        return self._dict.keys()


class Registers():
    """All module registers."""
    task = Register('task')
    model = Register('model')
    data_container = Register('data_container')
    data_provider = Register('data_provider')

    def __init__(self):
        self._already_setup = False

    @property
    def already_setup(self):
        return self._already_setup

    @already_setup.setter
    def already_setup(self, value):
        self._already_setup = value


registers = Registers()
