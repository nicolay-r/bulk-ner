class Entity(object):

    def __init__(self, value, e_type):
        assert(isinstance(value, str) and len(value) > 0)
        assert(isinstance(e_type, str) or e_type is None)
        self.__value = value
        self.__type = e_type

    @property
    def Value(self):
        return self.__value

    @property
    def Type(self):
        return self.__type
