class BatchIterator:

    def __init__(self, data_iter, batch_size):
        assert(isinstance(batch_size, int) and batch_size > 0)
        self.__data_iter = data_iter
        self.__index = 0
        self.__batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        buffer = []
        while True:
            try:
                data = next(self.__data_iter)
            except StopIteration:
                break
            buffer.append(data)
            if len(buffer) == self.__batch_size:
                break

        if len(buffer) > 0:
            self.__index += 1
            return buffer

        raise StopIteration