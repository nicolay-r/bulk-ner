class IdAssigner(object):

    def __init__(self):
        self.__id = 0

    def get_id(self):
        curr_id = self.__id
        self.__id += 1
        return curr_id


def iter_params(text):
    assert(isinstance(text, str))
    beg = 0
    while beg < len(text):
        try:
            pb = text.index('{', beg)
        except ValueError:
            break
        pe = text.index('}', beg+1)
        # Yield argument.
        yield text[pb+1:pe]
        beg = pe+1


def format_model_name(name):
    return name.replace("/", "_")


def iter_with_ids(content_it):
    for item_id, item in enumerate(content_it):
        yield item_id, item
