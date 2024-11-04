import logging
import sys


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
        

def parse_filepath(filepath, default_filepath=None, default_ext=None):
    """ This is an auxiliary function for handling sources and targets from cmd string.
    """
    if filepath is None:
        return default_filepath, default_ext, None
    info = filepath.split(":")
    filepath = info[0]
    meta = info[1] if len(info) > 1 else None
    ext = filepath.split('.')[-1] if default_ext is None else default_ext
    return filepath, ext, meta


def test_ner_demo(iter_answers=None):

    while True:

        user_input = input(f"Enter your text "
                           f"(or 'exit' to quit): ")

        if user_input.lower() == 'exit':
            break

        # Finally asking LLM.
        for a in iter_answers(user_input):
            print(a)


def setup_custom_logger(name, add_screen_handler=False, filepath=None):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if add_screen_handler:
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)

    if filepath is not None:
        handler = logging.FileHandler(filepath, mode='w')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger