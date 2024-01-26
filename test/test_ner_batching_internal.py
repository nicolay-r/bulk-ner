import json
import unittest
from pathlib import Path
from typing import Union

from deeppavlov import build_model, Chainer
from deeppavlov.core.data.utils import jsonify_data

from data import TEXTS
from src.batch_iter import BatchIterator


def predict_on_stream(config: Union[str, Path, dict], get_batch_func) -> None:
    """Make a prediction with the component described in corresponding configuration file."""

    model: Chainer = build_model(config)

    args_count = len(model.in_x)
    while True:
        batch = get_batch_func()

        if not batch:
            break

        args = []
        for i in range(args_count):
            args.append(batch[i::args_count])

        res = model(*args)
        if len(model.out_params) == 1:
            res = [res]
        for res in zip(*res):
            res = json.dumps(jsonify_data(res), ensure_ascii=False)
            yield res


class TestNerBatching(unittest.TestCase):

    def test_predict_stream_call(self):
        batch_size = 4
        batch_it = BatchIterator(iter(TEXTS), batch_size=batch_size, end_value=lambda: None)
        for data in predict_on_stream("ner_ontonotes_bert_mult", get_batch_func=lambda: next(batch_it)):
            print("---")
            print(data)

