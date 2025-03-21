import time
import unittest
import deeppavlov

from bulk_ner.src.pipeline.utils import BatchIterator


class TestNerBatching(unittest.TestCase):

    def test(self):

        self.__ner_model = deeppavlov.build_model("ner_ontonotes_bert_mult", download=True, install=True)

        TEXTS = ["This is America and this is Europe"]

        texts = [t.split() for t in TEXTS]
        print("------")
        for t in texts:
            print(len(t))
        print("------")

        for batch_size in range(len(texts)):
            start = time.time()
            for batch in BatchIterator(iter(texts), batch_size=batch_size + 1):
                _ = self.__ner_model(batch)
            end = time.time()
            print(f"BS: {batch_size}", end - start)
