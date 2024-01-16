import time
import unittest
import deeppavlov
from arekit.common.pipeline.utils import BatchIterator


class TestNerBatching(unittest.TestCase):

    def test(self):

        texts = [
            "Jessica, a talented violinist, performed a mesmerizing solo at the grand concert hall.",
            "The bustling city streets were filled with people hurrying to catch the next train, while the aroma of freshly brewed coffee wafted from a quaint cafe.",
            "Professor Anderson, an esteemed archaeologist, uncovered ancient artifacts that provided new insights into the lost civilization.",
            "Emily, an ambitious entrepreneur, launched her innovative tech startup, aiming to revolutionize the way we interact with artificial intelligence.",
            "The majestic Mount Everest, standing tall at 29,032 feet, attracted daring adventurers from around the globe.",
            "Captain Rodriguez skillfully navigated the colossal ocean liner through the treacherous waters, ensuring the safety of all passengers on board.",
            "In the enchanting forest, a wise old owl named Archimedes imparted valuable advice to the curious young fox, Oliver.",
            "The vibrant colors of the tropical fish mesmerized Lisa as she snorkeled in the crystal-clear waters of the Great Barrier Reef.",
            "Detective Miller, known for his keen intellect, solved the perplexing mystery that had baffled the local police force for weeks.",
            "As the spacecraft entered the orbit of Mars, Dr. Williams, a seasoned astronaut, marveled at the breathtaking Martian landscape unfolding before him.",
        ]

        self.__ner_model = deeppavlov.build_model("ner_ontonotes_bert_mult", download=True, install=True)

        texts = [t.split() for t in texts]
        print("------")
        for t in texts:
            print(len(t))
        print("------")

        for batch_size in range(len(texts)):
            start = time.time()
            for batch in BatchIterator(texts, batch_size=batch_size + 1):
                a, b = self.__ner_model(batch)
                for aa, bb in zip(a, b):
                    print(len(aa), len(bb))
            end = time.time()
            print(f"BS: {batch_size}", end - start)
