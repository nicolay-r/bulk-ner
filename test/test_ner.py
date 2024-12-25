import unittest
from os.path import dirname, realpath, join

from bulk_ner.api import NERAnnotator
from bulk_ner.src.service_dynamic import dynamic_init


class TestTransformersNERPipeline(unittest.TestCase):
    
    text = "It was in July, 1805, and the speaker was the well-known Anna Pávlovna" \
           " Schérer, maid of honor and favorite of the Empress Márya Fëdorovna." \
           " With these words she greeted Prince Vasíli Kurágin, a man of high" \
           " rank and importance, who was the first to arrive at her reception. Anna" \
           " Pávlovna had had a cough for some days. She was, as she said, suffering" \
           " from la grippe; grippe being then a new word in St. Petersburg, used" \
           " only by the elite."

    CURRENT_DIR = dirname(realpath(__file__))

    def test_benchmark(self):

        ner_model = dynamic_init(src_dir=join(TestTransformersNERPipeline.CURRENT_DIR, "../models"),
                                 class_filepath="dp_130.py",
                                 class_name="DeepPavlovNER")(model="ner_ontonotes_bert")

        annotator = NERAnnotator(ner_model=ner_model,
                                 chunk_limit=128)

        data_it = annotator.iter_annotated_data(
            data_dict_it=[{"text": [TestTransformersNERPipeline.text]}],
            prompt="{text}")
        
        for d in data_it:
            print(d)