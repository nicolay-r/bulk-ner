import unittest
from os.path import dirname, realpath, join
from arekit.common.pipeline.batching import BatchingPipelineLauncher
from arekit.common.pipeline.context import PipelineContext

from bulk_ner.src.entity import IndexedEntity
from bulk_ner.src.pipeline.entity_list import HandleListPipelineItem
from bulk_ner.src.pipeline.ner import NERPipelineItem
from bulk_ner.src.service_dynamic import dynamic_init
from bulk_ner.src.utils import IdAssigner


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

        pipeline = [
            NERPipelineItem(id_assigner=IdAssigner(), model=ner_model, chunk_limit=128),
            HandleListPipelineItem(map_item_func=lambda i, e: (i, e.Type, e.Value),
                                   filter_item_func=lambda i: isinstance(i, IndexedEntity),
                                   result_key="listed-entities"),
            HandleListPipelineItem(map_item_func=lambda _, t: f"[{t.Type}]" if isinstance(t, IndexedEntity) else t),
        ]

        ctx = PipelineContext(d={"input": [TestTransformersNERPipeline.text]})

        BatchingPipelineLauncher.run(pipeline=pipeline, pipeline_ctx=ctx, src_key="input")

        print(ctx.provide("result"))
        print(ctx.provide("listed-entities"))
