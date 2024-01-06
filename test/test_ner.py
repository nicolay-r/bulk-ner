import unittest

from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.common.pipeline.context import PipelineContext
from arekit.common.utils import split_by_whitespaces

from src.entity import IndexedEntity
from src.pipeline.dp import DeepPavlovNERPipelineItem

from src.pipeline.entity_list import HandleListPipelineItem
from src.utils import IdAssigner


class TestTransformersNERPipeline(unittest.TestCase):
    
    text = "It was in July, 1805, and the speaker was the well-known Anna Pávlovna" \
           " Schérer, maid of honor and favorite of the Empress Márya Fëdorovna." \
           " With these words she greeted Prince Vasíli Kurágin, a man of high" \
           " rank and importance, who was the first to arrive at her reception. Anna" \
           " Pávlovna had had a cough for some days. She was, as she said, suffering" \
           " from la grippe; grippe being then a new word in St. Petersburg, used" \
           " only by the elite."

    def test_benchmark(self):

        pipeline = [
            DeepPavlovNERPipelineItem(id_assigner=IdAssigner(),
                                      src_func=lambda text: split_by_whitespaces(text),
                                      ner_model_name="ner_ontonotes_bert"),
            HandleListPipelineItem(map_item_func=lambda i, e: (i, e.Type, e.Value),
                                   filter_item_func=lambda i: isinstance(i, IndexedEntity),
                                   result_key="listed-entities"),
            HandleListPipelineItem(map_item_func=lambda _, t: f"[{t.Type}]" if isinstance(t, IndexedEntity) else t),
        ]

        ctx = PipelineContext(d={"input": TestTransformersNERPipeline.text})

        BasePipelineLauncher.run(pipeline=pipeline, pipeline_ctx=ctx, src_key="input")

        print(ctx.provide("result"))
        print(ctx.provide("listed-entities"))
