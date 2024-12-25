import os

from arekit.common.pipeline.batching import BatchingPipelineLauncher
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.utils import BatchIterator

from bulk_ner.src.entity import IndexedEntity
from bulk_ner.src.pipeline.entity_list import HandleListPipelineItem
from bulk_ner.src.pipeline.ner import NERPipelineItem
from bulk_ner.src.service_prompt import DataService
from bulk_ner.src.utils import IdAssigner, iter_params


CWD = os.getcwd()


class NERAnnotator(object):

    def __init__(self, ner_model, chunk_limit):
        self.pipeline = [
            NERPipelineItem(id_assigner=IdAssigner(),
                            model=ner_model,
                            chunk_limit=chunk_limit),
            HandleListPipelineItem(map_item_func=lambda i, e: (i, e.Type, e.Value),
                                   filter_item_func=lambda i: isinstance(i, IndexedEntity),
                                   result_key="listed-entities"),
            HandleListPipelineItem(map_item_func=lambda _, t: f"[{t.Type}]" if isinstance(t, IndexedEntity) else t),
        ]

    def iter_annotated_data(self, data_dict_it, prompt, batch_size=1):
        """ This is the main API method for calling.
        """

        prompts_it = DataService.iter_prompt(data_dict_it=data_dict_it, prompt=prompt, parse_fields_func=iter_params)

        for batch in BatchIterator(prompts_it, batch_size=batch_size):
            index, input = zip(*batch)
            ctx = BatchingPipelineLauncher.run(pipeline=self.pipeline,
                                               pipeline_ctx=PipelineContext(d={"index": index, "input": input}),
                                               src_key="input")

            # Target.
            d = ctx._d

            for batch_ind in range(len(d["input"])):
                yield {k: v[batch_ind] if v is not None else None for k, v in d.items()}
