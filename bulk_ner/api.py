import os

from bulk_ner.src.entity_indexed import IndexedEntity
from bulk_ner.src.pipeline.batching import BatchingPipelineLauncher
from bulk_ner.src.pipeline.context import PipelineContext
from bulk_ner.src.pipeline.entity_list import HandleListPipelineItem
from bulk_ner.src.pipeline.item.merge import MergeTextEntries
from bulk_ner.src.pipeline.item.ner import NERPipelineItem
from bulk_ner.src.pipeline.utils import BatchIterator
from bulk_ner.src.service_prompt import DataService
from bulk_ner.src.utils import IdAssigner, iter_params


CWD = os.getcwd()


class NERAnnotator(object):

    def __init__(self, ner_model, chunk_limit, entity_func=None, do_merge_terms=True):

        entity_func = (lambda item: item) if entity_func is None else entity_func

        self.pipeline = [
            NERPipelineItem(id_assigner=IdAssigner(),
                            model=ner_model,
                            chunk_limit=chunk_limit,
                            create_entity_func=lambda **kwargs: IndexedEntity(**kwargs)),
            HandleListPipelineItem(
                map_item_func=lambda _, t: entity_func(t) if isinstance(t, IndexedEntity) else t),
            MergeTextEntries() if do_merge_terms else None
        ]

    def iter_annotated_data(self, data_dict_it, prompt, batch_size=1):
        """ This is the main API method for calling.
        """

        prompts_it = DataService.iter_prompt(data_dict_it=data_dict_it, prompt=prompt, parse_fields_func=iter_params)

        for batch in BatchIterator(prompts_it, batch_size=batch_size):
            index, input = zip(*batch)
            ctx = BatchingPipelineLauncher.run(pipeline=self.pipeline,
                                               pipeline_ctx=PipelineContext(d={"input": input}),
                                               src_key="input")

            # Target.
            d = ctx._d

            for batch_ind in range(len(d["input"])):
                yield {k: v[batch_ind] for k, v in d.items()}
