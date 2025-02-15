import os

from bulk_ner.src.entity_indexed import IndexedEntity
from bulk_ner.src.pipeline.batching import BatchingPipelineLauncher
from bulk_ner.src.pipeline.context import PipelineContext
from bulk_ner.src.pipeline.entity_list import HandleListPipelineItem
from bulk_ner.src.pipeline.item.base import BasePipelineItem
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

    def handle_batch(self, batch, col_output, col_prompt=None):
        assert(isinstance(batch, list))

        if col_prompt is None:
            col_prompt = col_output

        ctx = BatchingPipelineLauncher.run(pipeline=self.pipeline,
                                           pipeline_ctx=PipelineContext(d={col_prompt: batch}),
                                           src_key=col_prompt)

        # Target.
        d = ctx._d

        for batch_ind in range(len(d[col_prompt])):

            yield {(k if k != BasePipelineItem.DEFAULT_RESULT_KEY else col_output):
                       v[batch_ind] for k, v in d.items()}

    def iter_annotated_data(self, data_dict_it, schema, batch_size=1, keep_prompt=False):
        """ This is the main API method for calling.
        """
        assert(isinstance(schema, dict))

        for data_batch in BatchIterator(data_dict_it, batch_size=batch_size):
            for col_output, prompt in schema.items():

                prompts_it = DataService.iter_prompt(data_dict_it=data_batch,
                                                     prompt=prompt,
                                                     parse_fields_func=iter_params)

                handled_data_it = self.handle_batch(batch=list(prompts_it),
                                                    col_output=col_output,
                                                    col_prompt=f"prompt_{col_output}" if keep_prompt else None)

                # Applying updated content from the handled column.
                for record_ind, record in enumerate(handled_data_it):
                    data_batch[record_ind] |= record

            for item in data_batch:
                yield item
