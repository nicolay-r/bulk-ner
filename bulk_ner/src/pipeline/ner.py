from arekit.common.bound import Bound
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.pipeline.utils import BatchIterator
from arekit.common.text.partitioning import Partitioning

from bulk_ner.src.entity import IndexedEntity
from bulk_ner.src.ner.obj_desc import NerObjectDescriptor
from bulk_ner.src.utils import IdAssigner


class ChunkIterator:

    def __init__(self, data_iter, batch_size, chunk_limit):
        assert(isinstance(batch_size, int) and batch_size > 0)
        self.__data_iter = data_iter
        self.__index = -1
        self.__batch_size = batch_size
        self.__chunk_limit = chunk_limit
        self.__buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if len(self.__buffer) > 0:
                break
            try:
                data = next(self.__data_iter)
                self.__index += 1
            except StopIteration:
                break
            for chunk_start in range(0, len(data), self.__chunk_limit):
                chunk = data[chunk_start:chunk_start + self.__chunk_limit]
                self.__buffer.append([self.__index, chunk])

        if len(self.__buffer) > 0:
            return self.__buffer.pop(0)

        raise StopIteration


class NERPipelineItem(BasePipelineItem):

    def __init__(self, id_assigner, model, obj_filter=None, chunk_limit=128,
                 display_value_func=None, **kwargs):
        """ chunk_limit: int
                length of text part in words that is going to be provided in input.
        """
        assert(callable(obj_filter) or obj_filter is None)
        assert(isinstance(chunk_limit, int) and chunk_limit > 0)
        assert(isinstance(id_assigner, IdAssigner))
        assert(callable(display_value_func) or display_value_func is None)
        super(NERPipelineItem, self).__init__(**kwargs)

        # Initialize bert-based model instance.
        self.__dp_ner = model
        self.__obj_filter = obj_filter
        self.__chunk_limit = chunk_limit
        self.__id_assigner = id_assigner
        self.__disp_value_func = display_value_func
        self.__partitioning = Partitioning(text_fmt="list")

    @property
    def SupportBatching(self):
        return True

    def __iter_subs_values_with_bounds(self, batch_it):
        chunk_offset = 0
        handled_text_index = -1
        for batch in batch_it:
            text_indices, texts = zip(*batch)

            try:
                data = self.__dp_ner.extract(sequences=list(texts))
            except RuntimeError:
                data = None

            if data is not None:
                for i, d in enumerate(data):
                    terms, descriptors = d
                    if text_indices[i] != handled_text_index:
                        chunk_offset = 0
                    entities_it = self.__iter_parsed_entities(
                        descriptors=descriptors, terms_list=terms, chunk_offset=chunk_offset)
                    handled_text_index = text_indices[i]
                    chunk_offset += len(terms)
                    yield text_indices[i], terms, list(entities_it)
            else:
                for i in range(len(batch)):
                    yield text_indices[i], texts[i], []

    def __iter_parsed_entities(self, descriptors, terms_list, chunk_offset):
        for s_obj in descriptors:
            assert (isinstance(s_obj, NerObjectDescriptor))

            if self.__obj_filter is not None and not self.__obj_filter(s_obj):
                continue

            value = " ".join(terms_list[s_obj.Position:s_obj.Position + s_obj.Length])
            entity = IndexedEntity(
                value=value, e_type=s_obj.ObjectType, entity_id=self.__id_assigner.get_id(),
                display_value=self.__disp_value_func(value) if self.__disp_value_func is not None else None)
            yield entity, Bound(pos=chunk_offset + s_obj.Position, length=s_obj.Length)

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, list))

        batch_size = len(input_data)

        c_it = ChunkIterator(iter(input_data), batch_size=batch_size, chunk_limit=self.__chunk_limit)
        b_it = BatchIterator(c_it, batch_size=batch_size)

        parts_it = self.__iter_subs_values_with_bounds(b_it)

        terms = [[] for _ in range(batch_size)]
        bounds = [[] for _ in range(batch_size)]
        for i, t, e in parts_it:
            terms[i].extend(t)
            bounds[i].extend(e)

        for i in range(batch_size):
            yield self.__partitioning.provide(text=terms[i], parts_it=bounds[i])
