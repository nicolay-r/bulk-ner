# This is an adapter implementation for
# transformers==4.24.0
import torch
import numpy as np
from transformers import BertForTokenClassification, BertTokenizerFast, pipeline

from bulk_ner.src.core.bound import Bound
from bulk_ner.src.partitioning import Partitioning
from bulk_ner.src.ner.base import BaseNER
from bulk_ner.src.pipeline.item.base import BasePipelineItem
from bulk_ner.src.entity_indexed import IndexedEntity
from bulk_ner.src.utils import IdAssigner


class TransformersNERPipelineItem(BasePipelineItem):
    """ NOTE: This code is expected to be refactored for nesting BaseNER.
    """

    def __init__(self, id_assigner, ner_model_name, device, obj_filter=None, display_value_func=None, **kwargs):
        """ chunk_limit: int
                length of text part in words that is going to be provided in input.
        """
        assert(callable(obj_filter) or obj_filter is None)
        assert(isinstance(id_assigner, IdAssigner))
        assert(callable(display_value_func) or display_value_func is None)
        super(TransformersNERPipelineItem, self).__init__(**kwargs)

        # Transformers-related parameters.
        self.__device = device
        self.__model, self.__tokenizer = init_token_classification_model(model_path=ner_model_name,
                                                                         device=self.__device)

        # Initialize bert-based model instance.
        self.__obj_filter = obj_filter
        self.__id_assigner = id_assigner
        self.__disp_value_func = display_value_func
        self.__partitioning = Partitioning(text_fmt="str")

    # region Private methods

    def __get_parts_provider_func(self, input_data):
        assert(isinstance(input_data, str))
        parts = annotate_ner(model=self.__model, tokenizer=self.__tokenizer, text=input_data, device=self.__device)
        for entity, bound in self.__iter_parsed_entities(parts):
            yield entity, bound

    def __iter_parsed_entities(self, parts):
        for p in parts:
            assert (isinstance(p, dict))
            value = p["word"]

            if len(value) == 0:
                continue

            if self.__obj_filter is not None and not self.__obj_filter(p["entity_group"]):
                continue

            entity = IndexedEntity(
                value=value, e_type=p["entity_group"], entity_id=self.__id_assigner.get_id())

            yield entity, Bound(pos=p["start"], length=p["end"] - p["start"])

    @staticmethod
    def __iter_fixed_terms(terms):
        for e in terms:
            if isinstance(e, str):
                for term in split_by_whitespaces(e):
                    yield term
            else:
                yield e

    # endregion

    def apply_core(self, input_data, pipeline_ctx):
        parts_it = self.__get_parts_provider_func(input_data)
        handled = self.__partitioning.provide(text=input_data, parts_it=parts_it)
        return list(self.__iter_fixed_terms(handled))


########################################################
# Service related methods.
########################################################

def split_by_whitespaces(text):
    """
    Assumes to perform a word separation including a variety of space entries.
    In terms of the latter we consider any whitespace separator.
    """
    assert(isinstance(text, str))
    return text.split()


def init_token_classification_model(model_path, device):
    model = BertForTokenClassification.from_pretrained(model_path)
    model = model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    return model, tokenizer


def annotate_ner_ppl(model, tokenizer, device="cpu", batch_size=4):
    return pipeline("ner", model=model, aggregation_strategy='simple', tokenizer=tokenizer,
                    grouped_entities=True, batch_size=batch_size, device=device)


def annotate_ner(model, tokenizer, text, device="cpu"):
    """ This code is related to collection of the annotated objects from texts.

        return: list of dict
            every dict object contains entity, entity_group, location of the object
    """
    # Tokenize the text and get the offset mappings (start and end character positions for each token)
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    inputs_with_offsets = tokenizer(text, return_offsets_mapping=True)
    offsets = inputs_with_offsets["offset_mapping"]

    # Passing inputs into the model.
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)[0].tolist()
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()

    results = []
    idx = 0

    while idx < len(predictions):

        pred = predictions[idx]
        label = model.config.id2label[pred]

        if label != "O":
            label = label[2:]

            start, _ = offsets[idx]

            # Grab all the tokens labeled with I-label
            all_scores = []
            while (idx < len(predictions) and model.config.id2label[predictions[idx]] in [
                f"{BaseNER.begin_tag}{BaseNER.separator}{label}",
                f"{BaseNER.inner_tag}{BaseNER.separator}{label}"]
            ):
                all_scores.append(probabilities[idx][pred])
                _, end = offsets[idx]
                idx += 1

            score = np.mean(all_scores).item()
            word = text[start:end]

            results.append({
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end
            }
            )
        idx += 1

    return results
