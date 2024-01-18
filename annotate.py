import argparse

import pandas as pd
from arekit.common.pipeline.batching import BatchingPipelineLauncher
from arekit.common.pipeline.context import PipelineContext
from arekit.common.utils import split_by_whitespaces

from src.batch_iter import BatchIterator
from src.data_service import DataService
from src.entity import IndexedEntity
from src.json_service import JsonlService
from src.pandas_service import PandasService
from src.pipeline.dp import DeepPavlovNERPipelineItem
from src.pipeline.entity_list import HandleListPipelineItem
from src.utils import IdAssigner


def iter_annotated_data(texts_it, batch_size):
    for batch in BatchIterator(texts_it, batch_size=batch_size):
        index, input = zip(*batch)
        ctx = BatchingPipelineLauncher.run(pipeline=pipeline,
                                           pipeline_ctx=PipelineContext(d={"index": index, "input": input}),
                                           src_key="input")

        # Target.
        d = ctx._d

        # Removing meta-information.
        for m in args.del_meta:
            del d[m]

        yield d


parser = argparse.ArgumentParser(description="Apply NER annotation")

parser.add_argument('--model', dest='model', type=str, default="ner_ontonotes_bert_mult")
parser.add_argument('--del-meta', dest="del_meta", type=list, default=["parent_ctx"])
parser.add_argument('--csv-sep', dest='csv_sep', type=str, default='\t')
parser.add_argument('--prompt', dest='prompt', type=str, default="{text}")
parser.add_argument('--src', dest='src', type=str, default="./data/test.csv")
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--batch-size', dest='batch_size', type=int, default=5)
parser.add_argument('--chunk-limit', dest='chunk_limit', type=int, default=128)

args = parser.parse_args()

# Provide the default output.
if args.output is None:
    args.output = ".".join(args.src.split('.')[:-1]) + "-converted.jsonl"

input_formatters = {
    "csv": lambda: PandasService.iter_rows_as_dict(df=pd.read_csv(args.src, sep=args.csv_sep))
}

output_formatters = {
    "jsonl": lambda dicts_it: JsonlService.write(output=args.output, lines_it=dicts_it)
}

# Application of the NER for annotation texts.
pipeline = [
    DeepPavlovNERPipelineItem(id_assigner=IdAssigner(), ner_model_name=args.model,
                              src_func=lambda text: split_by_whitespaces(text),
                              chunk_limit=args.chunk_limit),
    HandleListPipelineItem(map_item_func=lambda i, e: (i, e.Type, e.Value),
                           filter_item_func=lambda i: isinstance(i, IndexedEntity),
                           result_key="listed-entities"),
    HandleListPipelineItem(map_item_func=lambda _, t: f"[{t.Type}]" if isinstance(t, IndexedEntity) else t),
]

texts_it = input_formatters["csv"]()
prompts_it = DataService.iter_prompt(data_dict_it=texts_it, prompt=args.prompt)
ctxs_it = iter_annotated_data(texts_it=prompts_it, batch_size=args.batch_size)
output_formatters["jsonl"](dicts_it=ctxs_it)
