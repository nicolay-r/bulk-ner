import argparse

import pandas as pd
from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser

from src.entity import IndexedEntity
from src.json_service import JsonlService
from src.pandas_service import PandasService
from src.pipeline.dp import DeepPavlovNERPipelineItem
from src.pipeline.entity_list import HandleListPipelineItem
from src.utils import IdAssigner

parser = argparse.ArgumentParser(description="Apply NER annotation")

parser.add_argument('--model', dest='model', type=str, default="ner_ontonotes_bert_mult")
parser.add_argument('--del-meta', dest="del_meta", type=list, default=["parent_ctx"])
parser.add_argument('--csv-sep', dest='csv_sep', type=str, default='\t')
parser.add_argument('--prompt', dest='prompt', type=str, default="{text}")
parser.add_argument('--src', dest='src', type=str, default="./data/test.csv")
parser.add_argument('--output', dest='output', type=str, default=None)

args = parser.parse_args()

# Provide the default output.
if args.output is None:
    args.output = ".".join(args.src.split('.')[:-1]) + "-converted.jsonl"

input_formatters = {
    "csv": lambda: PandasService.iter_prompts(pd.read_csv(args.src, sep=args.csv_sep), prompt=args.prompt)
}

output_formatters = {
    "jsonl": lambda dicts_it: JsonlService.write(output=args.output, lines_it=dicts_it)
}


def iter_annotated_data(texts_it):
    for text_index, p in texts_it:
        ctx = BasePipelineLauncher.run(pipeline=pipeline,
                                       pipeline_ctx=PipelineContext(d={"index": text_index, "input": p}),
                                       src_key="input")

        # Target.
        d = ctx._d

        # Removing meta-information.
        for m in args.del_meta:
            del d[m]

        yield d


# Application of the NER for annotation texts.
pipeline = [
    TermsSplitterParser(),
    DeepPavlovNERPipelineItem(id_assigner=IdAssigner(), ner_model_name=args.model),
    HandleListPipelineItem(map_item_func=lambda i, e: (i, e.Type, e.Value),
                           filter_item_func=lambda i: isinstance(i, IndexedEntity),
                           result_key="listed-entities"),
    HandleListPipelineItem(map_item_func=lambda _, t: f"[{t.Type}]" if isinstance(t, IndexedEntity) else t),
]

prompts_it = PandasService.iter_prompts(df=pd.read_csv(args.src, sep=args.csv_sep), prompt=args.prompt)
ctxs_it = iter_annotated_data(texts_it=prompts_it)
output_formatters["jsonl"](dicts_it=ctxs_it)
