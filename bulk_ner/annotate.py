import argparse
import os
import sys
from tqdm import tqdm

from arekit.common.pipeline.batching import BatchingPipelineLauncher
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.utils import BatchIterator

from bulk_ner.src.entity import IndexedEntity
from bulk_ner.src.pipeline.entity_list import HandleListPipelineItem
from bulk_ner.src.pipeline.ner import NERPipelineItem
from bulk_ner.src.service import JsonlService, DataService, CsvService
from bulk_ner.src.service_args import CmdArgsService
from bulk_ner.src.service_dynamic import dynamic_init
from bulk_ner.src.utils import IdAssigner, iter_params, parse_filepath, test_ner_demo


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

        for batch_ind in range(len(d["input"])):
            yield {k: v[batch_ind] for k, v in d.items()}


CWD = os.getcwd()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Apply NER annotation")

    parser.add_argument('--adapter', dest='adapter', type=str, default=None)
    parser.add_argument('--del-meta', dest="del_meta", type=list, default=["parent_ctx"])
    parser.add_argument('--csv-sep', dest='csv_sep', type=str, default='\t')
    parser.add_argument('--csv-escape-char', dest='csv_escape_char', type=str, default=None)
    parser.add_argument('--prompt', dest='prompt', type=str, default="{text}")
    parser.add_argument('--src', dest='src', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=5)
    parser.add_argument('--chunk-limit', dest='chunk_limit', type=int, default=128)

    native_args, model_args = CmdArgsService.partition_list(lst=sys.argv, sep="%%")

    args = parser.parse_args(args=native_args[1:])

    # Provide the default output.
    if args.output is None and args.src is not None:
        args.output = ".".join(args.src.split('.')[:-1]) + "-converted.jsonl"

    input_formatters = {
        None: lambda _: test_ner_demo(
            iter_answers=lambda example: iter_annotated_data(texts_it=iter([(0, example)]), batch_size=1)),
        "csv": lambda filepath: CsvService.read(target=filepath, delimiter=args.csv_sep,
                                                as_dict=True, skip_header=True, escapechar=args.csv_escape_char),
        "jsonl": lambda filepath: JsonlService.read_lines(src=filepath)
    }

    output_formatters = {
        "jsonl": lambda dicts_it: JsonlService.write(output=args.output, lines_it=dicts_it)
    }

    # Initialize NER model
    models_preset = {
        "dynamic": lambda: dynamic_init(src_dir=CWD, class_filepath=ner_model_name, class_name=ner_model_params)(
            # The rest of parameters could be provided from cmd.
            **CmdArgsService.args_to_dict(model_args))
    }

    # Parse the model name.
    params = args.adapter.split(':')

    # Making sure that we refer to the supported preset.
    assert(params[0] in models_preset)

    # Completing the remaining parameters.
    ner_model_name = params[1] if len(params) > 1 else params[-1]
    ner_model_params = ':'.join(params[2:]) if len(params) > 2 else None

    # Application of the NER for annotation texts.
    pipeline = [
        NERPipelineItem(id_assigner=IdAssigner(),
                        model=models_preset["dynamic"](),
                        chunk_limit=args.chunk_limit),
        HandleListPipelineItem(map_item_func=lambda i, e: (i, e.Type, e.Value),
                               filter_item_func=lambda i: isinstance(i, IndexedEntity),
                               result_key="listed-entities"),
        HandleListPipelineItem(map_item_func=lambda _, t: f"[{t.Type}]" if isinstance(t, IndexedEntity) else t),
    ]

    _, src_ext, _ = parse_filepath(args.src)
    texts_it = input_formatters[src_ext](args.src)

    # There is no need to perform export.
    if src_ext is None:
        exit(0)

    prompts_it = DataService.iter_prompt(data_dict_it=texts_it, prompt=args.prompt, parse_fields_func=iter_params)
    ctxs_it = iter_annotated_data(texts_it=prompts_it, batch_size=args.batch_size)
    output_formatters["jsonl"](dicts_it=tqdm(ctxs_it, desc=f"Processing `{args.src}`"))
