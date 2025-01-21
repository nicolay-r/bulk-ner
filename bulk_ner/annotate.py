import argparse
import sys

from tqdm import tqdm

from source_iter.service_csv import CsvService
from source_iter.service_jsonl import JsonlService

from bulk_ner.api import NERAnnotator, CWD
from bulk_ner.src.service_args import CmdArgsService
from bulk_ner.src.service_dynamic import dynamic_init
from bulk_ner.src.utils import parse_filepath, test_ner_demo, setup_custom_logger


if __name__ == '__main__':
    
    logger = setup_custom_logger("bulk-ner")

    parser = argparse.ArgumentParser(description="Apply NER annotation")

    parser.add_argument('--adapter', dest='adapter', type=str, default=None)
    parser.add_argument('--del-meta', dest="del_meta", type=list, default=["parent_ctx"])
    parser.add_argument('--prompt', dest='prompt', type=str, default="{text}")
    parser.add_argument('--src', dest='src', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=5)
    parser.add_argument('--chunk-limit', dest='chunk_limit', type=int, default=128)

    # Extract native arguments.
    native_args = CmdArgsService.extract_native_args(sys.argv, end_prefix="%%")
    args = parser.parse_args(args=native_args[1:])
    
    # Extract csv-related arguments.
    csv_args = CmdArgsService.find_grouped_args(lst=sys.argv, starts_with="%%csv", end_prefix="%%")
    csv_args_dict = CmdArgsService.args_to_dict(csv_args)

    # Extract model-related arguments and Initialize Large Language Model.
    model_args = CmdArgsService.find_grouped_args(lst=sys.argv, starts_with="%%m", end_prefix="%%")
    model_args_dict = CmdArgsService.args_to_dict(model_args)

    # Provide the default output.
    if args.output is None and args.src is not None:
        args.output = ".".join(args.src.split('.')[:-1]) + "-converted.jsonl"

    # Parse the model name.
    params = args.adapter.split(':')

    # Completing the remaining parameters.
    ner_model_name = params[1] if len(params) > 1 else params[-1]
    ner_model_params = ':'.join(params[2:]) if len(params) > 2 else None

    # Initialize NER model
    models_preset = {
        "dynamic": lambda: dynamic_init(src_dir=CWD, class_filepath=ner_model_name, class_name=ner_model_params)(
            # The rest of parameters could be provided from cmd.
            **model_args_dict)
    }

    annotator = NERAnnotator(ner_model=models_preset["dynamic"](),
                             entity_func=lambda t: [t.Value, t.Type, t.ID],
                             chunk_limit=args.chunk_limit)

    input_formatters = {
        None: lambda _: test_ner_demo(
            iter_answers=lambda example: annotator.iter_annotated_data(data_dict_it=iter([(0, example)]),
                                                                       prompt=args.prompt,
                                                                       batch_size=1)),
        "csv": lambda filepath: CsvService.read(src=filepath, as_dict=True, skip_header=True,
                                                delimiter=csv_args_dict.get("delimiter", ","),
                                                escapechar=csv_args_dict.get("escapechar", None)),
        "tsv": lambda filepath: CsvService.read(src=filepath, as_dict=True, skip_header=True,
                                                delimiter=csv_args_dict.get("delimiter", "\t"),
                                                escapechar=csv_args_dict.get("escapechar", None)),
        "jsonl": lambda filepath: JsonlService.read(src=filepath)
    }

    output_formatters = {
        "jsonl": lambda dicts_it: JsonlService.write(target=args.output, data_it=dicts_it)
    }

    _, src_ext, _ = parse_filepath(args.src)
    texts_it = input_formatters[src_ext](args.src)

    # There is no need to perform export.
    if src_ext is None:
        exit(0)

    ctxs_it = annotator.iter_annotated_data(data_dict_it=texts_it, prompt=args.prompt, batch_size=args.batch_size)
    output_formatters["jsonl"](dicts_it=tqdm(ctxs_it, desc=f"Processing `{args.src}`"))

    logger.info(f"Saved: {args.output}")
