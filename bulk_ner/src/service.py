import csv
import json


class JsonlService:

    @staticmethod
    def write(output, lines_it):
        with open(output, "w", encoding='utf8') as f:
            for line in lines_it:
                json.dump(line, fp=f, ensure_ascii=False)
                f.write("\n")

    @staticmethod
    def read_lines(src, row_id_key=None):
        assert (isinstance(src, str))
        with open(src, "r") as f:
            for line_ind, line in enumerate(f.readlines()):
                content = json.loads(line)
                if row_id_key is not None:
                    content[row_id_key] = line_ind
                print(content)
                yield content


class CsvService:

    @staticmethod
    def read(target, skip_header=False, cols=None, as_dict=False, row_id_key=None, **csv_kwargs):
        assert (isinstance(row_id_key, str) or row_id_key is None)
        assert (isinstance(cols, list) or cols is None)

        header = None
        with open(target, newline='\n') as f:
            for row_id, row in enumerate(csv.reader(f, **csv_kwargs)):
                if skip_header and row_id == 0:
                    header = ([row_id_key] if row_id_key is not None else []) + row
                    continue

                # Determine the content we wish to return.
                if cols is None:
                    content = row
                else:
                    row_d = {header[col_ind]: value for col_ind, value in enumerate(row)}
                    content = [row_d[col_name] for col_name in cols]

                content = ([row_id-1] if row_id_key is not None else []) + content

                # Optionally attach row_id to the content.
                if as_dict:
                    assert (header is not None)
                    assert (len(content) == len(header))
                    yield {k: v for k, v in zip(header, content)}
                else:
                    yield content


class DataService(object):

    @staticmethod
    def iter_prompt(data_dict_it, prompt, parse_fields_func):
        """ This method composes prompt from the multiple fields, mentioned in it.
            data_it: Iterator
                iterator of the dict, from which we can collect data.
        """
        assert(callable(parse_fields_func))
        field_names = list(parse_fields_func(prompt))
        for row_id, data_dict in enumerate(data_dict_it):
            assert(isinstance(data_dict, dict))
            fmt_d = {col_name: data_dict[col_name] for col_name in field_names}
            yield row_id, prompt.format(**fmt_d)
