from tqdm import tqdm

from src.utils import iter_params, iter_with_ids


class PandasService(object):
    
    @staticmethod
    def iter_texts(df, cols=None):
        for row_ind, data in tqdm(df.iterrows(), total=len(df)):
            data_dict = data.to_dict()
            yield [data_dict[c] for c in cols]

    @staticmethod
    def iter_prompts(df, prompt):
        column_names = list(iter_params(prompt))
        it = PandasService.iter_texts(df=df, cols=column_names)
        for row_id, item in iter_with_ids(it):
            fmt_d = {col_name: item[col_ind] for col_ind, col_name in enumerate(column_names)}
            yield row_id, prompt.format(**fmt_d)
