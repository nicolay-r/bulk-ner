from tqdm import tqdm


class PandasService(object):
    
    @staticmethod
    def iter_rows_as_dict(df):
        for _, data in tqdm(df.iterrows(), total=len(df)):
            yield data.to_dict()
