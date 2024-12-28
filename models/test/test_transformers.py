import time
import unittest

from tqdm import tqdm

from models.transformers_4_24_0 import annotate_ner_ppl, annotate_ner, init_token_classification_model


class TestTransformersNERPipeline(unittest.TestCase):

    def test_transformers_batch(self):

        sentences = ["This is America and this is Europe"]

        model, tokenizer = init_token_classification_model(model_path="dslim/bert-base-NER", device="cpu")

        print("Sentences: {}".format(len(sentences)))

        # E1.
        start = time.time()
        for s in tqdm(sentences):
            annotate_ner(model=model, tokenizer=tokenizer, text=s)
        end = time.time()
        print(end - start)

        # E2.
        batch_size = 8
        start = time.time()
        ppl = annotate_ner_ppl(model=model, tokenizer=tokenizer, batch_size=batch_size)
        for i in tqdm(range(0, len(sentences), batch_size)):
            b = sentences[i:i + batch_size]
            if len(b) != batch_size:
                b += [""] * (batch_size - len(b))
            ppl(b)
        end = time.time()
        print(end - start)

    model_names = [
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        "dslim/bert-base-NER",
        "Babelscape/wikineural-multilingual-ner"
    ]

    def test(self):
        text = "My name is Sylvain, and I work at Hugging Face in Brooklyn."
        model, tokenizer = init_token_classification_model(model_path=self.model_names[0], device="cpu")
        results = annotate_ner(model=model, tokenizer=tokenizer, text=text, device="cpu")
        print(results)

    def test_pipeline(self):
        text = "My name is Sylvain, and I work at Hugging Face in Brooklyn."
        model, tokenizer = init_token_classification_model(model_path=self.model_names[0], device="cpu")
        content = [text, text]
        ppl = annotate_ner_ppl(model=model, tokenizer=tokenizer, batch_size=len(content), device="cpu")
        results = ppl(content)
        print(results)