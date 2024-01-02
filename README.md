# ner-service
Named Entity Recognition (NER) service of wrapped AI models powered by 
[AREkit text-processing pipelines](https://github.com/nicolay-r/AREkit/wiki/Pipelines:-Text-Processing).

# Usage

```bash
pip install -r dependencies.txt
```

## Annotating texts

```bash
python annotate.py --src "data/test.csv" --prompt "{text}" --model "ner_ontonotes_bert"
```

## Powered by

* AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="https://github.com/nicolay-r/ARElight/assets/14871187/01232f7a-970f-416c-b7a4-1cda48506afe"/></a>
</p>
