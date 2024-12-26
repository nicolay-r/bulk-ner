# bulk-ner 0.25.0 
![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.25.0-orange.svg)
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/ner-service/blob/main/NER_annotation_service.ipynb)
[![twitter](https://img.shields.io/twitter/url/https/shields.io.svg?style=social)](https://x.com/nicolayr_/status/1842300499011260827)
[![PyPI downloads](https://img.shields.io/pypi/dm/bulk-ner.svg)](https://pypistats.org/packages/bulk-ner)

<p align="center">
    <img src="logo.png"/>
</p>

A no-strings inference implementation framework [Named Entity Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition) service of wrapped AI models powered by 
[AREkit](https://github.com/nicolay-r/AREkit) and the related [text-processing pipelines](https://github.com/nicolay-r/AREkit/wiki/Pipelines:-Text-Processing).

The key benefits of this tiny framework are as follows:
1. ☑️ Native support of batching;
2. ☑️ Native long-input contexts handling.

# Installation

```bash
pip install bulk-ner==0.25.0
```

# Usage

## API

Please take a look at the [**related Wiki page**](https://github.com/nicolay-r/bulk-ner/wiki)

## Shell

> **NOTE:** You have to install `source-iter` package

This is an example for using `DeepPavlov==1.3.0` as an adapter for NER models passed via `--adapter` parameter:

```bash
python -m bulk_ner.annotate \
    --src "test/data/test.tsv" \
    --prompt "{text}" \
    --batch-size 10 \
    --adapter "dynamic:models/dp_130.py:DeepPavlovNER" \
    --output "test-annotated.jsonl" \
    %% \
    --model "ner_ontonotes_bert_mult"
```

You can choose the other models via `--model` parameter.

List of the supported models is available here: 
https://docs.deeppavlov.ai/en/master/features/models/NER.html

## Deploy your model

> **Quick example**: Check out the [default DeepPavlov wrapper implementation](/models/dp_130.py)

All you have to do is to implement the `BaseNER` class that has the following protected method:
* `_forward(sequences)` -- expected to return two lists of the same length:
    * `terms` -- related to the list of atomic elements of the text (usually words)
    * `labels` -- B-I-O labels for each term.
  

## Powered by

The pipeline construction components were taken from AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="https://github.com/nicolay-r/ARElight/assets/14871187/01232f7a-970f-416c-b7a4-1cda48506afe"/></a>
</p>
