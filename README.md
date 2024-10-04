# bulk-ner 
![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.25.0-orange.svg)
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/ner-service/blob/main/NER_annotation_service.ipynb)

A no-strings inference implementation framework [Named Entity Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition) service of wrapped AI models powered by 
[AREkit](https://github.com/nicolay-r/AREkit) and the related [text-processing pipelines](https://github.com/nicolay-r/AREkit/wiki/Pipelines:-Text-Processing).

The key benefits of this tiny framework are as follows:
1. ☑️ Native support of batching;
2. ☑️ Native long-input contexts handling.

# Installation

```bash
pip install git+https://github.com/nicolay-r/bulk-ner@main
```

# Usage

This is an example for using `DeepPavlov==1.3.0` as an adapter for NER models:

```bash
python -m bulk_ner.annotate \
    --src "test/data/test.csv" \
    --prompt "{text}" \
    --batch-size 10 \
    --adapter "dynamic:models/dp_130.py:DeepPavlovNER" \
    %% \
    --model "ner_ontonotes_bert_mult"
```

List of the supported models is available here: https://docs.deeppavlov.ai/en/master/features/models/NER.html

## Powered by

* AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="https://github.com/nicolay-r/ARElight/assets/14871187/01232f7a-970f-416c-b7a4-1cda48506afe"/></a>
</p>
