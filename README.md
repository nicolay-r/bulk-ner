# fast-ner 
![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.25.0-orange.svg)
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/ner-service/blob/main/NER_annotation_service.ipynb)

A no-strings inference implementation framework [Named Entity Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition) service of wrapped AI models powered by 
[AREkit](https://github.com/nicolay-r/AREkit) and the related [text-processing pipelines](https://github.com/nicolay-r/AREkit/wiki/Pipelines:-Text-Processing).

> ⚠️ **Limitation:** at present this framework has an embedded support of [DeepPavlov](https://github.com/deeppavlov/DeepPavlov) NER models

The key benefits of this tiny framework are as follows:
1. ☑️ Native support of batching;
2. ☑️ Native long-input contexts handling.

# Installation

```bash
pip install -r dependencies.txt
```

# Usage

```bash
python annotate.py --src "data/test.csv" --prompt "{text}" --model "ner_ontonotes_bert" --batch-size 10
```

List of the supported models is available here: https://docs.deeppavlov.ai/en/master/features/models/NER.html

## Powered by

* AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="https://github.com/nicolay-r/ARElight/assets/14871187/01232f7a-970f-416c-b7a4-1cda48506afe"/></a>
</p>
