# From Zero to Hero

This repository contains the code for the experiments related to higher-level semantic tasks and related to the meta-learning from: "From Zero to Hero: On the Limitations of Zero Shot Cross-Lingual Transfer".
In case of any questions, please reach out to me. The corresponding publication can be found here: https://www.aclweb.org/anthology/2020.emnlp-main.363/ .

```
@inproceedings{lauscher-etal-2020-zero,
    title = "From Zero to Hero: {O}n the Limitations of Zero-Shot Language Transfer with Multilingual {T}ransformers",
    author = "Lauscher, Anne  and
      Ravishankar, Vinit  and
      Vuli{\'c}, Ivan  and
      Glava{\v{s}}, Goran",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.363",
    doi = "10.18653/v1/2020.emnlp-main.363",
    pages = "4483--4499",
    abstract = "Massively multilingual transformers (MMTs) pretrained via language modeling (e.g., mBERT, XLM-R) have become a default paradigm for zero-shot language transfer in NLP, offering unmatched transfer performance. Current evaluations, however, verify their efficacy in transfers (a) to languages with sufficiently large pretraining corpora, and (b) between close languages. In this work, we analyze the limitations of downstream language transfer with MMTs, showing that, much like cross-lingual word embeddings, they are substantially less effective in resource-lean scenarios and for distant languages. Our experiments, encompassing three lower-level tasks (POS tagging, dependency parsing, NER) and two high-level tasks (NLI, QA), empirically correlate transfer performance with linguistic proximity between source and target languages, but also with the size of target language corpora used in MMT pretraining. Most importantly, we demonstrate that the inexpensive few-shot transfer (i.e., additional fine-tuning on a few target-language instances) is surprisingly effective across the board, warranting more research efforts reaching beyond the limiting zero-shot conditions.",
}
```
