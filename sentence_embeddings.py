"""
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import os
from numpy import save
import itertools

def read_lines_from_file(path, filename):
    with open(os.path.join(path, filename), "r") as f:
        return list(f.readlines())


def load_embd_save(load_path, save_path, filename, model):
    lines = read_lines_from_file(load_path, filename)
    sentence_embeddings = model.encode(lines)
    filename = filename + ".npy"
    save(os.path.join(save_path, filename), sentence_embeddings)


#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
# Load Sentence model (based on BERT) from URL
#model = SentenceTransformer('/home/anlausch/.cache/torch/sentence_transformers/bert-base-multilingual-cased-mean-tokens')
#model = SentenceTransformer('/home/anlausch/.cache/torch/sentence_transformers/training_xnli_ar_continue_training-nli_bert-base-multilingual-cased-mean-2020-01-24_11-45-37')
model = SentenceTransformer('/home/anlausch/.cache/torch/sentence_transformers/training_xnli_bert-base-multilingual-cased-2020-01-24_15-14-51')
save_path = "/work/anlausch/DebunkMLBERT/data/ml_bert_xnli_mean_token_embs"

wikimatrix_path = "/work/anlausch/DebunkMLBERT/data/Wikimatrix/filtered"
print("Loading Wikimatrix files")
for comb in [("tr", "zh"), ("sv", "zh"), ("ru", "zh"), ("ko", "zh"), ("ja", "zh"), ("it", "zh"), ("hi", "zh"),
             ("he", "zh"),
             ("fi", "zh"), ("en", "zh"), ("ar", "zh"), ("eu", "tr"), ("eu", "sv"), ("eu", "ru"), ("eu", "ko"),
             ("eu", "ja"),
             ("eu", "it"), ("eu", "hi"), ("eu", "he"), ("eu", "fi"), ("en", "eu"), ("eu", "zh"),
             ("ar", "eu"), ]:
    # for all combinations, build the file names of the two alignment files
    filename_a = "wikimatrix_filtered_" + comb[0] + "_" + comb[1] + "." + comb[0]
    filename_b = "wikimatrix_filtered_" + comb[0] + "_" + comb[1] + "." + comb[1]

    load_embd_save(wikimatrix_path, save_path, filename_a, model)
    load_embd_save(wikimatrix_path, save_path, filename_b, model)


jw300_path = "/work/anlausch/DebunkMLBERT/data/JW300/filtered"
for comb in itertools.combinations(["ar", "en", "fi", "he", "hi", "it", "ja", "ko", "ru", "sv", "tr"], 2):
    filename_a = "jw300_filtered_" + comb[0] + "_" + comb[1] + "." + comb[0]
    filename_b = "jw300_filtered_" + comb[0] + "_" + comb[1] + "." + comb[1]
    load_embd_save(jw300_path, save_path, filename_a, model)
    load_embd_save(jw300_path, save_path, filename_b, model)