from sklearn.metrics.pairwise import cosine_similarity
from numpy import load
import numpy as np
import os
import json
import itertools

def create_ranks(load_path, filename_a, filename_b, save_path, comb):
    embds_a = load(os.path.join(load_path, filename_a))
    embds_b = load(os.path.join(load_path, filename_b))
    distances = cosine_similarity(embds_a, embds_b)

    runs_a = {}
    for q in range(len(distances)):
        runs_a["q" + str(q)] = {}
        for d in range(len(distances[q])):
            runs_a["q" + str(q)]["d" + str(d)] = distances[q][d].astype(float)

    filename_a = comb[0] + "_" + comb[1] + ".json"
    with open(os.path.join(save_path, filename_a), 'w') as fp:
        json.dump(runs_a, fp)

    runs_b = {}
    distances = np.transpose(distances)
    for q in range(len(distances)):
        runs_b["q" + str(q)] = {}
        for d in range(len(distances[q])):
            runs_b["q" + str(q)]["d" + str(d)] = distances[q][d].astype(float)

    filename_b = comb[1] + "_" + comb[0] + ".json"
    with open(os.path.join(save_path, filename_b), 'w') as fp:
        json.dump(runs_b, fp)



load_path = "/work/anlausch/DebunkMLBERT/data/ml_bert_mean_token_embs"
save_path = "/work/anlausch/DebunkMLBERT/data/run_ml_bert_mean_token_embs"

# print("Loading Wikimatrix lang pairs")
# for comb in [("tr", "zh"), ("sv", "zh"), ("ru", "zh"), ("ko", "zh"), ("ja", "zh"), ("it", "zh"), ("hi", "zh"),
#              ("he", "zh"),
#              ("fi", "zh"), ("en", "zh"), ("ar", "zh"), ("eu", "tr"), ("eu", "sv"), ("eu", "ru"), ("eu", "ko"),
#              ("eu", "ja"),
#              ("eu", "it"), ("eu", "hi"), ("eu", "he"), ("eu", "fi"), ("en", "eu"), ("eu", "zh"),
#              ("ar", "eu"), ]:
#
#     # for all combinations, build the file names of the two alignment files
#     filename_a = "wikimatrix_filtered_" + comb[0] + "_" + comb[1] + "." + comb[0] + ".npy"
#     filename_b = "wikimatrix_filtered_" + comb[0] + "_" + comb[1] + "." + comb[1] + ".npy"
#     create_ranks(load_path, filename_a, filename_b, save_path, comb)


for comb in itertools.combinations(["ar", "en", "fi", "he", "hi", "it", "ja", "ko", "ru", "sv", "tr"], 2):
    filename_a = "jw300_filtered_" + comb[0] + "_" + comb[1] + "." + comb[0]  + ".npy"
    filename_b = "jw300_filtered_" + comb[0] + "_" + comb[1] + "." + comb[1]  + ".npy"
    create_ranks(load_path, filename_a, filename_b, save_path, comb)