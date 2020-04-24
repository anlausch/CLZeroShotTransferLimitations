import numpy as np
from numpy import load
import os
import itertools

def mat_normalize(mat, norm_order=2, axis=1):
  return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])


def eval(mat1, mat2):
    print("Evaluating!")
    print("Normalizing matrices...")
    mat1_norm = mat_normalize(mat1, norm_order=2, axis=1)
    mat2_norm = mat_normalize(mat2, norm_order=2, axis=1)

    print("Computing cosines...")
    cosines = np.matmul(mat1_norm, np.transpose(mat2_norm))
    print("Cosines.shape: " + str(cosines.shape))
    ranked = np.flip(np.argsort(cosines, axis=1), axis=1)

    print("Getting ranks...")
    ranks = []
    for i in range(mat1_norm.shape[0]):
        if i % 100 == 0:
            print(i)
        rank = np.where(ranked[i] == i)[0]
        ranks.append(rank)

    mrr = np.sum([1.0 / (r + 1) for r in ranks]) / len(ranks)
    return mrr


def create_ranks(load_path, filename_a, filename_b, save_path, comb):
    try:
        embds_a = load(os.path.join(load_path, filename_a))
        embds_b = load(os.path.join(load_path, filename_b))
        mrr = eval(embds_a, embds_b)
        print(" ".join([str(comb[0]), str(comb[1]), "MAP", str(mrr), "\n"]))
    except Exception as e:
        print(e)


#for load_path in ["/work/anlausch/DebunkMLBERT/data/ml_bert_mean_token_embs", "/work/anlausch/DebunkMLBERT/data/ml_bert_allnli_mean_token_embs","/work/anlausch/DebunkMLBERT/data/ml_bert_allnli_stsb_mean_token_embs"]:
#for load_path in ["/work/anlausch/DebunkMLBERT/data/ml_bert_allnli_xnli_mean_token_embs",
#                  "/work/anlausch/DebunkMLBERT/data/ml_bert_allnli_xnli_ar_mean_token_embs",]:
#for load_path in ["/work/anlausch/DebunkMLBERT/data/ml_bert_xnli_mean_token_embs"]:
for load_path in ["/work/anlausch/DebunkMLBERT/data/ml_bert_xnli_crosslingual_rand_50_mean_token_embs"]:
    print(load_path)
    save_path = "blabla"

    for comb in [("tr", "zh"), ("sv", "zh"), ("ru", "zh"), ("ko", "zh"), ("ja", "zh"), ("it", "zh"), ("hi", "zh"),
                 ("he", "zh"),
                 ("fi", "zh"), ("en", "zh"), ("ar", "zh"), ("eu", "tr"), ("eu", "sv"), ("eu", "ru"), ("eu", "ko"),
                 ("eu", "ja"),
                 ("eu", "it"), ("eu", "hi"), ("eu", "he"), ("eu", "fi"), ("en", "eu"), ("eu", "zh"),
                 ("ar", "eu"), ]:

        # for all combinations, build the file names of the two alignment files
        filename_a = "wikimatrix_filtered_" + comb[0] + "_" + comb[1] + "." + comb[0] + ".npy"
        filename_b = "wikimatrix_filtered_" + comb[0] + "_" + comb[1] + "." + comb[1] + ".npy"
        create_ranks(load_path, filename_a, filename_b, save_path, comb)


    for comb in itertools.combinations(["ar", "en", "fi", "he", "hi", "it", "ja", "ko", "ru", "sv", "tr"], 2):
        filename_a = "jw300_filtered_" + comb[0] + "_" + comb[1] + "." + comb[0] + ".npy"
        filename_b = "jw300_filtered_" + comb[0] + "_" + comb[1] + "." + comb[1] + ".npy"
        create_ranks(load_path, filename_a, filename_b, save_path, comb)