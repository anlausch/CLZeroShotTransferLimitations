import lang2vec.lang2vec as l2v
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import csv
import analysis_utils

def compute_correlation(task, features, model, sampling_strategy, k, similarity_strategy, aggregation_strategy):
    if features != "sizes":
        vecs_matrix = analysis_utils.load_lang2vec_vectors(task, features)
        if vecs_matrix is None:
            return None
        ### make sure that the langs are always in the same order
        # TODO: Which similarities to take? Similarity from english? Avg similarity to the others? :) many possibilities
        # We start with similarities to english
        similarities = analysis_utils.compute_similarities_of_lang_vecs(vecs_matrix, strategy=similarity_strategy)
    else:
        langs = analysis_utils.get_langs_for_task(task)
        if model == "xlmr":
            similarities = analysis_utils.xlmr_input_corpus_sizes(langs)
        elif model == "mbert":
            similarities = analysis_utils.mbert_input_corpus_sizes(langs)

    if aggregation_strategy == "plain":
        scores = analysis_utils.get_experiment_scores(task, model, sampling_strategy, k)
    elif aggregation_strategy == "delta" and len(k) == 2:
        scores_a = analysis_utils.get_experiment_scores(task, model, sampling_strategy, k[0])
        scores_b = analysis_utils.get_experiment_scores(task, model, sampling_strategy, k[1])
        scores = [scores_a[i]-scores_b[i] for i in range(len(scores_a))]
    return pearsonr(similarities, scores), spearmanr(similarities, scores)


def compute_correlation_sizes_against_l2v(task, features, model, similarity_strategy):
    vecs_matrix = analysis_utils.load_lang2vec_vectors(task, features)
    if vecs_matrix is None:
        return None
    similarities = analysis_utils.compute_similarities_of_lang_vecs(vecs_matrix, strategy=similarity_strategy)
    langs = analysis_utils.get_langs_for_task(task)
    if model == "xlmr":
        sizes = analysis_utils.xlmr_input_corpus_sizes(langs)
    elif model == "mbert":
        sizes = analysis_utils.mbert_input_corpus_sizes(langs)
    return pearsonr(similarities, sizes), spearmanr(similarities, sizes)


def run_correlation_analysis(mode, task, features, model, sampling_strategy, k, aggregation_strategy, similarity_strategy):
    if mode == "normal":
        result = compute_correlation(task, features, model, sampling_strategy, k, similarity_strategy, aggregation_strategy)
    if mode == "sizesvsvecs":
        result = compute_correlation_sizes_against_l2v(task, features, model, similarity_strategy)
        aggregation_strategy=""
        k=""
        sampling_strategy=""

    if result is not None:
        pearson = result[0][0]
        pearsonp = result[0][1]
        spearman = result[1][0]
        spearmanp = result[1][1]
        print("\t".join([aggregation_strategy, task, model, sampling_strategy, str(k), str(features), similarity_strategy, str(pearson),str(pearsonp),str(spearman),str(spearmanp)]))
    else:
        return
        #print("\t".join(
        #    [aggregation_strategy, task, model, sampling_strategy, str(k), str(features), similarity_strategy, "", "", "", ""]))

if __name__ == "__main__":
    for feature in l2v.FEATURE_SETS:
        #result = run_correlation_analysis("xquad", feature, "mbert", "-", 0, "plain", "to_en")
        # for ner its RANDOM
        for task in ["xnli","xquad", "dep", "pos", "ner"]:
            result = run_correlation_analysis(mode="sizesvsvecs", task=task, features=feature, model="mbert", similarity_strategy="to_en",
                                              k=None, aggregation_strategy=None, sampling_strategy=None)
            if task in ["xnli", "xquad"]:
                result = run_correlation_analysis(mode="sizesvsvecs", task=task, features=feature, model="xlmr", similarity_strategy="to_en",
                                              k=None, aggregation_strategy=None, sampling_strategy=None)
        #result = run_correlation_analysis("dep", feature, "mbert?", "LONGEST", 0, "plain", "to_en")
    #result = run_correlation_analysis("xquad", "sizes", "mbert", "k_first", 6, "plain", "-")
    #result = run_correlation_analysis("ner", "sizes", "mbert", "RANDOM", 0, "plain", "-")