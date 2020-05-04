import numpy as np
import analysis_utils
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from scipy.stats import pearsonr, spearmanr
import lang2vec.lang2vec as l2v
from sklearn import preprocessing
from itertools import combinations, chain
import pandas as pd
from sklearn.metrics import mean_absolute_error

def powerset(iterable):
    """
    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    not a true powerset as empty is not returned
    :param iterable:
    :return:
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def _get_features(task, features, model, similarity_strategy=None):
    """
    Returns a feature vector for features given a certain task, model and similarity strategy
    :param task:
    :param features:
    :param model:
    :param similarity_strategy:
    :return:
    """
    X = []
    langs = analysis_utils.get_langs_for_task(task)
    for feature in features:
        if feature != "size":
            # this is a nested array
            X_feature = analysis_utils.load_lang2vec_vectors(task=task, features=feature)
            if X_feature is None:
                #continue
                return None
            if similarity_strategy != "-":
                # We start with similarities to english
                X_feature = [[sim] for sim in analysis_utils.compute_similarities_of_lang_vecs(X_feature, strategy=similarity_strategy)]
        elif feature == "size" and model == "xlmr":
            # this is an array, we put it in a list
            X_feature = [[size] for size in analysis_utils.xlmr_input_corpus_sizes(langs)]
        elif feature == "size" and model == "mbert":
            X_feature = [[size] for size in analysis_utils.mbert_input_corpus_sizes(langs)]
        else:
            raise ValueError()
        # we now have a feature vector for a single feature or feature set
        if len(X) == 0:
            X = np.array(X_feature)
        else:
            X = np.concatenate((X,np.array(X_feature)), axis=1)
    if len(X) == 0:
        return None
    return np.array(X, dtype=float)

def _get_labels(task, model, sampling_strategy, k, aggregation_strategy):
    if aggregation_strategy == "plain":
        scores = analysis_utils.get_experiment_scores(task, model, sampling_strategy, k)
    elif aggregation_strategy == "delta" and len(k) == 2:
        scores_a = analysis_utils.get_experiment_scores(task, model, sampling_strategy, k[0])
        scores_b = analysis_utils.get_experiment_scores(task, model, sampling_strategy, k[1])
        scores = [scores_a[i]-scores_b[i] for i in range(len(scores_a))]
    return scores


def _run_regression(task, features, model, sampling_strategy, k, similarity_strategy, aggregation_strategy):
    # TODO: Maybe I need to do feature normalization
    X = _get_features(task, features, model, similarity_strategy)
    if X is not None:
        X = preprocessing.scale(X)
        y = np.array(_get_labels(task, model, sampling_strategy, k, aggregation_strategy))

        assert len(X) == len(y)
        # set up leave-one-out cv
        all_preds = []
        all_ys = []
        coeffs = []
        for train_index, test_index in LeaveOneOut().split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            regressor = LinearRegression() # ElasticNet()
            regressor.fit(X_train, y_train)  # training the algorithm
            coeffs.append(regressor.coef_)
            y_pred = regressor.predict(X_test)
            all_preds.append(y_pred)
            all_ys.append(y_test)

        all_preds = np.array(all_preds).flatten()
        all_ys = np.array(all_ys).flatten()
        avg_coeffs = np.average(np.array(coeffs), axis=0)
        return pearsonr(all_ys, all_preds), spearmanr(all_ys, all_preds), avg_coeffs, mean_absolute_error(all_preds, all_ys)
    else:
        return None


def output_result(result, task, features, model, sampling_strategy, k, aggregation_strategy, similarity_strategy):
    if result is not None:
        pearson = result[0][0]
        pearsonp = result[0][1]
        spearman = result[1][0]
        spearmanp = result[1][1]
        coeffs = result[2]
        mae = result[3]

        if similarity_strategy == "to_en":
            print("\t".join(
                [aggregation_strategy, task, model, sampling_strategy, str(k), str(features), similarity_strategy,
                 str(pearson), str(pearsonp), str(spearman), str(spearmanp), str(coeffs), str(mae)]))
        else:
            print("\t".join(
                [aggregation_strategy, task, model, sampling_strategy, str(k), str(features), similarity_strategy,
                 str(pearson), str(pearsonp), str(spearman), str(spearmanp), str(mae)]))

def run_regression(task, features, model, sampling_strategy, k, aggregation_strategy, similarity_strategy):
    result = _run_regression(task, features, model, sampling_strategy, k, similarity_strategy, aggregation_strategy)
    return result
        #print("\t".join(
        #    [aggregation_strategy, task, model, sampling_strategy, str(k), str(features), similarity_strategy, "", "", "", ""]))


def run_regression_ffs(task, features, model, sampling_strategy, k, aggregation_strategy, similarity_strategy):
    initial_features = features
    best_features = []
    best_result = None
    old_max_pearson = 0.0
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pearson = pd.Series(index=remaining_features)
        results = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            result = run_regression(task, best_features + [new_column],
                                                     model,
                                                     sampling_strategy,
                                                     k,
                                                     aggregation_strategy,
                                                     similarity_strategy)
            if result is not None:
                new_pearson[new_column] = result[0][0]
                results[new_column] = result
            else:
                continue

        max_pearson = new_pearson.max()
        if(max_pearson > old_max_pearson):
            best_features.append(new_pearson.idxmax())
            old_max_pearson = max_pearson
            best_result = results[new_pearson.idxmax()]
        else:
            break
    print("The best features are " + str(best_features))
    output_result(best_result, task, features, model, sampling_strategy, k, aggregation_strategy, similarity_strategy)
    return best_features


if __name__ == "__main__":
    #'syntax_wals', 'syntax_sswl', 'syntax_knn',
    import warnings
    warnings.filterwarnings("ignore")

    all_features = list(l2v.FEATURE_SETS) + ["size"]
    for similarity_strat in ["to_en"]:
        run_regression_ffs("xquad", all_features, "mbert", "k_first", 0, "plain", similarity_strat)
        #for comb in powerset(all_features):
        #    result = run_regression("xnli", comb, "xlmr", "k_first", 0, "plain", similarity_strat) #"-",
    # for ner its RANDOM
    #result = run_correlation_analysis("dep", feature, "mbert?", "LONGEST", 0, "plain", "to_en")
    #result = run_correlation_analysis("xquad", "sizes", "mbert", "k_first", 6, "plain", "-")
    #result = run_correlation_analysis("ner", "sizes", "mbert", "RANDOM", 0, "plain", "-")