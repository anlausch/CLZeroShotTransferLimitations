import pytrec_eval
import json
import os
import itertools
import pickle

def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)

def compute_map(qrel, run):
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map'})
    result = evaluator.evaluate(run)
    result_list = []
    for key, value in result.items():
        result_list.append(value["map"])
    return pytrec_eval.compute_aggregated_measure('map', result_list)

method_name = "ml_bert_mean_token_embs"
#method_name = "ml_bert_allnli_mean_token_embs"
#method_name = "ml_bert_allnli_stsb_mean_token_embs"
run_load_path = "/work/anlausch/DebunkMLBERT/data/run_" + method_name
qrel_load_path = "/work/anlausch/DebunkMLBERT/data/qrel_small.json"
qrel = load_json(qrel_load_path)
all_results = {}

with open("./data/result_" + method_name +".txt", "w") as f:
    for comb in [#("tr", "zh"), ("sv", "zh"), ("ru", "zh"), ("ko", "zh"), ("ja", "zh"), ("it", "zh"), ("hi", "zh"),
                 #("he", "zh"),
                 #("fi", "zh"), ("en", "zh"), ("ar", "zh"), ("eu", "tr"),
                 #("eu", "sv"), ("eu", "ru"), ("eu", "ko"),
                 #("eu", "ja"),
                 #("eu", "it"), ("eu", "hi"),
                 ("eu", "he"), ("eu", "fi"), ("en", "eu"), ("eu", "zh"),
                 ("ar", "eu"), ]:
        filename_a = comb[0] + "_" + comb[1] + ".json"
        run_a = load_json(os.path.join(run_load_path, filename_a))
        all_results[(comb[0], comb[1])]= compute_map(qrel, run_a)
        print(" ".join([str(comb[0]), str(comb[1]), "MAP", str(all_results[(comb[0], comb[1])]), "\n"]))
        f.write(" ".join([str(comb[0]), str(comb[1]), "MAP", str(all_results[(comb[0], comb[1])]), "\n"]))

        filename_b = comb[1] + "_" + comb[0] + ".json"
        run_b = load_json(os.path.join(run_load_path, filename_b))
        all_results[(comb[1], comb[0])] = compute_map(qrel, run_b)
        print(" ".join([str(comb[1]), str(comb[0]), "MAP", str(all_results[(comb[1], comb[0])]), "\n"]))
        f.write(" ".join([str(comb[1]), str(comb[0]), "MAP", str(all_results[(comb[1], comb[0])]), "\n"]))

    # open a file, where you ant to store the data
    with open("./data/result_" + method_name + "_wikimatrix.p", 'wb') as f_out:
        # dump information to that file
        pickle.dump(all_results, f_out)

    for comb in itertools.combinations(["ar", "en", "fi", "he", "hi", "it", "ja", "ko", "ru", "sv", "tr"], 2):
        filename_a = comb[0] + "_" + comb[1] + ".json"
        run_a = load_json(os.path.join(run_load_path, filename_a))
        all_results[(comb[0], comb[1])]=compute_map(qrel, run_a)
        print(" ".join([str(comb[0]), str(comb[1]), "MAP", str(all_results[(comb[0], comb[1])]), "\n"]))
        f.write(" ".join([str(comb[0]), str(comb[1]), "MAP", str(all_results[(comb[0], comb[1])]), "\n"]))

        filename_b = comb[1] + "_" + comb[0] + ".json"
        run_b = load_json(os.path.join(run_load_path, filename_b))
        all_results[(comb[1], comb[0])] = compute_map(qrel, run_b)
        print(" ".join([str(comb[1]), str(comb[0]), "MAP", str(all_results[(comb[1], comb[0])]), "\n"]))
        f.write(" ".join([str(comb[1]), str(comb[0]), "MAP", str(all_results[(comb[1], comb[0])]), "\n"]))

# open a file, where you ant to store the data
with open("./data/result_" + method_name + ".p", 'wb') as f_out:
    # dump information to that file
    pickle.dump(all_results, f_out)