import numpy as np

# getting the xnli results
def get_results_xnli():

    for k in [0, 10, 50, 100, 200, 500, 1000]:
        all_accs = []
        for lang in ["fr", "es", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur", "de"]:
            try:
                path = ("/work/anlausch/DebunkMLBERT/data/eval_xlmr_xnli_retrain_%d_k_shortest_%s_3e-5_1.0_3/eval_results_test.txt" % (k, lang))
                with open(path,"r") as f:
                    line = f.readline()

                    all_accs.append(float(line.split("acc = ")[1].strip()))
            except Exception as e:
                print(str(k) + "\t" + "\t".join([str(acc) for acc in all_accs]))
                all_accs = []
                break
        print(str(k) + "\t" + "\t".join([str(acc) for acc in all_accs]))

def get_results_xnli_average_iterations():

    for k in [10, 50, 100, 200, 500, 1000]:
        all_accs = []
        if k==0:
            iterations = [1]
        else:
            iterations = [1, 2, 3, 4, 5]
        for iteration in iterations:
            all_accs_for_iteration = []
            for lang in ["en", "fr", "es", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur", "de"]:
                try:
                    path = (
                    "/work/anlausch/DebunkMLBERT/data/eval_xlmr_xnli_retrain_%d_k_shortest_%s_3e-5_1.0_%d/eval_results_test.txt" % (
                    k, lang, iteration))
                    with open(path, "r") as f:
                        line = f.readline()
                        all_accs_for_iteration.append(float(line.split("acc = ")[1].strip()))
                except Exception as e:
                    #print(str(k) + "\t" + "\t".join([str(acc) for acc in all_accs]))
                    #all_accs = []
                    break
            if len(all_accs_for_iteration) == len(["en", "fr", "es", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur", "de"]):
                np.array(all_accs_for_iteration)
                all_accs.append(all_accs_for_iteration)
        avg_accs = np.average(all_accs, axis=0)
        std_accs = np.std(all_accs, axis=0)
        avg_std_accs = list(zip(*(avg_accs,std_accs)))
        print(str(k) + "\t" + "\t".join([str(acc[0]) + "\t" + str(acc[1]) for acc in avg_std_accs]))

# getting the xquad results
def get_results_xquad_average_iterations():
    for k in [0, 2, 4, 6, 8, 10]:
        all_exacts = []
        all_f1 = []
        if k==0:
            iterations = [1]
        else:
            iterations = [1, 2, 3, 4, 5]
        for iteration in iterations:
            all_exacts_for_iteration = []
            all_f1_for_iteration = []
            for lang in ["en", "zh", "vi", "tr", "th", "ru", "hi", "es", "el", "de", "ar"]:
                try:
                    path = ("/work/anlausch/DebunkMLBERT/data/xquad_eval_xlmr_retrain_%d_%s_2e-5_1.0_%d/eval_results.txt" % (k, lang, iteration))
                    with open(path,"r") as f:
                        for i,line in enumerate(f.readlines()):
                            if i == 7:
                                all_exacts_for_iteration.append(float(line.split("exact = ")[1].strip()))
                            elif i == 8:
                                all_f1_for_iteration.append(float(line.split("f1 = ")[1].strip()))
                except Exception as e:
                    #print(str(k) + "\t" + "\t".join([str(score) for score in all_f1]))
                    #all_exacts = []
                    #all_f1 = []
                    break
            np.array(all_exacts_for_iteration)
            np.array(all_f1_for_iteration)
            all_exacts.append(all_exacts_for_iteration)
            all_f1.append(all_f1_for_iteration)
        avg_exacts = np.average(all_exacts, axis=0)
        std_exacts = np.std(all_exacts, axis=0)
        avg_std_exacts = list(zip(*(avg_exacts,std_exacts)))
        avg_f1 = np.average(all_f1, axis=0)
        std_f1 = np.std(all_f1, axis=0)
        avg_std_f1 = list(zip(*(avg_f1,std_f1)))
        print(str(k) + "\t" + "\t".join([str(acc[0]) + "\t" + str(acc[1]) for acc in avg_std_exacts]))

# getting the xquad results
def get_results_xquad():
    for k in [0, 2, 4, 6, 8, 10]:
        all_exacts = []
        all_f1 = []
        for lang in ["en", "zh", "vi", "tr", "th", "ru", "hi", "es", "el", "de", "ar"]:
            try:
                path = ("/work/anlausch/DebunkMLBERT/data/xquad_eval_retrain_%d_%s_3e-5_1.0_2/eval_results.txt" % (k, lang))
                with open(path,"r") as f:
                    for i,line in enumerate(f.readlines()):
                        if i == 7:
                            all_exacts.append(float(line.split("exact = ")[1].strip()))
                        elif i == 8:
                            all_f1.append(float(line.split("f1 = ")[1].strip()))
            except Exception as e:
                print(str(k) + "\t" + "\t".join([str(score) for score in all_f1]))
                all_exacts = []
                all_f1 = []
                break
        print(str(k) + "\t" + "\t".join([str(score) for score in all_f1]))

def main():
    get_results_xnli_average_iterations()
    #get_results_xquad_average_iterations()

if __name__=="__main__":
    main()