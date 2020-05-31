# how many are there for each language?
# how many do we like to preserve for testing
import csv
import os


class XNLIDataReader(object):
    """
    Reads in the Stanford NLI dataset and the MultiGenre NLI dataset
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_rows_for_lang(self, filename, lang, split_index):
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files s1.$data_split.gz,  s2.$data_split.gz,
        labels.$data_split.gz, e.g., for the train split, s1.train.gz, s2.train.gz, labels.train.gz
        """
        rows = []
        with open(os.path.join(self.dataset_folder,filename)) as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for i,row in enumerate(reader):
                #if i == 0:
                #    continue
                if lang == row["language"]:
                    rows.append(row)
        print("Loaded lang data for %s, num sentences %d" % (lang, len(rows)))
        print("Split index is %d" % split_index)
        train_portion = rows[:split_index]
        test_portion = rows[split_index:]
        return train_portion, test_portion


def write_train_and_test_portion():
    output_folder = "./finetune_data/XNLI/XNLI-1.0-new"
    reader = XNLIDataReader("./finetune_data/XNLI/XNLI-1.0-original")
    filename = "xnli.dev.tsv"
    all_train_portions = []
    all_test_portions = []
    for lang in ["fr", "es", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur", "de"]:
        train_portion, test_portion = reader.get_rows_for_lang(filename=filename, lang=lang, split_index=200)
        all_train_portions = all_train_portions + train_portion
        all_test_portions = all_test_portions + test_portion
    print("Total length train portions %d" % len(all_train_portions))
    print("Total length test portions %d" % len(all_test_portions))
    train_file_name = "xnli.test.train.tsv"
    test_file_name = "xnli.test.test.tsv"

    header = ["language", "gold_label", "sentence1_binary_parse", "sentence2_binary_parse",
              "sentence1_parse", "sentence2_parse", "sentence1", "sentence2", "promptID", "pairID",
              "genre", "label1", "label2", "label3", "label4", "label5", "sentence1_tokenized",
              "sentence2_tokenized", "match"]

    with open(os.path.join(output_folder, train_file_name), "w") as f_train:
        train_writer = csv.DictWriter(f_train,
                                      fieldnames=header,
                                      delimiter="\t")
        train_writer.writeheader()
        train_writer.writerows(all_train_portions)

    with open(os.path.join(output_folder, test_file_name), "w") as f_test:
        test_writer = csv.DictWriter(f_test,
                                     fieldnames=header,
                                     delimiter="\t")
        test_writer.writeheader()
        test_writer.writerows(all_test_portions)


def main():
    write_train_and_test_portion()

if __name__ == "__main__":
    main()