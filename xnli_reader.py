import csv
import os
from typing import Union, List

### TODO: Anne Lauscher changed the code a bit to also be able to produce cl pairs
import random

class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str, texts: List[str], label: Union[int, float]):
        """
        Creates one InputExample with the given texts, guid and label

        str.strip() is called on both texts.

        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.label = label

class XNLIDataReader(object):
    """
    Reads in the Stanford NLI dataset and the MultiGenre NLI dataset
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_examples(self, filename, lang, max_examples=0):
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files s1.$data_split.gz,  s2.$data_split.gz,
        labels.$data_split.gz, e.g., for the train split, s1.train.gz, s2.train.gz, labels.train.gz
        """
        s1, s2, labels = [], [], []
        with open(os.path.join(self.dataset_folder,filename)) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i,row in enumerate(reader):
                if i == 0:
                    continue
                elif not lang:
                    s1.append(row[6])
                    s2.append(row[7])
                    labels.append(row[1])
                elif lang == row[0]:
                    s1.append(row[6])
                    s2.append(row[7])
                    labels.append(row[1])

        if not lang:
            print("Loaded all data, num sentences %d" % len(s1))
        elif lang:
            print("Loaded lang data for %s, num sentences %d" % (lang, len(s1)))

        examples = []
        id = 0
        for sentence_a, sentence_b, label in zip(s1, s2, labels):
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]


class XNLIDataCLPairsReader(object):
    """
    Reads in the Stanford NLI dataset and the MultiGenre NLI dataset
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_examples(self, filename, lang1, lang2, num_examples=0):
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files s1.$data_split.gz,  s2.$data_split.gz,
        labels.$data_split.gz, e.g., for the train split, s1.train.gz, s2.train.gz, labels.train.gz
        """
        candidates_lang1 = []
        candidates_lang2 = []
        s1, s2, labels = [], [], []

        # read in all instances corresponding to our two languages
        with open(os.path.join(self.dataset_folder,filename)) as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for i,row in enumerate(reader):
                if row["language"] == lang1:
                    candidates_lang1.append({"promptid_pairid": row["promptID"] + "_" + row["pairID"], "sentence1": row["sentence1"], "sentence2": row["sentence2"], "lang": row["language"], "label": row["gold_label"]})
                elif row["language"] == lang2:
                    candidates_lang2.append({"promptid_pairid": row["promptID"] + "_" + row["pairID"], "sentence1": row["sentence1"], "sentence2": row["sentence2"], "lang": row["language"], "label": row["gold_label"]})


        # select num_examples random instance ids
        promptpairIDs = list(set([c["promptid_pairid"] for c in candidates_lang1]))
        random.seed(0)
        if len(promptpairIDs) >= num_examples:
            choices = random.choices(promptpairIDs, k=num_examples)

            # construct cl examples from the selection
            #pairs = []
            for c1 in candidates_lang1:
                if c1["promptid_pairid"] in choices:
                    for c2 in candidates_lang2:
                        if c1["promptid_pairid"] == c2["promptid_pairid"]:
                            # we have a match
                            #pair = {"promptid_pairid": c1["promptid_pairid"], "sentence1": c1["sentence1"], "sentence2": c2["sentence2"], "lang1": c1["language"], "lang2": c2["language"], "label": c1["gold_label"]}
                            #pairs.append(pair)
                            if random.random() > 0.5:
                                s1.append(c1["sentence1"])
                                s2.append(c2["sentence2"])
                            else:
                                s1.append(c1["sentence2"])
                                s2.append(c2["sentence1"])
                            labels.append(c1["label"])
                            break
            assert len(s1) == num_examples
            examples = []
            id = 0
            for sentence_a, sentence_b, label in zip(s1, s2, labels):
                guid = "%s-%d" % (filename, id)
                id += 1
                examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=self.map_label(label)))
            return examples
        else:
            print("Not enough data")
            return None

    @staticmethod
    def get_labels():
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]