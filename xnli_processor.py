# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" XNLI utils (dataset loading and evaluation) """


import logging
import os
import numpy as np
import random

from transformers import DataProcessor, InputExample


logger = logging.getLogger(__name__)


class XnliProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self, language, train_language=None):
        self.language = language
        self.train_language = train_language

    def get_train_examples(self, data_dir, num_examples=None, sampling_strategy="k_first", tokenizer=None):
        """See base class."""
        lg = self.language if self.train_language is None else self.train_language
        if lg == "en":
            lines = self._read_tsv(os.path.join(data_dir, "XNLI-MT-1.0/multinli/multinli.train.{}.tsv".format(lg)))
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % ("train", i)
                text_a = line[0]
                text_b = line[1]
                label = "contradiction" if line[2] == "contradictory" else line[2]
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        else:
            lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.dev.tsv"))
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                language = line[0]
                if language != self.language:
                    continue
                guid = "%s-%s" % ("dev", i)
                text_a = line[6]
                text_b = line[7]
                label = line[1]
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if num_examples:
            if sampling_strategy == "k_first":
                random.seed(0)
                random.shuffle(examples)
                examples = examples[:num_examples]
            elif sampling_strategy == "k_unique_wp":
                print("Strategy is k_unique_wp")
                example_lengths = [len(list(set(tokenizer.encode(example.text_a)
                                       + tokenizer.encode(example.text_b)))) for example in examples]
                print("Total number of examples %d" % len(examples))
                # this is now the length sorted indices starting from the smallest ones
                # get last k
                example_indices = np.argsort(example_lengths)[-num_examples:]
                examples = np.array(examples)[example_indices]
                print("Number of examples returned %d " % len(examples))
            elif sampling_strategy == "k_longest":
                print("Strategy is k_longest")
                example_lengths = [len(tokenizer.encode(example.text_a)
                                       + tokenizer.encode(example.text_b)) for example in examples]
                print("Total number of examples %d" % len(examples))
                # this is now the length sorted indices starting from the smallest ones
                # get last k
                example_indices = np.argsort(example_lengths)[-num_examples:]
                print("Smallest example length %d " % example_lengths[example_indices[0]])
                print("Biggest example length %d " % example_lengths[example_indices[-1]])
                examples = np.array(examples)[example_indices]
                print("Number of examples returned %d " % len(examples))
                assert len(examples) == num_examples
            elif sampling_strategy == "k_shortest":
                print("Strategy is k_shortest")
                example_lengths = [len(tokenizer.encode(example.text_a)
                                       + tokenizer.encode(example.text_b)) for example in examples]
                print("Total number of examples %d" % len(examples))
                # this is now the length sorted indices starting from the smallest ones
                # get first k
                example_indices = np.argsort(example_lengths)[:num_examples]
                print("Smallest example length %d " % example_lengths[example_indices[0]])
                print("Biggest example length %d " % example_lengths[example_indices[-1]])
                examples = np.array(examples)[example_indices]
                assert len(examples) == num_examples
                print("Number of examples returned %d " % len(examples))
            else:
                raise ValueError()
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            language = line[0]
            if language != self.language:
                continue
            guid = "%s-%s" % ("test", i)
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    #def get_dev_examples(self, data_dir):
    #    """See base class."""
    #    lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.dev.tsv"))
    #    examples = []
    #    for (i, line) in enumerate(lines):
    #        if i == 0:
    #            continue
    #        language = line[0]
    #        if language != self.language:
    #            continue
    #        guid = "%s-%s" % ("dev", i)
    #        text_a = line[6]
    #        text_b = line[7]
    #        label = line[1]
    #        assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
    #        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    #   return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]


xnli_processors = {
    "xnli": XnliProcessor,
}

xnli_output_modes = {
    "xnli": "classification",
}

xnli_tasks_num_labels = {
    "xnli": 3,
}