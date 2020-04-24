"""
This example loads the pre-trained bert-base-nli-mean-tokens models from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader
from xnli_reader import XNLIDataReader
import logging
from datetime import datetime


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
model_load_path = '/home/anlausch/.cache/torch/sentence_transformers/training_nli_bert-base-multilingual-cased-mean-2020-01-20_15-15-48'
model_name = 'nli_bert-base-multilingual-cased-mean'
train_batch_size = 16
num_epochs=1
lang = 'ar'

nli_reader = XNLIDataReader('./finetune_data/XNLI-1.0')
sts_reader = STSDataReader('./finetune_data/stsbenchmark')
train_num_labels = nli_reader.get_num_labels()
model_save_path = '/home/anlausch/.cache/torch/sentence_transformers/training_xnli_' + lang + 'continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_load_path)

# Convert the dataset to a DataLoader ready for training
logging.info("Read XNLI dev dataset")
train_data = SentencesDataset(nli_reader.get_examples('xnli.dev.tsv', lang), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)



logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)


warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

model.evaluate(evaluator)