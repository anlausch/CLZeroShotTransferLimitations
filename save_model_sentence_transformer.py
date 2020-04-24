from sentence_transformers import models, SentenceTransformer

# Use CamemBERT for mapping tokens to embeddings
word_embedding_model = models.BERT('bert-base-multilingual-cased')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#Save model
model.save('/home/anlausch/.cache/torch/sentence_transformers/bert-base-multilingual-cased-mean-tokens')

#Load from disc
model_loaded = SentenceTransformer('/home/anlausch/.cache/torch/sentence_transformers/bert-base-multilingual-cased-mean-tokens')