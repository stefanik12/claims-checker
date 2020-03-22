from sentence_transformers import SentencesDataset, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader
from torch.utils.data import DataLoader

# model = SentenceTransformer('bert-base-nli-mean-tokens')
sts_reader = STSDataReader('stsbenchmark_data', normalize_scores=True)
train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

print(train_data)
