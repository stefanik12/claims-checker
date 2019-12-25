from sentence_transformers import SentencesDataset, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from transformers.data.processors.glue import MrpcProcessor

processor = MrpcProcessor()

model = SentenceTransformer('bert-base-nli-mean-tokens')
train_examples = processor.get_train_examples("/data/misc/cc/1_extraction/1.1_squad_classification/glue_data/MRPC")
train_examples = [InputExample(ie.guid, [ie.text_a, ie.text_b], float(ie.label)) for ie in train_examples]
train_data = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=2)
train_loss = losses.CosineSimilarityLoss(model=model)

dev_examples = processor.get_dev_examples("/data/misc/cc/1_extraction/1.1_squad_classification/glue_data/MRPC")
dev_examples = [InputExample(ie.guid, [ie.text_a, ie.text_b], float(ie.label)) for ie in dev_examples]
dev_data = SentencesDataset(dev_examples, model)
dev_dataloader = DataLoader(train_data, shuffle=True, batch_size=2)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=5,
          evaluation_steps=1000,
          warmup_steps=500,
          output_path="out")

print("done")
