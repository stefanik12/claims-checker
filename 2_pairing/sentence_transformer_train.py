import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers.evaluation import SentenceEvaluator


st_model = SentenceTransformer('bert-base-nli-mean-tokens')
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = st_model.encode(sentences)
np.array(sentence_embeddings).shape

from run_glue import load_and_cache_examples, BertTokenizer, convert_examples_to_features, evaluate

args = dict()
args["local_rank"] = -1
args["model_name_or_path"] = 'bert-multilingual-base'
args["max_seq_length"] = 128
args["overwrite_cache"] = True
args["data_dir"] = "/data/misc/cc/2_pairing/glue_data/MRPC"
args["model_type"] = "bert"
args["task_name"] = "mrpc"
args["output_dir"] = "/tmp/mrpc_eval_outputs"
args["per_gpu_eval_batch_size"] = 2
args["n_gpu"] = 1
args["device"] = "cuda"
args["output_mode"] = "classification"

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

train_dataset = load_and_cache_examples(args, "mrpc", tokenizer, evaluate=False)


def split_token_pair(ids_tensor, sep_token_id=102, target_len=128):
    ids_np = ids_tensor.numpy()
    split_idxs = np.argwhere(ids_np == sep_token_id).flatten()
    out = [ids_np[:split_idxs[0]], ids_np[split_idxs[0]+1:split_idxs[1]]]
    out = [np.append(part, np.zeros(target_len-len(part)), axis=0) for part in out]
    return torch.tensor(out)


train_tokens = torch.stack([split_token_pair(train_dataset[i][0]) for i, _ in enumerate(train_dataset)])
train_y = torch.stack([train_dataset[i][-1] for i, _ in enumerate(train_dataset)])

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
train_dataset = TensorDataset(train_tokens, train_y)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=4)


class MrpcEvaluator(SentenceEvaluator):
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.
        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """
        return evaluate(args, model, tokenizer)["f1"]


from torch.nn import CrossEntropyLoss

train_dataset = TensorDataset(train_tokens, train_y)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=4)

st_model.fit(train_objectives=[(train_dataloader, CrossEntropyLoss())], evaluator=MrpcEvaluator())

train_dataloader = DataLoader(train_dataset, batch_size=2)
list(train_dataloader)[0][0]