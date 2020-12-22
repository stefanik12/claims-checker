import abc
import torch
from typing import List, Tuple

from transformers import BatchEncoding

from common import TransformerBase
from nn_wrappers import PreTrainedAttentionWrapper, AutoTransformerModule


class Embedding(abc.ABC):

    @abc.abstractmethod
    def tokenize_text(self, text: str) -> List[str]:
        raise NotImplemented()

    @abc.abstractmethod
    def tokenize_embed_text(self, text: str) -> Tuple[List[str], List[List[float]]]:
        """
        Main method of every Weighting. Splits the input text into method-specific textual units
        and returns such units aligned with their assigned weights
        :param text: str
        :return: Tuple of (tokens, weights)
        """
        raise NotImplemented()


class TransformerEmbedding(TransformerBase, Embedding):

    def __init__(self, transformer: AutoTransformerModule, embed_strategy: str = "last-4"):
        super().__init__(transformer)
        self.embed_strategy = embed_strategy

    def tokenize_embed_text(self, text: str, get_special_tokens: bool = False) -> Tuple[List[str], List[List[float]]]:
        inputs = self._get_inputs(text)
        tokens = self._tokenize_inputs(inputs)
        embeddings = self.embed_inputs(inputs)[0]
        if not get_special_tokens:
            tokens = self.drop_special_tokens(inputs, tokens)
            embeddings = self.drop_special_tokens(inputs, embeddings)

        return tokens, embeddings

    def _embedings_from_outputs(self, hidden_states: torch.Tensor):
        if "last" in self.embed_strategy:
            try:
                how_many = int(self.embed_strategy.split("-")[-1])
                return torch.cat([hidden_states[i] for i in range(-how_many, 0)], -1)
            except (IndexError, ValueError):
                ValueError("Invalid embed_strategy: %s" % self.embed_strategy)
        else:
            raise ValueError("Invalid embed_strategy: %s" % self.embed_strategy)

    def embed_inputs(self, inputs: BatchEncoding) -> List[List[List[float]]]:
        device = self.transformer.model.device
        outputs = self.transformer.model(**inputs.to(device), output_hidden_states=True)[-1]
        embeddings_t = self._embedings_from_outputs(outputs)
        return embeddings_t.detach().cpu().tolist()
