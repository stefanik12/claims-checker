import abc
import torch
from typing import List, Tuple

from nn_wrappers import PreTrainedAttentionWrapper, AutoTransformerModule


class Embedding(abc.ABC):

    @abc.abstractmethod
    def tokenize_text(self, text: str) -> List[str]:
        raise NotImplemented()

    @abc.abstractmethod
    def tokenize_embed_text(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Main method of every Weighting. Splits the input text into method-specific textual units
        and returns such units aligned with their assigned weights
        :param text: str
        :return: Tuple of (tokens, weights)
        """
        raise NotImplemented()


class TransformerEmbedding(Embedding):

    wrapper = None

    def __init__(self, transformer: AutoTransformerModule, embed_strategy: str = "last-4"):
        self.transformer = transformer
        self.embed_strategy = embed_strategy

    def _get_inputs(self, text: str) -> torch.Tensor:
        return self.wrapper.module.tokenizer.encode_plus(text, return_tensors='pt')

    def _tokenize_inputs(self, inputs: torch.Tensor):
        return self.wrapper.module.tokenizer.convert_ids_to_tokens(inputs["input_ids"].flatten().tolist())

    def tokenize_text(self, text: str) -> List[str]:
        inputs = self._get_inputs(text)
        return self._tokenize_inputs(inputs)

    def tokenize_embed_text(self, text: str) -> Tuple[List[str], List[List[float]]]:
        inputs = self._get_inputs(text)
        return self._tokenize_inputs(inputs), self.embed_inputs(inputs)

    def _embedings_from_outputs(self, outputs: torch.Tensor):
        hidden_states = outputs[2]
        if "last" in self.embed_strategy:
            try:
                how_many = int(self.embed_strategy.split("-")[-1])
                return torch.cat([hidden_states[i] for i in range(-how_many, 0)], 2)
            except (IndexError, ValueError):
                ValueError("Invalid embed_strategy: %s" % self.embed_strategy)
        else:
            raise ValueError("Invalid embed_strategy: %s" % self.embed_strategy)

    def embed_inputs(self, inputs: torch.Tensor) -> List[List[float]]:
        outputs = self.transformer.model(inputs)
        embeddings_t = self._embedings_from_outputs(outputs)
        return embeddings_t.detach().cpu().tolist()
