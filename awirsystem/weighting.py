import abc
import torch
from typing import List, Tuple

from nn_wrappers import PreTrainedAttentionWrapper, AutoTransformerModule


class Weighting(abc.ABC):

    @abc.abstractmethod
    def tokenize_text(self, text: str) -> List[str]:
        raise NotImplemented()

    @abc.abstractmethod
    def tokenize_weight_text(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Main method of every Weighting. Splits the input text into method-specific textual units
        and returns such units aligned with their assigned weights
        :param text: str
        :return: Tuple of (tokens, weights)
        """
        raise NotImplemented()


class AttentionWeighting(Weighting, abc.ABC):

    wrapper = None

    def __init__(self, attention_module: AutoTransformerModule):
        self.wrapper = PreTrainedAttentionWrapper(attention_module)

    def _get_inputs(self, text: str) -> torch.Tensor:
        return self.wrapper.module.tokenizer.encode_plus(text, return_tensors='pt')

    def _tokenize_inputs(self, inputs: torch.Tensor):
        return self.wrapper.module.tokenizer.convert_ids_to_tokens(inputs["input_ids"].flatten().tolist())

    def tokenize_text(self, text: str) -> List[str]:
        inputs = self._get_inputs(text)
        return self._tokenize_inputs(inputs)

    def tokenize_weight_text(self, text: str) -> Tuple[List[str], List[float]]:
        inputs = self._get_inputs(text)
        tokenized = self._tokenize_inputs(inputs)
        weights = self.weight_inputs(inputs)
        assert len(tokenized) == len(weights)

        return tokenized, weights

    @abc.abstractmethod
    def weight_inputs(self, inputs: torch.Tensor) -> List[float]:
        raise NotImplemented()


class SelectedHeadsAttentionWeighting(AttentionWeighting):

    def __init__(self, attention_module: AutoTransformerModule,
                 heads_subset: List[Tuple[int, int]],
                 aggregation_strategy: str):
        super().__init__(attention_module)

        self.heads = heads_subset
        self.agg_strategy = aggregation_strategy

    def weight_inputs(self, inputs: torch.Tensor) -> List[float]:
        all_attentions = self.wrapper.module(inputs)
        weights = self._aggregate_attentions(all_attentions, self.heads, self.agg_strategy)
        return weights.detach().cpu().numpy().tolist()

    def _aggregate_attentions(self, attentions: torch.Tensor,
                              heads: List[Tuple[int, int]],
                              strategy: str) -> torch.Tensor:
        """
        :param attentions: matrices of attentions for selected heads
        :param strategy: {mean, sum, per-layer-mean+sum, per-layer-sum+mean}
        :return: tensor representations of weights
        """

        # TODO: penultimate layer works well for translation referencing
        if "per-layer" in strategy:
            layers = [lh[0] for lh in heads]
            per_layer_heads = {l: [h for l_i, h in heads if l == l_i] for l in layers}
            per_layer_strategy = "mean" if "mean" in strategy.split("+")[0] else "sum"
            per_layer_aggs = dict()
            for l_i, l_heads in per_layer_heads:
                per_layer_aggs[l_i] = self._aggregate_attentions(attentions,
                                                                 heads=[(l_i, lhead) for lhead in l_heads],
                                                                 strategy=per_layer_strategy)
            heads_cat = torch.cat(list(per_layer_aggs.values()), dim=-1)
            out_strategy = "mean" if "mean" in strategy.split("+")[1] else "sum"
            # from here, we get the same format as when concatenating all the heads directly
        else:
            heads_cat = torch.cat([attentions[l_i, h_i] for l_i, h_i in heads], dim=-1)
            out_strategy = strategy
        if out_strategy == "sum":
            return heads_cat.sum(dim=-1)
        elif out_strategy == "mean":
            return heads_cat.mean(dim=-1)
        else:
            raise ValueError("unknown strategy: %s " % out_strategy)
