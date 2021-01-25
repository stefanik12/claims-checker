import abc
import torch
from typing import List, Tuple

from transformers import BatchEncoding

from common import TransformerBase
from nn_wrappers import AutoTransformerModule


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


class AttentionWeighting(TransformerBase, Weighting, abc.ABC):

    def __init__(self, module: AutoTransformerModule):
        super().__init__(module)

    def tokenize_weight_text(self, text: str, get_special_tokens: bool = False) -> Tuple[List[str], List[float]]:
        inputs = self._get_inputs(text)
        tokenized = self._tokenize_inputs(inputs)
        weights = self.weight_inputs(inputs)[0]

        if not get_special_tokens:
            tokenized = self.drop_special_tokens(inputs, tokenized)
            weights = self.drop_special_tokens(inputs, weights)

        assert len(tokenized) == len(weights)
        return tokenized, weights

    @abc.abstractmethod
    def weight_inputs(self, inputs: BatchEncoding) -> List[List[float]]:
        raise NotImplemented()


class SelectedHeadsAttentionWeighting(AttentionWeighting):

    def __init__(self, attention_module: AutoTransformerModule,
                 heads_subset: List[Tuple[int, int]],
                 aggregation_strategy: str,
                 normalize_weights: bool = True):
        super().__init__(attention_module)

        self.heads = heads_subset
        self.agg_strategy = aggregation_strategy
        self.normalize_weights = normalize_weights

    def weight_inputs(self, inputs: BatchEncoding) -> List[float]:
        device = self.transformer.model.device
        all_attentions = self.transformer.model(**inputs.to(device), output_attentions=True)[-1]
        weights = self._aggregate_attentions(all_attentions, self.heads, self.agg_strategy).detach().cpu()
        if self.normalize_weights:
            norm = torch.linalg.norm(weights, ord=2)
            weights = torch.tensor(weights) / torch.max(norm, 1e-10 * torch.ones_like(norm))

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
            for l_i, l_heads in per_layer_heads.items():
                per_layer_aggs[l_i] = self._aggregate_attentions(attentions,
                                                                 heads=[(l_i, lhead) for lhead in l_heads],
                                                                 strategy=per_layer_strategy)
            heads_cat = torch.cat(list(per_layer_aggs.values()), dim=0) .unsqueeze(0)
            # from here, we get the same format as when concatenating all the heads directly
            out_strategy = "mean" if "mean" in strategy.split("+")[1] else "sum"
        else:
            heads_cat = torch.cat([attentions[l_i][:, h_i] for l_i, h_i in heads], dim=0)
            out_strategy = strategy
        if out_strategy == "sum":
            return heads_cat.sum(dim=-2)
        elif out_strategy == "mean":
            return heads_cat.mean(dim=-2)
        else:
            raise ValueError("unknown strategy: %s " % out_strategy)



