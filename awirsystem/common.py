from typing import List, Tuple

import torch
from transformers import BatchEncoding

from nn_wrappers import AutoTransformerModule


class TransformerBase:

    def __init__(self, module: AutoTransformerModule, irrelevant_tokens: Tuple = (".", ",", "?", "!", "the")):
        self.transformer = module
        self.irrelevant_token_ids = torch.tensor(self.transformer.tokenizer.convert_tokens_to_ids(irrelevant_tokens))

    def _get_inputs(self, text: str) -> BatchEncoding:
        return self.transformer.tokenizer(text, return_tensors='pt', truncation=True)

    def tokenize_text(self, text: str) -> List[str]:
        inputs = self._get_inputs(text)
        return self._tokenize_inputs(inputs)

    def _tokenize_inputs(self, inputs: BatchEncoding) -> List[str]:
        return self.transformer.tokenizer.convert_ids_to_tokens(inputs["input_ids"].flatten().tolist())

    def drop_special_tokens(self, inputs: BatchEncoding, aligned_sequence: List):
        assert inputs["input_ids"].shape[-1] == len(aligned_sequence)
        special_mask = self.transformer.tokenizer.get_special_tokens_mask(inputs["input_ids"].flatten().tolist(),
                                                                          already_has_special_tokens=True)
        irrelevant_mask = (inputs["input_ids"].flatten().unsqueeze(-1) == self.irrelevant_token_ids.to(inputs.input_ids.device)).sum(dim=1)
        return [aligned_sequence[i] for i, mask in enumerate(special_mask) if not mask and not irrelevant_mask[i]]
