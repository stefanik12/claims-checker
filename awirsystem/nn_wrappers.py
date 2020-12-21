from typing import List

import torch
import transformers


class AutoTransformerModule(torch.nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model = transformers.PreTrainedModel.from_pretrained(model_name_or_path)
        self.tokenizer = transformers.PreTrainedTokenizer.from_pretrained(model_name_or_path)
        self.config = transformers.PretrainedConfig.from_pretrained(model_name_or_path)


class PreTrainedAttentionWrapper(torch.nn.Module):

    module = None

    def __init__(self, module: AutoTransformerModule):
        super().__init__()
        self.module = module

    def from_pretrained(self, model_name_or_path: str):
        self.module = AutoTransformerModule(model_name_or_path)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        try:
            outputs = self.module.model(inputs, output_attentions=True)
            attentions = outputs[-1].detach().cpu()
            return attentions
        except AttributeError:
            print("%s does not have output_attentions param in forward()" % self.module.model.__class__)
