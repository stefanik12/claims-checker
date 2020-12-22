from typing import List

import torch
import transformers


class AutoTransformerModule(torch.nn.Module):
    def __init__(self, model_name_or_path: str, device: str):
        super().__init__()
        self.config = transformers.AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = transformers.AutoModel.from_pretrained(model_name_or_path, config=self.config).to(device)

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)


class PreTrainedAttentionWrapper:
    """Deprecated"""
    module = None

    def __init__(self, module: AutoTransformerModule):
        self.module = module
        raise DeprecationWarning()

    def from_pretrained(self, model_name_or_path: str):
        self.module = AutoTransformerModule(model_name_or_path)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        try:
            outputs = self.module.model(inputs, output_attentions=True)
            attentions = outputs[-1].detach().cpu()
            return attentions
        except AttributeError:
            print("%s does not have output_attentions param in forward()" % self.module.model.__class__)
