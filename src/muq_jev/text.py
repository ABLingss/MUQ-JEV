from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

class TextTokenizer:

    def __init__(self, path: str | Path, max_length: int=64):
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.max_length = int(max_length)

    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        return (enc['input_ids'].long(), enc['attention_mask'].long())

class TextTower(nn.Module):

    def __init__(self, path: str | Path):
        super().__init__()
        self.model = AutoModel.from_config(AutoConfig.from_pretrained(str(path), local_files_only=True))
        self.dim = int(self.model.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-06)
        return F.normalize(pooled, dim=-1)
