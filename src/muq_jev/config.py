from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    muq_path: Path
    minilm_path: Path
    taps: tuple[int, ...] = (9, 10, 11, 12)
    audio_dim: int = 1024
    text_dim: int = 384
    hidden: int = 512
    heads: int = 8
    ffn: int = 2048
    cross_layers: int = 4
    mixer_layers: int = 2
    mixer_kernel: int = 5
    dropout: float = 0.1
    max_text_len: int = 64
    sample_rate: int = 24000
    grad_checkpointing: bool = True
    fusion: str = 'v1'
    head_dim: int = 256
    head_hidden: int = 256
    head_attn_layers: int = 0
