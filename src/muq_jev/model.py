from __future__ import annotations
import os
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig
from .text import TextTower
FRAME_HOP = 960

@contextmanager
def _preserve_bn(module: nn.Module):
    saved = [(buf, buf.clone()) for name, buf in module.named_buffers() if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    try:
        yield
    finally:
        with torch.no_grad():
            for buf, value in saved:
                buf.copy_(value)

class MuQBackbone(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from ._muq import MuQ
        import json
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        weight_dir = Path(cfg.muq_path)
        if weight_dir.is_file():
            weight_dir = weight_dir.parent
        self.model = MuQ(json.loads((weight_dir / 'config.json').read_text()))
        self.model.eval()
        self.taps = tuple((int(x) for x in cfg.taps))
        self.layers = 12
        self.hidden_dim = int(self.model.model.conformer.config.hidden_size)
        if self.taps and max(self.taps) > self.layers:
            raise ValueError(f'tap {max(self.taps)} exceeds {self.layers} layers')
        self.checkpointing = bool(cfg.grad_checkpointing)
        self.finetune_layers: int | None = None
        self.set_checkpointing(self.checkpointing)
        self.set_frozen(True)

    def set_checkpointing(self, enabled: bool) -> None:
        self.checkpointing = bool(enabled)
        conv = self.model.model.conv
        if not hasattr(self, '_conv_forward'):
            self._conv_forward = conv.forward

            def conv_forward(*args, **kwargs):
                if self.checkpointing and conv.training and torch.is_grad_enabled() and any((p.requires_grad for p in conv.parameters())):
                    return checkpoint(self._conv_forward, *args, use_reentrant=False, context_fn=lambda: (nullcontext(), _preserve_bn(conv)), **kwargs)
                return self._conv_forward(*args, **kwargs)
            conv.forward = conv_forward
        for layer in self.model.model.conformer.layers:
            if enabled and (not hasattr(layer, 'gradient_checkpointing')):
                raise RuntimeError('MuQ conformer does not support layer checkpointing')
            layer.gradient_checkpointing = bool(enabled)
            layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False, context_fn=lambda layer=layer: (nullcontext(), _preserve_bn(layer)))

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)
        if not frozen:
            self.finetune_layers = None
        for name, parameter in self.model.named_parameters():
            active = name.startswith(('model.conv.', 'model.conformer.'))
            active = active and (not name.startswith('model.conformer.pos_conv_embed.'))
            parameter.requires_grad = bool(not frozen and active)
        self.model.train(not frozen)

    def set_finetune_layers(self, count: int) -> None:
        layers = list(self.model.model.conformer.layers)
        if not 0 < int(count) <= len(layers):
            raise ValueError(f'finetune layer count must be in 1..{len(layers)}')
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for layer in layers[-int(count):]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.frozen = False
        self.finetune_layers = int(count)
        self.train(self.training)

    def train(self, mode: bool=True):
        super().train(mode)
        if getattr(self, 'frozen', False):
            self.model.train(False)
        elif getattr(self, 'finetune_layers', None):
            layers = list(self.model.model.conformer.layers)
            self.model.train(False)
            for layer in layers[-int(self.finetune_layers):]:
                layer.train(mode)
        else:
            self.model.train(mode)
        return self

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor | None=None) -> dict[str, Any]:
        attention = None
        if lengths is not None:
            attention = torch.arange(waveform.shape[1], device=waveform.device)[None] < lengths[:, None]
        output = self.model(waveform, attention_mask=attention, output_hidden_states=True)
        states = output.hidden_states
        if states is None:
            raise RuntimeError('MuQ did not return hidden states')
        taps = [states[index] for index in self.taps]
        return {'taps': taps, 'frames': taps[0].shape[1] if taps else 0}

class GatedTemporalBlock(nn.Module):

    def __init__(self, dim: int, kernel: int, dilation: int):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation, groups=dim)
        self.gate = nn.Conv1d(dim, 2 * dim, 1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x * m
        y = self.dw(x.transpose(1, 2))
        a, b = self.gate(y).chunk(2, dim=1)
        y = (a * torch.sigmoid(b)).transpose(1, 2)
        x = self.norm1(x + y)
        x = x + self.ff(self.norm2(x))
        return x * m

class AudioEncoder(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.backbone = MuQBackbone(cfg)
        self.layer_logits = nn.Parameter(torch.zeros(len(cfg.taps)))
        self.norm = nn.LayerNorm(cfg.audio_dim)
        self.proj = nn.Linear(cfg.audio_dim, cfg.hidden)
        self.mixer = nn.ModuleList([GatedTemporalBlock(cfg.hidden, cfg.mixer_kernel, dilation=2 ** i) for i in range(cfg.mixer_layers)])
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor):
        out = self.backbone(waveform, lengths)
        taps = torch.stack(out['taps'], dim=-1)
        weights = torch.softmax(self.layer_logits, dim=0)
        fused = (taps * weights).sum(dim=-1)
        x = self.proj(self.norm(fused))
        x = self.drop(x)
        frames = x.shape[1]
        frame_lengths = torch.ceil(lengths.float() / FRAME_HOP).long().clamp(min=1, max=frames)
        mask = torch.arange(frames, device=x.device)[None] < frame_lengths[:, None]
        for block in self.mixer:
            x = block(x, mask)
        return (x, mask)

    def set_phase(self, phase: str) -> None:
        if phase == 'a':
            self.backbone.set_frozen(True)
        elif phase == 'b':
            self.backbone.set_finetune_layers(4)
        elif phase == 'c':
            self.backbone.set_frozen(False)
        else:
            raise ValueError(phase)

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q(q).unsqueeze(0)
        kvn = self.norm_kv(kv).unsqueeze(0)
        out, _ = self.attn(qn, kvn, kvn, key_padding_mask=(~kv_mask).unsqueeze(0), need_weights=False)
        q = q + self.drop(out.squeeze(0))
        q = q + self.drop(self.ff(self.norm_ff(q)))
        return q

class CandidateScorer(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(q)).squeeze(-1)

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

class CandidateHeadV2(nn.Module):

    def __init__(self, cfg: ModelConfig, text_dim: int):
        super().__init__()
        self.text_dim = int(text_dim)
        self.head_dim = int(cfg.head_dim)
        self.audio_norm = nn.LayerNorm(cfg.hidden)
        self.audio_proj = nn.Linear(cfg.hidden, self.head_dim, bias=False)
        self.text_norm = nn.LayerNorm(self.text_dim)
        self.text_proj = nn.Linear(self.text_dim, self.head_dim, bias=False)
        self.bias = nn.Sequential(nn.Linear(self.text_dim, cfg.head_hidden), nn.GELU(), nn.Linear(cfg.head_hidden, 1))
        self.attn = nn.ModuleList([CrossAttentionBlock(cfg.hidden, cfg.heads, cfg.ffn, cfg.dropout) for _ in range(int(cfg.head_attn_layers))])
        if self.attn:
            self.attn_score = CandidateScorer(cfg.hidden)
            self.attn_gate = nn.Parameter(torch.tensor(0.05))
        else:
            self.attn_score = None
            self.attn_gate = None
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, audio_vec: torch.Tensor, text: torch.Tensor, spans: list[list[dict[str, Any]]], frames: torch.Tensor | None=None, frame_mask: torch.Tensor | None=None) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        a = self.drop(self.audio_proj(self.audio_norm(audio_vec)))
        t = self.drop(self.text_proj(self.text_norm(text)))
        bias = self.bias(text).squeeze(-1)
        scale = self.head_dim ** (-0.5)
        scores = a * scale @ t.transpose(0, 1)
        logits: list[torch.Tensor] = []
        flat: list[dict[str, Any]] = []
        for b, group_spans in enumerate(spans):
            if not group_spans:
                continue
            for span in group_spans:
                sl = slice(span['start'], span['end'])
                logit = scores[b, sl] + bias[sl]
                if self.attn:
                    q = text[sl]
                    for block in self.attn:
                        q = block(q, frames[b], frame_mask[b])
                    logit = logit + self.attn_gate * self.attn_score(q)
                logits.append(logit)
                flat.append(span)
        return (logits, flat)

class U1Model(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.fusion = str(getattr(cfg, 'fusion', 'v1'))
        if self.fusion not in ('v1', 'v2'):
            raise ValueError(f"unknown fusion layer {self.fusion!r} (want 'v1' or 'v2')")
        self.audio = AudioEncoder(cfg)
        self.text_tower = TextTower(cfg.minilm_path)
        for name, parameter in self.text_tower.model.named_parameters():
            if name.startswith('pooler.'):
                parameter.requires_grad = False
        self.text_proj = nn.Linear(self.text_tower.dim, cfg.hidden)
        if self.fusion == 'v1':
            self.cross = nn.ModuleList([CrossAttentionBlock(cfg.hidden, cfg.heads, cfg.ffn, cfg.dropout) for _ in range(cfg.cross_layers)])
            self.scorer = CandidateScorer(cfg.hidden)
        else:
            self.head = CandidateHeadV2(cfg, cfg.hidden)
        self.phase = 'a'

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.audio.set_phase(phase)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor, text_ids: torch.Tensor, text_mask: torch.Tensor, spans: list[list[dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        audio, audio_mask = self.audio(waveforms, lengths)
        text = self.text_proj(self.text_tower(text_ids, text_mask))
        if self.fusion == 'v2':
            pooled = masked_mean(audio, audio_mask)
            return self.head(pooled, text, spans, audio, audio_mask)
        logits: list[torch.Tensor] = []
        flat: list[dict[str, Any]] = []
        for b, group_spans in enumerate(spans):
            if not group_spans:
                continue
            for span in group_spans:
                q = text[span['start']:span['end']]
                for block in self.cross:
                    q = block(q, audio[b], audio_mask[b])
                logits.append(self.scorer(q))
                flat.append(span)
        return (logits, flat)

    def total_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters()))

    def trainable_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters() if p.requires_grad))

    def parameter_groups(self) -> list[dict[str, Any]]:
        muq, text, fusion, head = ([], [], [], [])
        for name, p in self.named_parameters():
            if name.startswith('audio.backbone.'):
                muq.append((name, p))
            elif name.startswith('text_tower.'):
                text.append((name, p))
            elif self.fusion == 'v2' and name.startswith('head.'):
                head.append((name, p))
            else:
                fusion.append((name, p))
        groups = [{'name': 'muq', 'params': muq}, {'name': 'text', 'params': text}, {'name': 'fusion', 'params': fusion}]
        if self.fusion == 'v2':
            groups.append({'name': 'head', 'params': head})
        return groups
