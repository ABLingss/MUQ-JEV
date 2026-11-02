from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from scipy.signal import resample_poly

from .config import ModelConfig
from .model import U1Model
from .prompts import build_prompt
from .text import TextTokenizer


class MUQJEV:
    def __init__(self, directory: str | Path, device: str = "cpu"):
        self.directory = Path(directory).resolve()
        self.config = json.loads((self.directory / "config.json").read_text())
        if self.config.get("format_version") != 1:
            raise ValueError("Unsupported model format")
        model_config = dict(self.config["model"])
        for key in ("muq_path", "minilm_path"):
            path = (self.directory / model_config[key]).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError("Model asset path escapes the snapshot")
            model_config[key] = path
        model_config["taps"] = tuple(model_config["taps"])
        self.device = torch.device(device)
        self.model = U1Model(ModelConfig(**model_config))
        state = load_file(str(self.directory / "model.safetensors"), device="cpu")
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.to(self.device).eval()
        self.tokenizer = TextTokenizer(model_config["minilm_path"], model_config["max_text_len"])
        self.sample_rate = int(self.config["sample_rate"])
        self.max_seconds = float(self.config["max_seconds"])

    @classmethod
    def from_pretrained(cls, model_id: str | Path = "ABLingss/MUQ-JEV", *,
                        device: str = "cpu", revision: str | None = None,
                        token: str | bool | None = None,
                        local_files_only: bool = False) -> "MUQJEV":
        path = Path(model_id)
        if not path.is_dir():
            path = Path(snapshot_download(
                repo_id=str(model_id), revision=revision, token=token,
                local_files_only=local_files_only,
                allow_patterns=["config.json", "model.safetensors", "muq/*", "text/*"],
            ))
        return cls(path, device=device)

    def _read_audio(self, audio_path: str | Path, start: float, duration: float) -> np.ndarray:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in {".opus", ".ogg", ".mp3", ".m4a"}:
            try:
                return self._read_soundfile(path, start, duration)
            except (RuntimeError, sf.LibsndfileError):
                pass
        import imageio_ffmpeg
        import subprocess
        result = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error",
            "-ss", f"{start:.4f}", "-i", str(path), "-t", f"{duration:.4f}",
            "-f", "f32le", "-ac", "1", "-ar", str(self.sample_rate), "-",
        ], capture_output=True)
        if result.returncode:
            if path.suffix.lower() in {".opus", ".ogg"}:
                return self._read_soundfile(path, start, duration)
            raise RuntimeError("Audio decoding failed")
        return np.frombuffer(result.stdout, dtype=np.float32).copy()

    def _read_soundfile(self, path: Path, start: float, duration: float) -> np.ndarray:
        with sf.SoundFile(path) as stream:
            rate = stream.samplerate
            offset = round(start * rate)
            if offset >= len(stream):
                raise ValueError("start is outside the audio")
            stream.seek(offset)
            audio = stream.read(round(duration * rate), dtype="float32", always_2d=True).mean(axis=1)
        if rate != self.sample_rate:
            divisor = math.gcd(rate, self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // divisor, rate // divisor).astype(np.float32)
        return np.ascontiguousarray(audio, dtype=np.float32)

    @torch.inference_mode()
    def predict(self, audio_path: str | Path, *, task: str = "genre",
                candidates: list[str] | None = None, kind: str | None = None,
                start: float = 0.0, duration: float = 30.0,
                threshold: float = 0.5, policy: str = "model_default") -> dict:
        if task not in self.config["tasks"]:
            raise ValueError("Unknown task: " + task)
        if policy not in {"raw", "model_default"}:
            raise ValueError("policy must be raw or model_default")
        if not math.isfinite(start) or start < 0:
            raise ValueError("start must be a finite nonnegative number")
        if not math.isfinite(duration) or not 0 < duration <= self.max_seconds:
            raise ValueError(f"duration must be in (0, {self.max_seconds}]")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        spec = self.config["tasks"][task]
        labels = list(spec["candidates"] if candidates is None else candidates)
        if not labels or any(not isinstance(x, str) or not x.strip() for x in labels):
            raise ValueError("candidates must contain nonempty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("candidates must be unique")
        kind = spec["kind"] if kind is None else kind
        if kind not in {"choice", "multi", "noul"}:
            raise ValueError("kind must be choice, multi or noul")
        if kind == "noul" and len(labels) != 2:
            raise ValueError("noul requires two candidates")
        gated = task == "vocal_gender" and policy == "model_default" and kind != "multi"
        if gated and ("none" not in labels or not set(labels) <= {"male", "female", "none", "mixed"} or len(labels) < 2):
            raise ValueError("Use policy='raw' for custom vocal-gender candidates")
        audio = self._read_audio(audio_path, start, duration)
        if len(audio) < self.sample_rate // 4 or not np.isfinite(audio).all():
            raise ValueError("Audio must contain at least 0.25 seconds of finite samples")
        decisions = [{"task": task, "kind": kind, "candidates": labels}]
        if gated:
            decisions.append({"task": "voice_presence", "kind": "noul", "candidates": ["no", "yes"]})
        texts, spans = [], []
        for decision in decisions:
            offset = len(texts)
            texts.extend(build_prompt(decision["task"], label) for label in decision["candidates"])
            spans.append({**decision, "start": offset, "end": len(texts)})
        ids, mask = self.tokenizer.encode(texts)
        logits, _ = self.model(
            torch.from_numpy(audio)[None].to(self.device),
            torch.tensor([len(audio)], dtype=torch.long, device=self.device),
            ids.to(self.device), mask.to(self.device), [spans],
        )
        results = []
        for decision, logit in zip(decisions, logits):
            value = logit.float()
            if not torch.isfinite(value).all():
                raise RuntimeError("Model produced non-finite scores")
            multi = decision["kind"] == "multi"
            probs = (value.sigmoid() if multi else value.softmax(-1)).cpu().tolist()
            raw_scores = dict(zip(decision["candidates"], probs))
            scores = dict(raw_scores)
            rule = self.config.get("decision_policy", {}).get(decision["task"])
            if not multi and policy != "raw" and rule and set(scores) == set(rule["priors"]):
                scores = {k: v / rule["priors"][k] ** rule["strength"] for k, v in scores.items()}
                total = sum(scores.values())
                scores = {k: v / total for k, v in scores.items()}
            prediction = [k for k, v in scores.items() if v >= threshold] if multi else max(scores, key=scores.get)
            results.append({"task": decision["task"], "prediction": prediction,
                            "scores": scores, "raw_scores": raw_scores})
        result = results[0]
        if gated:
            voice = results[1]
            if voice["prediction"] == "no":
                scores = {k: float(k == "none") for k in labels}
            else:
                present = [k for k in labels if k != "none"]
                indices = [labels.index(k) for k in present]
                probabilities = logits[0][indices].float().softmax(-1).cpu().tolist()
                scores = {k: 0.0 for k in labels}
                scores.update(zip(present, probabilities))
            result.update(prediction=max(scores, key=scores.get), scores=scores,
                          voice_prediction=voice["prediction"])
        result["policy"] = policy
        return result
