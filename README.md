<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="MuQ-Jev" width="480">
  </picture>
  <p><strong>Listen. Compare. Decide.</strong></p>
  <p>
    <a href="#-paper"><img src="https://img.shields.io/badge/Paper-Coming%20soon-8B5CF6?style=flat-square" alt="Paper: Coming soon"></a>
    <a href="https://huggingface.co/ABLingss/MUQ-JEV"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-Hugging%20Face-F59E0B?style=flat-square" alt="Models on Hugging Face"></a>
    <a href="https://github.com/ABLingss/MUQ-JEV"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&amp;logo=github" alt="GitHub code"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20License-MIT-0F766E?style=flat-square" alt="Code license: MIT"></a>
  </p>
</div>

**MuQ-Jev** is a compact, candidate-based music classification model built on MuQ.
Supports genre, mood, instruments, voice presence and vocal gender.

## 🗞 News

- **[2026-11-02]** 🎉 Initial inference code and model weights uploaded — v0.1.0.

## 🎧 Quick Start

Python 3.10+. Install matching PyTorch and torchaudio builds for your device, then:

```bash
git clone https://github.com/ABLingss/MUQ-JEV.git
cd MUQ-JEV
pip install .
muq-jev music.wav --device cuda --task genre
```

```python
from muq_jev import MUQJEV

model = MUQJEV.from_pretrained("ABLingss/MUQ-JEV", device="cuda")
result = model.predict("music.wav", task="genre", candidates=["jazz", "rock", "classical"])
print(result["prediction"])
```

Uses up to 30 seconds per call. Scores are uncalibrated; custom-candidate performance may vary.
Use `policy="raw"` for direct model scores, or `device="cpu"` for CPU inference.
While the model is private, authenticate with your Hugging Face account before downloading.

## 🧩 Examples

Run all six tasks on the same audio interval:

```bash
python examples/multi_task.py music.wav --device cuda --output results.json
```

See [examples](examples/README.md) for task outputs, custom candidates and interval selection.

## 📄 Paper

Coming soon.

## 📜 License

Code: MIT. Pretrained components retain their original licenses; see [notices](THIRD_PARTY_NOTICES.md).
