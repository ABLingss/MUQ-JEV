# Examples

Install the package following the [Quick Start](../README.md#-quick-start), then run these commands from the repository root with your own audio file.

## All six tasks

```bash
python examples/multi_task.py music.wav --device cuda --output results.json
```

The script loads the model once and calls `predict` for each task on the same interval. It prints predictions and candidate scores as JSON, and optionally saves them to a file.

| Task | Output |
| --- | --- |
| `genre` | One genre from the candidate set |
| `mood` | One mood from the candidate set |
| `instrument` | Multiple detected candidates, using a score threshold |
| `solo_instrument` | One solo-instrument candidate |
| `voice_presence` | `yes` or `no` |
| `vocal_gender` | `male`, `female`, `none` or `mixed` |

These are model predictions, not benchmark accuracy or calibrated confidence. For audio without a solo instrument, the solo task still selects from its supplied candidates.

## Choose tasks and an interval

```bash
python examples/multi_task.py music.wav --device cuda --tasks genre mood instrument --start 10 --duration 20
```

`--duration` is at most 30 seconds. Use `--device cpu` for CPU inference, `--model /path/to/model` for a local snapshot, or `--policy raw` for direct model scores.

## Custom candidates

Single-label genre choice:

```bash
muq-jev music.wav --device cuda --task genre --candidates jazz rock classical
```

Multi-label instrument detection:

```bash
muq-jev music.wav --device cuda --task instrument --candidates piano guitar drums --threshold 0.5
```

The supplied candidate strings are the output keys. `instrument` can return several labels or an empty list; single-label tasks select one supplied candidate.
