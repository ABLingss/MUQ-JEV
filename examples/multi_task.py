import argparse
import json
from pathlib import Path


TASKS = ("genre", "mood", "instrument", "solo_instrument", "voice_presence", "vocal_gender")


def main():
    parser = argparse.ArgumentParser(description="Run MuQ-Jev tasks on the same audio interval.")
    parser.add_argument("audio")
    parser.add_argument("--model", default="ABLingss/MUQ-JEV", help="Hugging Face model ID or local model directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--policy", choices=("model_default", "raw"), default="model_default")
    parser.add_argument("--output", type=Path, help="Also save the predictions as JSON")
    args = parser.parse_args()

    from muq_jev import MUQJEV

    model = MUQJEV.from_pretrained(args.model, device=args.device)
    results = {
        task: model.predict(args.audio, task=task, start=args.start, duration=args.duration,
                            threshold=args.threshold, policy=args.policy)
        for task in args.tasks
    }
    text = json.dumps(results, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
