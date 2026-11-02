import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="MuQ-Jev music classification")
    parser.add_argument("audio")
    parser.add_argument("--model", default="ABLingss/MUQ-JEV")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", default="genre", choices=["genre", "mood", "instrument", "solo_instrument", "voice_presence", "vocal_gender"])
    parser.add_argument("--candidates", nargs="+")
    parser.add_argument("--kind", choices=["choice", "multi", "noul"])
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--policy", choices=["raw", "model_default"], default="model_default")
    args = parser.parse_args()
    from .inference import MUQJEV
    model = MUQJEV.from_pretrained(args.model, device=args.device)
    result = model.predict(args.audio, task=args.task, candidates=args.candidates, kind=args.kind,
                           start=args.start, duration=args.duration,
                           threshold=args.threshold, policy=args.policy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
