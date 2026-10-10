from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import load_panel
from .trainer import predict, train, verify


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone TRA alpha model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--config", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--run-dir", required=True)
    predict_parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command == "check":
        config = load_config(args.config)
        panel = load_panel(config)
        print(json.dumps({"status": "valid", "panel": panel.summary(config["splits"]),
                          "config": config["_config_path"]}, ensure_ascii=False))
    elif args.command == "train":
        run_dir = train(args.config)
        print(json.dumps({"status": "complete", "run_dir": str(run_dir)}, ensure_ascii=False))
    elif args.command == "predict":
        output = predict(args.run_dir, args.split)
        print(json.dumps({"status": "complete", "signals": str(output)}, ensure_ascii=False))
    else:
        print(json.dumps(verify(args.run_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
