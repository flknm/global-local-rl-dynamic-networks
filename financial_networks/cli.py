"""Command-line entry point for individual training runs."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .trainer import ProjectedReinforceTrainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON configuration.")
    parser.add_argument("--output-dir", required=True, help="New or resumable run directory.")
    parser.add_argument("--resume", help="Optional checkpoint from the same configuration.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    trainer = ProjectedReinforceTrainer(
        config, args.output_dir, resume=args.resume is not None
    )
    try:
        if args.resume:
            trainer.load_checkpoint(args.resume)
        summary = trainer.train()
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
