import argparse

import wandb


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download checkpoints from wandb.")
    parser.add_argument(
        "-A", "--artifact", help="Path to wandb artifact with the checkpoint."
    )
    parser.add_argument(
        "-D",
        "--dir",
        default="checkpoints",
        help="Directory where the checkpoint will be saved. Default: %(default)s.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = cli()
    api = wandb.Api()
    artifact = api.artifact(args.artifact)
    artifact.download(args.dir)
