import argparse

import coolname
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from cfg import ModelConfig
from data import WebFaceDatamodule
from modules import ArcFaceModule, get_resnet_builder


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-C", "--checkpoint-path", help="Path to model checkpoint.", required=True
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="Threshold value used when deciding identities.",
        type=float,
        required=True,
    )
    parser.add_argument(
        "-E",
        "--wandb-entity",
        help="Wandb entity for which test results will be logged. If not set, the default entity is used.",
    )
    parser.add_argument(
        "-P",
        "--wandb-project",
        default="biometry",
        help="Wandb project for which test results will be logged. Default: %(default)s.",
    )
    parser.add_argument(
        "-D",
        "--root-dir",
        default=".",
        help="Root dir for this test logs. Defaults: %(default)s.",
    )
    parser.add_argument(
        "-r",
        "--resnet-version",
        type=int,
        default=ModelConfig.resnet_version,
        help="Resnet version, must match the version used by the checkpointed model. Default: %(default)s.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = cli()

    torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    del hparams["threshold"]
    del hparams["batch_size"]
    state_dict = {
        key.replace("._orig_mod", ""): value
        for key, value in checkpoint["state_dict"].items()
    }
    embedding_dim = len(state_dict["resnet.head.weight"])

    resnet_builder = get_resnet_builder(embedding_dim, args.resnet_version)

    module = ArcFaceModule(
        resnet_builder.build(pretrained=False), threshold=args.threshold, **hparams
    )
    module.load_state_dict(state_dict)

    datamodule = WebFaceDatamodule(resnet_builder.transforms, 1024)

    logger = WandbLogger(
        name=coolname.generate_slug(2),
        save_dir=args.root_dir,
        project=args.wandb_project,
        entity=args.wandb_entity,
        tags=["test", str(args.resnet_version), "casia-webface"],
    )

    trainer = Trainer(
        default_root_dir=args.root_dir,
        logger=logger,
        enable_checkpointing=False,
    )

    trainer.test(model=module, datamodule=datamodule)
