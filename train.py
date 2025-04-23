import coolname
import pytorch_lightning
import torch
import tyro
from pytorch_lightning.loggers.wandb import WandbLogger

from cfg import Config
from data import WebFaceDatamodule
from modules import ArcFaceModule, get_resnet_builder

if __name__ == "__main__":
    cfg = tyro.cli(Config)

    torch.set_float32_matmul_precision("high")

    if cfg.experiment.seed is not None:
        pytorch_lightning.seed_everything(cfg.experiment.seed)

    resnet_builder = get_resnet_builder(
        embedding_dim=cfg.model.embedding_dim, version=cfg.model.resnet_version
    )

    datamodule = WebFaceDatamodule(
        transform=resnet_builder.transforms,
        batch_size=cfg.experiment.batch_size,
    )

    module = ArcFaceModule(
        resnet=resnet_builder.build(cfg.model.pretrained),
        num_classes=datamodule.num_train_classes,
        arc_face_scale=cfg.model.arc_face_scale,
        arc_face_margin=cfg.model.arc_face_margin,
        threshold=cfg.experiment.threshold,
        epochs=cfg.experiment.epochs,
        optimizer_config=cfg.optimizer,
        compile_submodules=cfg.model.compile,
    )

    logger = WandbLogger(
        name=coolname.generate_slug(2),
        save_dir=cfg.logger.root_dir,
        log_model=True,
        project=cfg.logger.wandb_project,
        entity=cfg.logger.wandb_entity,
        tags=["train", str(cfg.model.resnet_version), "casia-webface"],
    )
    logger.experiment.config.update(cfg.serialize())

    trainer = pytorch_lightning.Trainer(
        default_root_dir=cfg.logger.root_dir,
        logger=logger,
        check_val_every_n_epoch=cfg.logger.validation_every_n_epochs,
        enable_checkpointing=True,
        max_epochs=cfg.experiment.epochs,
    )
