import os

import torch
from random import randint
from omegaconf import OmegaConf
import sys
from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
import numpy as np
import random
from dataset.depth_dataset import ReplicaDataset, ScannetDataset
from dataset.datamodule import DataModule
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import (ModelCheckpoint)
from models.depth_wrapper import ModelWrapper
from tensorboardX import SummaryWriter
import time


def train(args):
    # This allows the current step to be shared with the data loader processes.
    # step_tracker = StepTracker()
    cfg = OmegaConf.load(args.cfg)
    print(cfg)
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            os.path.join(args.output_dir, "checkpoints"),
            filename="ckpt_step-{step:06d}",
            auto_insert_metric_name=False,
            monitor="val/ssim",
            save_top_k=3,
            mode="max"
        )
    )

    date = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logger = loggers.TensorBoardLogger("tb_logs", name=f'{args.expname}', version=f'{date}')

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=[7],
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )

    model_wrapper = ModelWrapper(
        cfg,
        args,
    )
    data_module = DataModule(
        ReplicaDataset if "replica" in args.cfg else ScannetDataset,
        OmegaConf.merge(cfg.base, cfg.data),
        global_rank=trainer.global_rank,
    )
    if args.mode == 'train':
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=args.ckpt if args.ckpt != "" else None)
    else:
        trainer.validate(model_wrapper, datamodule=data_module, ckpt_path=args.ckpt)


def seed_everything(seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--dataset_name', type=str, default="scannet")
    parser.add_argument('--cfg', type=str, default="./configs/scannet_depth.yaml")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--gpus', type=list, default=[0])
    parser.add_argument('--save_dir', type=str, default="./save")
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--pre_feature', type=str, default="")
    parser.add_argument('--expname', type=str, default="base")
    parser.add_argument("--save_model", action="store_true", help="save")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.save_dir, args.expname)

    ## Setting seeds
    seed_everything(args.seed)
    pl.seed_everything(args.seed)

    train(args)

