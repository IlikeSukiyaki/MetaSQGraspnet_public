import os
import argparse
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from loader import get_dataloader
from models import get_model
from optimizers import get_optimizer
from trainers import get_trainer
from utils import save_yaml


def run(cfg, writer):

    # Setup seeds
    seed = cfg.get("seed", 1)
    print(f"running with random seed : {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setup device
    device = cfg.device

    # Setup Dataloader
    datainfo = cfg.data
    dataloaders = {}
    for key, data_cfg in datainfo.items():
        dataloaders[key] = get_dataloader(data_cfg)

    # Setup Model
    model = get_model(cfg).to(device)
    trainer = get_trainer(cfg)

    # Setup optimizer, lr_scheduler and loss function
    if hasattr(model, "own_optimizer") and model.own_optimizer:
        optimizer = model.get_optimizer(cfg["training"]["optimizer"])
    else:
        optimizer = get_optimizer(cfg["training"]["optimizer"], model.parameters())

    model, train_result = trainer.train(
        model,
        optimizer,
        dataloaders,
        writer=writer,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--device", default=0)
    parser.add_argument("--logdir", default="train_results/")
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.device == "cpu" or not torch.cuda.is_available():
        cfg["device"] = "cpu"
    else:
        cfg["device"] = f"cuda:{args.device}"

    if args.run is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
    else:
        run_id = args.run

    config_basename = os.path.basename(args.config).split(".")[0]
    logdir = os.path.join(args.logdir, config_basename, str(run_id))
    writer = SummaryWriter(log_dir=logdir)
    print("Result directory: {}".format(logdir))

    # copy config file
    copied_yml = os.path.join(logdir, os.path.basename(args.config))
    save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
    print(f"config saved as {copied_yml}")

    run(cfg, writer)
