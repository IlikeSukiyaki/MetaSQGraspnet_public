"""
TensorBoard training entry point for sq_esti_model / extsq_esti_model recognition models.

Mirrors train.py but logs decomposed loss components (geometric fit, orientation
prior) and additional convergence diagnostics to TensorBoard instead of W&B.
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

TRAINING_ROOT = Path(__file__).resolve().parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from data_loaders import get_dataloader
from losses import get_loss
from metrics.meter import averageMeter
from optimizers import get_optimizer
from registry.models import get_model
from utils import save_yaml
from utils.checkpoints import best_checkpoint_name


def _has_compute_terms(loss_fn):
    return callable(getattr(loss_fn, "compute_terms", None))


def _compute_loss_terms(loss_fn, x_gt, l_gt, output):
    if _has_compute_terms(loss_fn):
        return loss_fn.compute_terms(x_gt, l_gt, output)
    loss_total = loss_fn(x_gt, l_gt, output)
    return {
        "loss_total": loss_total,
        "loss_dist": loss_total,
        "loss_z": torch.tensor(0.0, device=loss_total.device),
        "loss_z_weighted": torch.tensor(0.0, device=loss_total.device),
    }


def _log_prediction_diagnostics(writer, output, global_step, is_extsq):
    """Log lightweight scalar stats from raw network output."""
    size = output[:, 7:10]
    shape = output[:, 10:12]
    writer.add_scalar("prediction/size_min", size.min().item(), global_step)
    writer.add_scalar("prediction/size_max", size.max().item(), global_step)
    writer.add_scalar("prediction/size_mean", size.mean().item(), global_step)
    writer.add_scalar("prediction/shape_min", shape.min().item(), global_step)
    writer.add_scalar("prediction/shape_max", shape.max().item(), global_step)
    writer.add_scalar("prediction/shape_mean", shape.mean().item(), global_step)
    if is_extsq and output.shape[1] >= 16:
        taper = output[:, 12:13]
        bending = output[:, 13:14]
        writer.add_scalar("prediction/taper_mean", taper.mean().item(), global_step)
        writer.add_scalar("prediction/bending_mean", bending.mean().item(), global_step)


def _make_meters():
    keys = ("total", "dist", "orient", "orient_weighted")
    return {split: {key: averageMeter() for key in keys} for split in ("train", "val")}


def _update_meters(meters, split, terms):
    meters[split]["total"].update(terms["loss_total"].item())
    meters[split]["dist"].update(terms["loss_dist"].item())
    meters[split]["orient"].update(terms["loss_z"].item())
    meters[split]["orient_weighted"].update(terms["loss_z_weighted"].item())


def _log_split_losses(writer, meters, split, global_step):
    prefix = "loss"
    tag = "train" if split == "train" else "val"
    writer.add_scalar(f"{prefix}/{tag}_total", meters[split]["total"].avg, global_step)
    writer.add_scalar(
        f"{prefix}/{tag}_geometric_fit", meters[split]["dist"].avg, global_step
    )
    writer.add_scalar(
        f"{prefix}/{tag}_orientation_prior", meters[split]["orient"].avg, global_step
    )
    writer.add_scalar(
        f"{prefix}/{tag}_orientation_prior_weighted",
        meters[split]["orient_weighted"].avg,
        global_step,
    )


def save_model(model, logdir, best=False, i_iter=None, epoch=None, model_arch="sq_esti_model"):
    if best:
        pkl_name = best_checkpoint_name(model_arch)
    elif i_iter is not None:
        pkl_name = f"model_iter_{i_iter}.pkl"
    else:
        pkl_name = f"model_epoch_{epoch}.pkl"
    state = {"epoch": epoch, "model_state": model.state_dict(), "iter": i_iter}
    save_path = os.path.join(logdir, pkl_name)
    torch.save(state, save_path)
    print(f"Model saved: {pkl_name}")


def train_recognition(model, optimizer, dataloaders, loss_fn, cfg, device, writer, logdir):
    train_loader = dataloaders["training"]
    val_loader = dataloaders["validation"]
    training_cfg = cfg["training"]

    clip_grad = training_cfg.get("clip_grad", 1)
    is_extsq = cfg["model"]["arch"] == "extsq_esti_model"
    model_arch = cfg["model"]["arch"]
    best_val_loss = np.inf
    iteration = 0
    meters = _make_meters()
    time_meter = averageMeter()

    for epoch in range(training_cfg.n_epoch):
        for x, x_gt, l_gt, _y in train_loader:
            iteration += 1
            model.train()
            x = x.to(device)
            x_gt = x_gt.to(device)
            l_gt = l_gt.to(device)

            optimizer.zero_grad()
            start_ts = time.time()
            output = model(x)
            terms = _compute_loss_terms(loss_fn, x_gt, l_gt, output)
            terms["loss_total"].backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_grad if clip_grad is not None else float("inf")
            )
            optimizer.step()

            time_meter.update(time.time() - start_ts)
            _update_meters(meters, "train", terms)

            if iteration % training_cfg.print_interval == 0:
                _log_split_losses(writer, meters, "train", iteration)
                writer.add_scalar("train/grad_norm", grad_norm.item(), iteration)
                writer.add_scalar("train/epoch", epoch, iteration)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)
                _log_prediction_diagnostics(writer, output.detach(), iteration, is_extsq)
                print(
                    f"[Training] Iter [{iteration:d}] "
                    f"Total: {meters['train']['total'].avg:.4f} "
                    f"Geom: {meters['train']['dist'].avg:.4f} "
                    f"Orient: {meters['train']['orient'].avg:.4f} "
                    f"Elapsed: {time_meter.sum:.4f}s"
                )
                for key in meters["train"]:
                    meters["train"][key].reset()
                time_meter.reset()

            if iteration % training_cfg.save_interval == 0:
                save_model(model, logdir, best=False, i_iter=iteration)

            if iteration % training_cfg.val_interval == 0:
                model.eval()
                for split_key in meters["val"]:
                    meters["val"][split_key].reset()

                with torch.no_grad():
                    for val_x, val_x_gt, val_l_gt, _val_y in val_loader:
                        val_x = val_x.to(device)
                        val_x_gt = val_x_gt.to(device)
                        val_l_gt = val_l_gt.to(device)
                        val_output = model(val_x)
                        val_terms = _compute_loss_terms(
                            loss_fn, val_x_gt, val_l_gt, val_output
                        )
                        _update_meters(meters, "val", val_terms)

                _log_split_losses(writer, meters, "val", iteration)
                val_loss = meters["val"]["total"].avg
                print(
                    f"[Validation] Iter [{iteration:d}] "
                    f"Total: {val_loss:.4f} "
                    f"Geom: {meters['val']['dist'].avg:.4f} "
                    f"Orient: {meters['val']['orient'].avg:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(model, logdir, best=True, i_iter=iteration, model_arch=model_arch)
                    writer.add_scalar("validation/best_loss", best_val_loss, iteration)
                    print(
                        f"[Validation] Iter [{iteration:d}] best model saved "
                        f"{val_loss:.4f} <= previous best"
                    )
                else:
                    writer.add_scalar("validation/best_loss", best_val_loss, iteration)

    return model, best_val_loss


def run(cfg, writer, logdir):
    seed = cfg.get("seed", 1)
    print(f"running with random seed : {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = cfg.device
    dataloaders = {
        key: get_dataloader(data_cfg) for key, data_cfg in cfg.data.items()
    }

    model = get_model(cfg).to(device)
    loss_fn = get_loss(cfg["training"], device=device)

    if hasattr(model, "own_optimizer") and model.own_optimizer:
        optimizer = model.get_optimizer(cfg["training"]["optimizer"])
    else:
        optimizer = get_optimizer(cfg["training"]["optimizer"], model.parameters())

    if cfg.get("trainer", "recognition") != "recognition":
        raise NotImplementedError(
            "train_tensorboard.py currently supports trainer: recognition only."
        )

    return train_recognition(
        model, optimizer, dataloaders, loss_fn, cfg, device, writer, logdir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train sq_esti_model / extsq_esti_model with TensorBoard logging."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", default=0)
    parser.add_argument("--logdir", default="train_results_tensorboard/")
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.device == "cpu" or not torch.cuda.is_available():
        cfg["device"] = "cpu"
    else:
        cfg["device"] = f"cuda:{args.device}"

    run_id = datetime.now().strftime("%Y%m%d-%H%M") if args.run is None else args.run
    config_basename = os.path.basename(args.config).split(".")[0]
    logdir = os.path.join(args.logdir, config_basename, str(run_id))
    tb_logdir = os.path.join(logdir, "tensorboard")
    print(f"Result directory: {logdir}")
    print(f"TensorBoard log directory: {tb_logdir}")

    os.makedirs(logdir, exist_ok=True)
    copied_yml = os.path.join(logdir, os.path.basename(args.config))
    save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
    print(f"config saved as {copied_yml}")

    writer = SummaryWriter(log_dir=tb_logdir)
    try:
        run(cfg, writer, logdir)
    finally:
        writer.close()
        print(f"TensorBoard logs saved under {tb_logdir}")
        print(f"View with: tensorboard --logdir {args.logdir}")
