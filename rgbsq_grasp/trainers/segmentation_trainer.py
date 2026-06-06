import os
import time

import torch
import numpy as np

from metrics import get_metric, averageMeter
from loss import get_loss


class SegmentationTrainer:
    """Trainer for a conventional iterative training of model"""

    def __init__(self, training_cfg, device):
        self.cfg = training_cfg
        self.device = device
        self.time_meter = averageMeter()
        self.loss_meter = {"train": averageMeter(), "val": averageMeter()}

        self.loss = get_loss(training_cfg)
        self.show_metric = False
        dict_metric = training_cfg.get("metric", None)
        if dict_metric is not None:
            self.metric = get_metric(dict_metric)
            self.show_metric = True

    def train(self, model, opt, dataloaders, writer):
        cfg = self.cfg
        logdir = writer.log_dir
        best_val_loss = np.inf
        iter = 0

        train_loader, val_loader = (
            dataloaders["training"],
            dataloaders["validation"],
        )

        for epoch in range(cfg.n_epoch):
            for x, y, _ in train_loader:
                # training
                iter += 1

                model.train()
                x = x.to(self.device)
                y = y.to(self.device)

                start_ts = time.time()
                train_step_result = model.train_step(
                    x, y=y, optimizer=opt, loss_function=self.loss, kwargs=cfg
                )

                self.time_meter.update(time.time() - start_ts)
                self.loss_meter["train"].update(train_step_result["loss"])

                # record
                if iter % cfg.print_interval == 0:
                    self.record_results(writer, iter, "train", train_step_result)
                    print(
                        f"[Training] Iter [{iter:d}] Avg Loss: {self.loss_meter['train'].avg:.4f} Elapsed time: {self.time_meter.sum:.4f}"
                    )
                    self.time_meter.reset()

                # save model
                if iter % cfg.save_interval == 0:
                    self.save_model(model, logdir, best=False, i_iter=iter)

                if iter % cfg.val_interval == 0:
                    # validation
                    model.eval()
                    for val_x, val_y, _ in val_loader:
                        val_x = val_x.to(self.device)
                        val_y = val_y.to(self.device)

                        val_step_result = model.validation_step(
                            val_x, y=val_y, loss_function=self.loss, kwargs=cfg
                        )
                        self.loss_meter["val"].update(val_step_result["loss"])
                        self.metric.update(
                            val_step_result["gt"].argmax(axis=2),
                            val_step_result["pred"].argmax(axis=2),
                        )

                    # record
                    self.record_results(writer, iter, "val", train_step_result)
                    val_loss = self.loss_meter["val"].avg
                    print(f"[Validation] Iter [{iter:d}] Avg Loss: {val_loss:.4f}")

                    # save model
                    if val_loss < best_val_loss:
                        self.save_model(model, logdir, best=True, i_iter=iter)
                        print(
                            f"[Validation] Iter [{iter:d}] best model saved {val_loss} <= {best_val_loss}"
                        )
                        best_val_loss = val_loss

        return model, best_val_loss

    def record_results(self, writer, i, tag, results):
        # record loss
        writer.add_scalar(f"loss/{tag}_loss", self.loss_meter[tag].avg, i)

        # record segmentation result
        if i % self.cfg["visualize_interval"] == 0:
            pc = results["pc"][0 : self.cfg["visualize_number"]]
            gt = results["gt"][0 : self.cfg["visualize_number"]]
            pred = results["pred"][0 : self.cfg["visualize_number"]]

            pc_gt_colors = color_pc_segmentation(pc, gt)
            pc_pred_colors = color_pc_segmentation(pc, pred)

            writer.add_mesh(
                f"{tag} gt", vertices=pc, colors=pc_gt_colors, global_step=i
            )
            writer.add_mesh(
                f"{tag} pred", vertices=pc, colors=pc_pred_colors, global_step=i
            )

        # record metrics
        if tag == "val":
            scores = self.metric.get_scores()
            self.metric.reset()
            for key, score in scores.items():
                writer.add_scalar(f"metrics/{tag}_{key.replace(' ', '_')}", score, i)

    def save_model(self, model, logdir, best=False, i_iter=None, epoch=None):
        if best:
            pkl_name = "model_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = f"model_iter_{i_iter}.pkl"
            else:
                pkl_name = f"model_epoch_{epoch}.pkl"
        state = {"epoch": epoch, "model_state": model.state_dict(), "iter": i_iter}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f"Model saved: {pkl_name}")


rgb_colors = {
    "black": [0, 0, 0],
    "red": [255, 0, 0],
    "pink": [255, 96, 208],
    "purple": [160, 32, 255],
    "light_blue": [80, 208, 255],
    "blue": [0, 32, 255],
    "green": [0, 192, 0],
    "orange": [255, 160, 16],
    "brown": [160, 128, 96],
    "gray": [128, 128, 128],
}


def color_pc_segmentation(pc, label):
    class_idx_label = np.argmax(label, axis=2)

    pc_colors = np.zeros(pc.shape)
    for batch_idx in range(pc_colors.shape[0]):
        for point_idx in range(pc_colors.shape[1]):
            pc_colors[batch_idx, point_idx, :] = list(rgb_colors.values())[
                class_idx_label[batch_idx, point_idx]
            ]

    return pc_colors
