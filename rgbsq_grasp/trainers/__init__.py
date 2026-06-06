from .segmentation_trainer import SegmentationTrainer
from .recognition_trainer import RecognitionTrainer


def get_trainer(cfg):
    trainer_type = cfg.get("trainer", None)
    device = cfg["device"]
    if trainer_type == "segmentation":
        trainer = SegmentationTrainer(cfg["training"], device=device)
    elif trainer_type == "recognition":
        trainer = RecognitionTrainer(cfg["training"], device=device)
    else:
        raise NotImplementedError(f"trainer {trainer_type} not implemented")
    return trainer

