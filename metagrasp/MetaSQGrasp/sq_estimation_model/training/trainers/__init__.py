from trainers.recognition_trainer import RecognitionTrainer


def get_trainer(cfg):
    trainer_type = cfg.get("trainer", None)
    device = cfg["device"]
    if trainer_type == "recognition":
        return RecognitionTrainer(cfg["training"], device=device)
    raise NotImplementedError(f"trainer {trainer_type} not implemented")
