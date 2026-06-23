from .extsq_esti_loss import ExtSqEstiLoss
from .sq_esti_loss import SqEstiLoss


def get_loss(cfg, *args, **kwargs):
    loss_dict = cfg["loss"]
    name = loss_dict["type"]
    loss_instance = get_loss_instance(name)
    return loss_instance(**loss_dict)


def get_loss_instance(name):
    try:
        return {
            "sq_esti_loss": SqEstiLoss,
            "extsq_esti_loss": ExtSqEstiLoss,
        }[name]
    except KeyError as exc:
        raise ValueError(f"Loss {name} not available") from exc
