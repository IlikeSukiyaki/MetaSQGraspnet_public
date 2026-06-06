from .dsq_loss import DeformableSuperquadricLoss
from .sq_loss import SuperquadricLoss
from .segmentation_loss import SegmentationLoss

def get_loss(cfg, *args, **kwargs):
    loss_dict = cfg["loss"]
    name = loss_dict["type"]
    loss_instance = get_loss_instance(name)
    return loss_instance(**loss_dict)

def get_loss_instance(name):
    try:
        return {
            "sq_loss": SuperquadricLoss,
            "dsq_loss": DeformableSuperquadricLoss,
            "segmentation_loss": SegmentationLoss,
        }[name]
    except:
        raise ("Loss {} not available".format(name))
