import logging

from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

logger = logging.getLogger("sq_estimation_model")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(opt_dict, model_params):
    optimizer = _get_optimizer_instance(opt_dict)
    params = {k: v for k, v in opt_dict.items() if k != "name"}
    return optimizer(model_params, **params)


def _get_optimizer_instance(opt_dict):
    if opt_dict is None:
        logger.info("Using SGD optimizer")
        return SGD

    opt_name = opt_dict["name"]
    if opt_name not in key2opt:
        raise NotImplementedError(f"Optimizer {opt_name} not implemented")
    logger.info("Using %s optimizer", opt_name)
    return key2opt[opt_name]
