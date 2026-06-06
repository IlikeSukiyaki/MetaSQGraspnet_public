from .segmentation_metric import runningScore_seg
from .meter import averageMeter

def get_metric(metric_dict, **kwargs):
    name = metric_dict.pop("type")
    metric_instance = get_metric_instance(name)
    return metric_instance(**metric_dict)

def get_metric_instance(name):
    try:
        return {
            "segmentation": runningScore_seg,
        }[name]
    except:
        raise ("Metric {} not available".format(name))
