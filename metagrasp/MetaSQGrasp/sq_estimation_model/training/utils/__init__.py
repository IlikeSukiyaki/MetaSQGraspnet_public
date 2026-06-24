import yaml

from utils.checkpoints import best_checkpoint_name


def save_yaml(filename, text):
    with open(filename, "w", encoding="utf-8") as handle:
        yaml.dump(yaml.safe_load(text), handle, default_flow_style=False)
