import yaml

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, 'w') as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)