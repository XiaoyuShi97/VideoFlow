import torch
def build_network(cfg):
    name = cfg.network 
    if name == 'MOFNetStack':
        from .MOFNetStack.network import MOFNet as network
    else:
        raise ValueError(f"Network = {name} is not a valid optimizer!")

    return network(cfg[name])
