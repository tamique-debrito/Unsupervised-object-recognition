from utils import show_activation_map
import numpy as np

def show_all_activation_maps(model, x, restrict_to_layers=None):
    maps = model.get_activation_maps(x)
    for i, block_map in enumerate(maps):
        if restrict_to_layers is not None and i not in restrict_to_layers:
            continue
        for j, act_map in enumerate(block_map):
            print(f"min={np.min(act_map)} max={np.max(act_map)} mean={np.mean(act_map)}")
            show_activation_map(act_map/(np.max(act_map) + 1e-10), f"l={i+1} ch={j + 1}")