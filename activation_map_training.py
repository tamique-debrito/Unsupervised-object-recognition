import numpy as np
import torch
import matplotlib.pyplot as plt

from model import BasicNetIndex

def format_single_activation_map_frames_for_model(activation_map_frames, frame_period):
    # frame_period: the amount of timesteps between paired frames. Frames will get paired and concatenated to give signal about the motion at a given timestep
    formatted_frames = np.expand_dims(activation_map_frames, 1)
    formatted_frames = np.concatenate([formatted_frames[:-frame_period], formatted_frames[frame_period:]], 1)
    return formatted_frames

def format_multiple_activation_map_frames_for_model(activation_maps_frames, frame_period):
    #activation_maps_frames: list of activation map frames for different activation maps
    formatted_frames = np.stack(activation_maps_frames, axis=1)
    formatted_frames = np.concatenate([formatted_frames[:-frame_period], formatted_frames[frame_period:]], 1)
    return formatted_frames

def get_activation_map_regression_model(activation_map_frames, n_train_steps, lr=0.01, show_heatmap=False):
    activation_map_frames = torch.Tensor(activation_map_frames)
    n_timesteps = activation_map_frames.shape[0]
    n_channels = activation_map_frames.shape[1]
    frames_dim = max(activation_map_frames.shape[2], activation_map_frames.shape[3])
    device = "cuda"
    model = BasicNetIndex(n_timesteps, frames_dim, [n_channels, 4, 4, 4, 4, 8]).to(device)
    activation_map_frames = activation_map_frames.to(device)
    targets = torch.arange(0, n_timesteps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(n_train_steps):
        pred = model(activation_map_frames)
        loss = model.get_loss(pred, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (e % (n_train_steps // 5 + 1) == 0):
            if show_heatmap:
                get_activation_prediction(pred)
            print(f"loss = {loss.detach().cpu().item()}")
    
    return model

def get_heatmap_for_model(model, frames, show=True):
    pred = model(frames)
    return get_activation_prediction(pred, show)


def get_activation_prediction(pred, show=True):
    softmax = torch.nn.Softmax(dim=1)
    p = softmax(pred)
    p = p.detach().cpu().numpy()
    if show:
        show_heatmap(p)
    return p

def show_heatmap(p):
    plt.imshow(p, cmap='hot', interpolation='nearest')
    plt.ylabel("Timestep")
    plt.xlabel("Predicted timestep")
    plt.show()