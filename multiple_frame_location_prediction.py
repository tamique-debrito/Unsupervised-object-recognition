# One encoder, multiple final linear layers, one for each distinct frame
import numpy as np
import torch

from model import BasicNetIndexFullImgMultiFrame
import random

from torch_utils import show_all_activation_maps
from utils import load_video


def get_frame_model(frames, h, w, n_train_steps, lr=0.001, use_aug=False, target_based=True, dropout=None, use_max_pool=True, weight_for_class=0.8, dims=None, starting_model=None, display_mode="periodic"):
    device = "cuda"
    frame_indices = list(range(len(frames)))
    frames = torch.Tensor(frames)
    frames = frames.to(device)
    if dims is None:
        if use_max_pool: dims = [3, 16, 32, 64]
        else: dims = [3, 8, 8, 16, 16, 32, 32, 32, 64, 64]
    if starting_model is None: model = BasicNetIndexFullImgMultiFrame(h, w, len(frame_indices), dims, target_based=target_based, dropout=dropout, use_max_pool=use_max_pool)
    else: model = starting_model
    model: BasicNetIndexFullImgMultiFrame
    model.to(device)
    if target_based: model.cross_entropy = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1 - weight_for_class, weight_for_class]).to(device))
    targets = model.get_labels().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_metrics = []

    if use_aug:
        aug_conv = torch.nn.Conv2d(3, 3, 1)
        aug_conv.eval()

    #show_activation_map(frame, model)
    for e in range(n_train_steps):
        random.shuffle(frame_indices)
        one_epoch_all_frames_train_metrics = []
        for frame_idx in frame_indices:
            frame_to_use = frames[frame_idx].unsqueeze(0)
            if use_aug:
                w = torch.normal(torch.eye(3, device=device).unsqueeze(2).unsqueeze(3), 0.5).clip(0, 1.2) * np.random.uniform(0.9, 1.1)
                b = torch.normal(torch.zeros(3, device=device), 0.05).clip(0, 1)
                frame_to_use = torch.nn.functional.conv2d(frame_to_use, w, b).detach()
                #display_image(frame_to_use.squeeze().cpu().numpy())

            pred = model(frame_to_use, frame_idx)
            loss = model.get_loss(pred, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()

            accuracy = (pred.argmax(1) == targets).type(torch.float).sum().item() / pred.shape[0]

            one_epoch_all_frames_train_metrics.append({
                "loss": round(loss.item(), 3),
                "accuracy": round(accuracy, 3),
            })

        if e % (n_train_steps // 10 + 2) == 0:
            if display_mode == "periodic":
                print(one_epoch_all_frames_train_metrics)
        train_metrics.append(one_epoch_all_frames_train_metrics)

    if display_mode == "final only":
        print(train_metrics[-1])

    return model

def test():
    frames = load_video("./multi-motion take 1.mp4", 50, (256, 256), 2) / 256
    h, w = frames[0].shape[1:]
    dims = [3, 8, 8, 16]
    use_max_pool = [True, True, False]
    assert len(dims) - 1 == len(use_max_pool)
    model1 = get_frame_model(frames, h, w, 200, lr=0.001, use_aug=False, target_based=False, dropout=0.3, use_max_pool=use_max_pool, weight_for_class=0.8, dims=dims)
    model1.eval()
    for i in range(10):
        frame = torch.Tensor(frames[i]).to("cuda").unsqueeze(0)
        show_all_activation_maps(model1, frame, [0, 1])
    return

if __name__ == "__main__":
    test()