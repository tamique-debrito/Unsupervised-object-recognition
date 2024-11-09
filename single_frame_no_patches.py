import numpy as np
import torch
import matplotlib.pyplot as plt

from activation_map_training import format_multiple_activation_map_frames_for_model, get_activation_map_regression_model
from model import BasicNetIndexFullImg
from torch_utils import show_all_activation_maps
from training_eval_base import display_image
from utils import activation_map_to_rgb, get_nth_frame, load_video, play_video

def get_frame_model(frame, h, w, c_y_norm, c_x_norm, r_norm, n_train_steps, lr=0.001, show_heatmaps=False, use_aug=False, target_based=True, dropout=None, use_max_pool=True, weight_for_class=0.8, dims=None, starting_model=None, display_mode="periodic"):
    frame = torch.Tensor(frame)
    device = "cuda"
    if dims is None:
        if use_max_pool: dims = [3, 16, 32, 64]
        else: dims = [3, 8, 8, 16, 16, 32, 32, 32, 64, 64]
    if starting_model is None: model = BasicNetIndexFullImg(h, w, dims, target_based=target_based, dropout=dropout, use_max_pool=use_max_pool)
    else: model = starting_model
    model.to(device)
    if target_based: model.cross_entropy = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1 - weight_for_class, weight_for_class]).to(device))
    frame = frame.to(device).unsqueeze(0)
    targets = model.get_labels(c_y_norm, c_x_norm, r_norm).to(device)
    if False:
        label_img = targets.reshape(model.post_conv_dims).cpu().numpy()
        label_img = activation_map_to_rgb(label_img)[0]
        display_image(label_img, transpose=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_metrics = []

    if use_aug:
        aug_conv = torch.nn.Conv2d(3, 3, 1)
        aug_conv.eval()

    #show_activation_map(frame, model)
    for e in range(n_train_steps):
        if use_aug:
            w = torch.normal(torch.eye(3, device=device).unsqueeze(2).unsqueeze(3), 0.5).clip(0, 1.2) * np.random.uniform(0.9, 1.1)
            b = torch.normal(torch.zeros(3, device=device), 0.05).clip(0, 1)
            frame_to_use = torch.nn.functional.conv2d(frame, w, b).detach()
            #display_image(frame_to_use.squeeze().cpu().numpy())
        else:
            frame_to_use = frame

        pred = model(frame_to_use)
        loss = model.get_loss(pred, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

        accuracy = (pred.argmax(1) == targets).type(torch.float).sum().item() / pred.shape[0]

        train_metrics.append({
            "loss": loss.item(),
            "accuracy": accuracy,
        })

        if e % (n_train_steps // 10 + 2) == 0:
            if show_heatmaps:
                show_target_activation_map(frame, model, model.get_class_at_center(c_y_norm, c_x_norm))
            if display_mode == "periodic":
                print(train_metrics[-1])
    if display_mode == "final only":
        print(train_metrics[-1])

    return model

def show_target_activation_map(frame, model: BasicNetIndexFullImg, class_id=1):
    activation_map = model.get_target_activation_map(frame, class_id)
    activation_map = activation_map.detach().cpu().numpy()
    activation_map = activation_map_to_rgb(activation_map)[0]
    display_image(activation_map, transpose=False)

def show_target_activation_video(frames, model: BasicNetIndexFullImg, class_id=1):
    activation_map = model.get_target_activation_map(frames, class_id)
    activation_map = activation_map.detach().cpu().numpy()
    activation_map = activation_map_to_rgb(activation_map)
    play_video(activation_map, swap_dims=False, fps=4)

def show_activation_video_center_classes(frames, model: BasicNetIndexFullImg, c_y_norm, c_x_norm, r_norm):
    activation_map = model.get_target_activation_map_for_classes(frames, c_x_norm, c_y_norm, r_norm)
    activation_map = activation_map.detach().cpu().numpy()
    activation_map = activation_map_to_rgb(activation_map)
    play_video(activation_map, swap_dims=False, fps=4)

def get_coords(activ_map: np.ndarray):
    y_dim, x_dim = activ_map.shape
    y_coords = np.repeat(np.expand_dims(np.arange(y_dim),1), x_dim, 1)
    x_coords = np.repeat(np.expand_dims(np.arange(x_dim),0), y_dim, 0)
    coords = np.stack([y_coords, x_coords], axis=2)
    return coords, y_dim, x_dim

def update_target(orig_c_y_norm, orig_c_x_norm, orig_r_norm, new_activation_map):
    new_r_norm = orig_r_norm # Don't try to do anything with this for now
    coords, y_dim, x_dim = get_coords(new_activation_map)
    orig_c_y = int(y_dim * orig_c_y_norm)
    orig_c_x = int(x_dim * orig_c_x_norm)
    closeness_r = int(orig_r_norm * (y_dim + x_dim) / 2 * 1.5)
    closeness_mask = np.zeros((y_dim, x_dim))
    closeness_mask[orig_c_y -  closeness_r: orig_c_y + closeness_r, orig_c_x - closeness_r: orig_c_x + closeness_r] = 1
    masked_map = new_activation_map * closeness_mask
    masked_map = np.expand_dims(masked_map, 2)
    weighted_coords = coords * masked_map
    mean_coord = weighted_coords.sum((0, 1)) / masked_map.sum()
    new_c_y_norm = mean_coord[0] / y_dim
    new_c_x_norm = mean_coord[1] / x_dim
    return new_c_y_norm, new_c_x_norm, new_r_norm

def get_info_obj_1():
    c_y_norm, c_x_norm, r_norm = 200 / 256, 175 / 256, 16 / 256
    return c_y_norm, c_x_norm, r_norm

def get_info_obj_2():
    c_y_norm, c_x_norm, r_norm = 195 / 256, 100 / 256, 16 / 256
    return c_y_norm, c_x_norm, r_norm

def test_average_multi_run():
    maps = []
    resize_dim = 128
    steps_per_run = 2500
    frame_num = 20
    frames = load_video("./multi-motion take 1.mp4", 64, frame_skip=1, resize_to=(resize_dim, resize_dim)) / 256
    cuda_frames = torch.Tensor(frames).to("cuda")
    frame = frames[frame_num]
    h, w = frame.shape[1:]
    c_y_norm, c_x_norm, r_norm = get_info_obj_1()
    
    training_params_list = [(0.001, False, None, 0.5), (0.0003, True, None, 0.5), (0.001, True, None, 0.5), (0.007, False, None, 0.8), (0.0005, False, None, 0.99), (0.001, False, None, 0.2), (0.0003, False, None, 0.99)]
    for lr, use_aug, dropout, weight in training_params_list:
        print("New model")
        model = get_frame_model(frame, h, w, c_y_norm, c_x_norm, r_norm, steps_per_run, lr=lr, use_aug=use_aug, show_heatmaps=False, dropout=dropout, use_max_pool=False, weight_for_class=weight)
        act_map = model.get_target_activation_map(cuda_frames, 1).detach().cpu().numpy()
        maps.append(act_map)
        del model
    
    avg_map = np.array(maps).mean(axis=0)
    # for _ in range(4):
    #     play_video(activation_map_to_rgb(avg_map), swap_dims=False, fps=3, name="Averaged map")

    from PIL import Image

    imgs = activation_map_to_rgb(avg_map)
    imgs = np.array(imgs / imgs.max() * 255, dtype=np.uint8)
    imgs = [Image.fromarray(img) for img in imgs]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save("activation_map_seq.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)



def test_multiframe_tracking_training():
    use_aug=False
    c1 = np.array([200, 175])
    r1 = 16
    frame_num = 20
    frame = get_nth_frame("./multi-motion take 1.mp4", frame_num) / 256
    h, w = frame.shape[1:]
    c_y_norm, c_x_norm, r_norm = get_info_obj_1()
    model = get_frame_model(frame, h, w, c_y_norm, c_x_norm, r_norm, 500, lr=0.0005, use_aug=use_aug, show_heatmaps=False, dropout=None, use_max_pool=False, weight_for_class=0.5)
    maps_before_train_step = []
    maps_after_train_step = []
    maps_after_full_train = []
    coord_tracking = []
    coord_tracking.append((c_y_norm, c_x_norm, r_norm))
    frames = load_video("./multi-motion take 1.mp4", 128, frame_skip=2) / 256
    cuda_frames = torch.Tensor(frames).to("cuda")
    n_tracking_steps = 0
    for i in range(n_tracking_steps):
        print(f"new frame {i}")
        frame = frames[frame_num + i]
        cuda_frame = cuda_frames[frame_num + i].unsqueeze(0)
        model.eval()
        activation_map = model.get_target_activation_map(cuda_frame, 1).squeeze(0).detach().cpu().numpy()
        maps_before_train_step.append(activation_map)
        c_y_norm, c_x_norm, r_norm = update_target(c_y_norm, c_x_norm, r_norm, activation_map)
        coord_tracking.append((c_y_norm, c_x_norm, r_norm))
        model.train()
        model = get_frame_model(frame, h, w, c_y_norm, c_x_norm, r_norm, 10, lr=0.00005, use_aug=use_aug, show_heatmaps=False, dropout=None, use_max_pool=False, weight_for_class=0.7, starting_model=model)
        model.eval()
        activation_map = model.get_target_activation_map(cuda_frame, 1).squeeze(0).detach().cpu().numpy()
        maps_after_train_step.append(activation_map)
    model.eval()
    for i in range(100):
        cuda_frame = cuda_frames[i].unsqueeze(0)
        activation_map = model.get_target_activation_map(cuda_frame, 1).squeeze(0).detach().cpu().numpy()
        maps_after_full_train.append(activation_map)
    
    for map_anim, name in zip([maps_before_train_step, maps_after_train_step, maps_after_full_train], ["before train", "after train", "after full train"]):
        if len(map_anim) == 0: continue
        maps_frames = np.array(map_anim)
        play_video(activation_map_to_rgb(maps_frames), swap_dims=False, fps=3, name=name)
    print(coord_tracking)
    




def test():
    c1 = np.array([200, 175])
    r1 = 16
    c2 = np.array([195, 100])
    r2 = 16
    frame_num = 20
    frame = get_nth_frame("./multi-motion take 1.mp4", frame_num) / 256
    h, w = frame.shape[1:]
    c_y_norm, c_x_norm, r_norm = get_info_obj_1()
    dims = [3, 8, 8, 16, 16, 32]
    use_max_pool = [True, True, False, True, False]
    assert len(dims) - 1 == len(use_max_pool)
    model1 = get_frame_model(frame, h, w, c_y_norm, c_x_norm, r_norm, 2000, lr=0.001, use_aug=False, target_based=False, show_heatmaps=False, dropout=0.3, use_max_pool=use_max_pool, weight_for_class=0.8, dims=dims)
    frame = torch.Tensor(frame).to("cuda").unsqueeze(0)
    model1.eval()
    show_all_activation_maps(model1, frame)
    return
    c_y_norm, c_x_norm, r_norm = get_info_obj_2()
    #model2 = get_frame_model(frame, h, w, c_y_norm, c_x_norm, r_norm, 1000, lr=0.01, use_aug=False, show_heatmaps=False, dropout=0.5)

    #model1.eval()
    #model2.eval()

    frames = load_video("./multi-motion take 1.mp4", 32, frame_skip=2) / 256
    frames = torch.Tensor(frames).to("cuda")

    # for (dims,) in zip([[3, 16, 16, 16, 16], [3, 16, 16, 16, 16, 32], [3, 16, 16, 32, 32, 64, 64]]):
    #     multi_run(frame, frames, c1, r1, steps=2000, lr=0.003, aug=False, dropout=0.5, maxpool=False, w=0.6, use_eval=True, dims=dims)


    activation_map_frames1 = model1.get_target_activation_map(frames, 1).detach().cpu().numpy()
    #activation_map_frames2 = model2.get_activation_map(frames, 1).detach().cpu().numpy()
    play_video(activation_map_to_rgb(activation_map_frames1), swap_dims=False, fps=3)
    #play_video(activation_map_to_rgb(activation_map_frames2), swap_dims=False, 10, fps=8)
    #formatted_frames = format_multiple_activation_map_frames_for_model([activation_map_frames1, activation_map_frames2], 8)
    #model = get_activation_map_regression_model(formatted_frames, 20, 0.01, True)

def run_with_loaded_info(frame, frames, c1, r1, steps, lr, aug, dropout, maxpool, w, use_eval, dims):
    h, w = frame.shape[1:]
    c_y_norm, c_x_norm, r_norm = get_info_obj_1()
    model1 = get_frame_model(frame, h, w, c_y_norm, c_x_norm, r_norm, steps, lr=lr, use_aug=aug, show_heatmaps=False, dropout=dropout, use_max_pool=maxpool, weight_for_class=w, dims=dims)
    if use_eval: model1.eval()
    activation_map_frames1 = model1.get_target_activation_map(frames, 1).detach().cpu().numpy()
    play_video(activation_map_to_rgb(activation_map_frames1), swap_dims=False, fps=5)


if __name__ == "__main__":
    #test()
    test_average_multi_run()
"""
from single_frame_no_patches import *
c1 = np.array([200, 175])
r1 = 16
frame_num = 20
frame = get_nth_frame("./multi-motion take 1.mp4", frame_num) / 256
frames = load_video("./multi-motion take 1.mp4", 32, frame_skip=2) / 256
def live_run(steps, lr, aug, dropout, maxpool, w, use_eval):
    model1 = get_frame_model(frame, c1, r1, steps, lr=lr, use_aug=aug, show_heatmaps=False, dropout=dropout, use_max_pool=maxpool, weight_for_class=w)
    if use_eval: model1.eval()
    activation_map_frames1 = model1.get_activation_map(frames, 1).detach().cpu().numpy()
    play_video(activation_map_to_rgb(activation_map_frames1), swap_dims=False, 5, fps=8)
"""