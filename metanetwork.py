#a more "out there" idea: some kind of "feature agnostic" learning. e.g. if many networks are training using a common training algorithm, then a "meta network" could predict based on the features of those networks, with the idea that if you trained a new network, the features of that network would give the same outputs as the other ones when put into the "meta network"

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
from psutil import disk_partitions
from torch import Tensor
import torch

from model import BasicNetIndex, BasicNetIndexFullImg
from single_frame_no_patches import get_frame_model
from utils import get_nth_frame, run_in_new_process

@dataclass
class TrainParams:
    lr: float
    n_steps: int

def gen_model(resize_dim):
    return BasicNetIndexFullImg(resize_dim, resize_dim, [3, 8, 8, 16], False, None, True)

def train_single_image_metanetwork_fixed_params():
    # Same parameters for all data networks
    resize_dim = 256
    buffer_name = "METANET_TRAINING_SET"
    video_filepath = "./multi-motion take 1.mp4"
    ref_frame_number = 25
    train_samples = 32
    eval_samples = 16
    total_samples = train_samples + eval_samples
    queue = mp.Queue()

    frames_save_filepath = "./tmp/data/buffer"
    saved_frames = False

    if saved_frames:
        print("Loading saved frames")
        frames_from_buffer = np.load(frames_save_filepath)
    else:
        run_in_new_process(get_shape, (resize_dim, queue))
        buffer_shape = (total_samples,) + queue.get()
        print(f"buffer shape = {buffer_shape}")
        buffer_size = int(np.prod(buffer_shape) * np.dtype(np.float32).itemsize)
        buffer = shared_memory.SharedMemory(create=True, size=buffer_size, name=buffer_name)
        frames_from_buffer = np.ndarray(shape=buffer_shape, dtype=np.float32, buffer=buffer.buf)
        train_params = TrainParams(0.0005, 2500)
        for buffer_index in range(total_samples):
            print(f"training sample {buffer_index + 1} / {total_samples}")
            run_in_new_process(
                train_and_fill_single, 
                (video_filepath, ref_frame_number, resize_dim,
                train_params, gen_model,
                buffer_name, buffer_index, buffer_size, buffer_shape)
            )

    if not saved_frames:
        np.save(frames_save_filepath, frames_from_buffer)

    print("training metanetwork")

    
    metanetwork = train_metanetwork(frames_from_buffer[:-eval_samples], 5000, 0.003)
    eval_metanetwork(frames_from_buffer[-eval_samples:], metanetwork)
    print(f"metanetwork output shape = {metanetwork.post_conv_shape}")

    metanetwork = train_metanetwork(frames_from_buffer[:-eval_samples], 5000, 0.001)
    eval_metanetwork(frames_from_buffer[-eval_samples:], metanetwork)
    
def train_metanetwork(frames, n_train_steps, lr):
    n_sample, channel, h, w = frames.shape
    device = "cuda"
    metanetwork = BasicNetIndexFullImg(h, w, [channel, 32, 32, 32], use_max_pool=False).to(device)
    opt = torch.optim.Adam(metanetwork.parameters(), lr=lr)
    frames = Tensor(frames).to(device)
    labels = metanetwork.get_labels().repeat(n_sample).to(device)
    train_metrics = []
    for e in range(n_train_steps):
        pred = metanetwork(frames)
        loss = metanetwork.get_loss(pred, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        accuracy = (pred.argmax(1) == labels).type(torch.float).sum().item() / pred.shape[0]

        train_metrics.append({
            "loss": loss.item(),
            "accuracy": accuracy,
        })

        if e % (n_train_steps // 10 + 2) == 0:
            print(train_metrics[-1])
    return metanetwork

def eval_metanetwork(frames, metanetwork):
    metanetwork.eval()
    n_sample, channel, h, w = frames.shape
    device = "cuda"
    frames = Tensor(frames).to(device)
    labels = metanetwork.get_labels().repeat(n_sample).to(device)
    pred = metanetwork(frames)
    loss = metanetwork.get_loss(pred, labels)

    accuracy = (pred.argmax(1) == labels).type(torch.float).sum().item() / pred.shape[0]

    print({
        "eval loss": loss.item(),
        "eval accuracy": accuracy,
    })

def get_shape(resize_dim, queue: mp.Queue):
    net: BasicNetIndexFullImg = gen_model(resize_dim)
    queue.put(net.post_conv_shape)

def train_and_fill_single(
        video_filepath, ref_frame_number, resize_dim,
        train_params: TrainParams, gen_model,
        buffer_name, buffer_index, buffer_size, buffer_shape):
    frame = get_nth_frame(video_filepath, ref_frame_number, (resize_dim, resize_dim))
    model = get_frame_model(frame, resize_dim, resize_dim, None, None, None, train_params.n_steps, train_params.lr, target_based=False, starting_model=gen_model(resize_dim), display_mode="final only")
    cuda_frame = Tensor(frame).to("cuda").unsqueeze(0)
    reps = model.convs(cuda_frame).squeeze(0).detach().cpu().numpy()
    print(f"representation shape = {reps.shape}")

    buffer = shared_memory.SharedMemory(size=buffer_size, name=buffer_name)
    frames_from_buffer = np.ndarray(shape=buffer_shape, dtype=np.float32, buffer=buffer.buf)
    frames_from_buffer[buffer_index] = reps

if __name__ == "__main__":
    train_single_image_metanetwork_fixed_params()