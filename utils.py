from multiprocessing import Process
from time import sleep
from PIL import Image
import cv2
import numpy as np

def activation_map_to_rgb(frames):
    frames = np.expand_dims(frames, 3)
    frames = np.tile(frames, (1, 1, 1, 3))
    return frames

def show_activation_map(frame, name, resize_to=(256, 256)):
    frames = np.expand_dims(frame, 2)
    frames = np.tile(frames, (1, 1, 3))

    if resize_to is not None:
        frame = cv2.resize(frame, resize_to)

    cv2.imshow(name, frame)
    while True:
        if cv2.waitKey(100) & 0xFF == ord('n'):
            break

def save_video(frames, filename):
    pass

def save_activation_map_to_video(frames, filename):
    frames = activation_map_to_rgb(frames)
    save_video(frames, filename)

def play_video(frames, swap_dims=True, fps=24, name="Frame"):
    # frames: numpy array of frames. If it's in order (channel, y, x), pass swap_dims=True 
    delay = int(1000 / fps)
    for frame in frames:
        if swap_dims:
            frame = np.transpose(frame, (1, 2, 0))
        
        frame = cv2.resize(frame, (256, 256))

        cv2.imshow(name, frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break


def get_nth_frame(filename, n, resize_to=(256, 256)):
    cap = cv2.VideoCapture(filename) # type: ignore
    for _ in range(n): ret, frame = cap.read()
    frame = Image.fromarray(frame)
    frame = frame.resize(resize_to)
    frame = np.array(frame).transpose((2, 0, 1))
    return frame

def get_first_frame(filename):
    return get_nth_frame(filename, 1)

def load_video(filename, n_frames, resize_to=(256, 256), frame_skip=1):
    # Load video into numpy array of (frame, channel, h, w)
    cap = cv2.VideoCapture(filename) # type: ignore
    frames = []
    for idx in range(n_frames):
        for _ in range(frame_skip): ret, frame = cap.read()
        if not ret:
            break
        frame = Image.fromarray(frame)
        frame = frame.resize(resize_to)
        frame = np.array(frame).transpose((2, 0, 1))
        frames.append(frame)
    return np.array(frames)

def run_in_new_process(target, args):
    p = Process(target=target, args=args)
    p.start()
    p.join()

if __name__ == "__main__":
    video = load_video("./moving_sunglasses.mp4", 64)
    play_video(video, True)
