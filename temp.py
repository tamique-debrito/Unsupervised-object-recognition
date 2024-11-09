import numpy as np
from torch import Tensor
from activation_map_training import format_multiple_activation_map_frames_for_model, get_activation_map_regression_model
from object_tracking_a import run_and_evaluate_location_prediction
from training_eval_base import display_image
from utils import activation_map_to_rgb, get_nth_frame, load_video, play_video



def put_object_area(frame: np.ndarray, center, radius):
    y, x = center
    f = frame.copy()
    f[:, y-radius:y+radius, x-radius:x+radius] = 0
    return f

def show_annotation():
    c1 = np.array([200, 175])
    r1 = 20
    c2 = np.array([195, 100])
    r2 = 16
    frame_num = 20
    frame = get_nth_frame("./multi-motion take 1.mp4", frame_num) / 255
    annot_frame = put_object_area(frame, c1, r1)
    annot_frame = put_object_area(annot_frame, c2, r2)
    #display_image(annot_frame)
    
    model1 = run_and_evaluate_location_prediction("./multi-motion take 1.mp4", frame_num, r1, c1)
    model2 = run_and_evaluate_location_prediction("./multi-motion take 1.mp4", frame_num, r2, c2)
    frames = load_video("./multi-motion take 1.mp4", 128, frame_skip=1) / 255
    frames = Tensor(frames).to("cuda")
    activation_map_frames1 = model1.get_activation_map(frames, 1).detach().cpu().numpy()
    activation_map_frames2 = model2.get_activation_map(frames, 1).detach().cpu().numpy()
    play_video(activation_map_to_rgb(activation_map_frames1), swap_dims=False, fps=8)
    play_video(activation_map_to_rgb(activation_map_frames2), swap_dims=False, fps=8)
    formatted_frames = format_multiple_activation_map_frames_for_model([activation_map_frames1, activation_map_frames2], 8)
    #activation_map_frames = np.zeros((64, 15, 15))
    model = get_activation_map_regression_model(formatted_frames, 20, 0.01, True)

show_annotation()