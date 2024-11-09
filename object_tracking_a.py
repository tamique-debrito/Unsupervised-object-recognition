# Manually define the coordinate to target + radius around. Train with a "1" label for any coord within the radius, "0" for not
# Then on the next frame, run the prediction. Average out the coordinates to get the new coordinate (don't worry about outliers at first)
from data_gen import PredictionType
from training_eval_base import display_image, eval_on_img, init_objects, train_on_img
import numpy as np
from torch import Tensor, nn

from utils import activation_map_to_rgb, get_first_frame, get_nth_frame, load_video, play_video

def run_and_evaluate_location_prediction(filename, frame_number, radius, center, include_eval=False):
    img = get_nth_frame(filename, frame_number) / 255
    #display_image(img)

    patch_size = 32
    stride = 4
    lr = 0.0002
    prediction_type = PredictionType.CLOSE_TO_TARGET
    device, model, opt, dataloader, dataset = init_objects(prediction_type, patch_size=patch_size, center=center, radius=radius, stride=stride, img=img, lr=lr)
    epochs_per_measure = 50
    model.cross_entropy = nn.CrossEntropyLoss(weight=Tensor([0.05, 0.95]).to(device))

    annot_imgs = []
    train_metrics = []
    for i in range(3):
        print(f"iter {i}")
        if include_eval:
            annot_img = eval_on_img(dataset, model, device, prediction_type, True)
            annot_imgs.append(annot_img)
        train_on_img(dataloader, model, opt, prediction_type, epochs_per_measure, device, train_metrics)
    for m in train_metrics:
        print(m)

    if include_eval:
        annot_img = eval_on_img(dataset, model, device, prediction_type, True)
        annot_imgs.append(annot_img)

        for i in range(10):
            # Now take a frame from a few timesteps forward and see what the prediction annotation looks like
            new_img = get_nth_frame(filename, frame_number + 6 * (i + 1))
            dataset.update_image(new_img)
            annot_img = eval_on_img(dataset, model, device, prediction_type, True)
            annot_imgs.append(annot_img)

        for annot_img in annot_imgs:
            display_image(annot_img)
    # for metric in train_metrics:
    #     print(metric)
    return model


def run_test():
    
    c1 = np.array([200, 175])
    r1 = 16
    frame_num = 20
    
    model1 = run_and_evaluate_location_prediction("./multi-motion take 1.mp4", frame_num, r1, c1, include_eval=False)
    frames = load_video("./multi-motion take 1.mp4", 128, frame_skip=1)
    play_video(frames)
    frames = Tensor(frames).to("cuda")
    activation_map_frames1 = model1.get_activation_map(frames, 1).detach().cpu().numpy()
    print(activation_map_frames1.max((1, 2)))
    activation_map_frames1 = activation_map_frames1 / (activation_map_frames1.max() + 0.0000001)
    #play_video(activation_map_to_rgb(activation_map_frames1), swap_dims=False, scale=20, fps=8)
    #run_and_evaluate_location_prediction("./moving_sunglasses.mp4", 1, 16, np.array([105, 165]))
    return model1, frames, activation_map_frames1

if __name__ == "__main__":
    run_test()

    # from object_tracking_a import *
    # m, f, a = run_test()