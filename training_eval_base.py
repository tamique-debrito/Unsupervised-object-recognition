
from model import BasicNetIndex, ModelType, get_model
from data_gen import PatchDataset, PredictionType, get_dataloader, get_num_patches, is_cat_pred_type, is_coord_pred_type
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_img(filepath):
    img = Image.open(filepath)
    return np.array(img.resize((512, 512))).transpose((2, 0, 1))

def run_test():
    prediction_type = PredictionType.CLUSTERED_COORD
    device, model, opt, dataloader, dataset = init_objects(prediction_type)
    epochs_per_measure = 50

    annot_imgs = []
    train_metrics = []
    for i in range(10):
        #model.fallback_loss = [0.1, 0.2, 0.1, 0.05, 0.1, 0.2, 0.25, 0.1, 0.3, 0.1][i] #type: ignore
        annot_img = eval_on_img(dataset, model, device, prediction_type, True)
        annot_imgs.append(annot_img)
        train_on_img(dataloader, model, opt, prediction_type, epochs_per_measure, device, train_metrics)
        

    annot_img = eval_on_img(dataset, model, device, prediction_type, True)
    annot_imgs.append(annot_img)

    for annot_img in annot_imgs:
        display_image(annot_img)
    for metric in train_metrics:
        print(metric)

def init_objects(prediction_type, model_dim_override = None, patch_size = 32, img = None, center = None, radius = None, stride = None, lr=0.001, batch_size=256):
    if img is None: img = load_img("./room_img.jpeg")
    #display_img(img)

    device = "cuda"

    n_cluster = 2
    
    num_patches = get_num_patches(img, patch_size, patch_size)[0]

    model_dim = model_dim_override if model_dim_override is not None else num_patches

    model = get_model(model_dim, patch_size, n_cluster, prediction_type).to(device) # each patch is a separate class
    opt = torch.optim.Adam(model.parameters(), lr=lr) #type:ignore

    dataloader, dataset = get_dataloader(img, batch_size, prediction_type, patch_size, center, radius, stride)
    return device, model, opt, dataloader, dataset

def run_full(video, patch_size, batch_size, epochs_per_frame):
    loss_fn = nn.CrossEntropyLoss()
    num_patches = get_num_patches(video[0], patch_size, patch_size)[0]
    model = BasicNetIndex(num_patches, patch_size) # each patch is a separate class
    opt = torch.optim.adam.Adam(model.parameters(), lr=0.01)
    # TODO

def train_on_img(dataloader, model: ModelType, opt, prediction_type, num_epochs, device, train_metrics):
    for e in range(num_epochs):
        sum_loss = 0
        n_correct = 0
        n_samples = 0
        for (patches, targets) in dataloader:
            targets: Tensor
            (patches, targets) = (patches.to(device), targets.to(device))
            pred = model(patches)
            loss = model.get_loss(pred, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_loss += loss
            
            if is_cat_pred_type(prediction_type): n_correct += (pred.argmax(1) == targets).type(torch.float).sum().item()
            n_samples += len(pred)
        
        if e % (max(num_epochs // 3, 1)) == 0: train_metrics.append({
            "loss": sum_loss.item() / len(dataloader),
            "accuracy": n_correct / n_samples if is_cat_pred_type(prediction_type) else None,
        })


def eval_on_img(patch_dataset: Dataset, model: ModelType, device, prediction_type, make_img=True, should_show_img=False):
    model.eval()
    annotated_img = None
    loss_color = np.array([0, 255, 0])
    unconfident_color = np.array([255, 0, 0])
    confident_color = np.array([0, 0, 255])
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])

    if make_img:
        annotated_img = np.copy(patch_dataset.img)
    for index in range(len(patch_dataset)):
        patch, target = patch_dataset[index]
        patch = torch.Tensor(patch).unsqueeze(0)
        target = torch.Tensor(np.expand_dims(target, 0))
        patch, target = patch.to(device), target.to(device)
        pred = model.forward(patch)
        loss: torch.Tensor = model.get_loss(pred, target)
        loss_float = loss.detach().squeeze().cpu().numpy()
        if is_coord_pred_type(prediction_type): loss_float *= 100
        blend = float(np.exp(-loss_float * 2))
        x_l, x_u, y_l, y_u = patch_dataset.patch_center_bounds[index]
        if annotated_img is not None:
            color_img(annotated_img, patch_dataset.coords[index], patch_dataset.stride, loss_color, 1, blend * 0.5) # Map of loss
            if prediction_type == PredictionType.CLUSTERED_COORD:
                _, confidence = pred
                confidence = confidence.detach().squeeze().cpu().numpy()
                confidence = (confidence - model.minimum_confidence) / (1 - model.minimum_confidence)
                small_patch = annotated_img[:, x_l:x_l + 5, y_l:y_l + 5]
                color = unconfident_color * (1 - confidence) + confident_color * confidence
                annotated_img[:, x_l:x_l + 5, y_l:y_l + 5] = color_patch(small_patch * 0, color, 1)
            if prediction_type == PredictionType.CLOSE_TO_TARGET:
                small_patch = annotated_img[:, x_l:x_l + patch_dataset.stride // 2, y_l:y_l + patch_dataset.stride // 2]
                color, blend = (confident_color, 1) if target > 0 else (black, 0)
                color_img(annotated_img, patch_dataset.coords[index], patch_dataset.stride, color, 0.7, blend) # Map of true value
                activation = torch.softmax(pred.squeeze().detach().cpu(),0)[1].item()
                color = white * activation
                color_img(annotated_img, patch_dataset.coords[index], patch_dataset.stride, color, 0.4, 1) # Map of classification probability
    if annotated_img is not None and should_show_img:
        display_image(annotated_img)
    return annotated_img

def display_image(annotated_img, transpose=True):
    if transpose: annotated_img = annotated_img.transpose(1, 2, 0)
    plt.imshow(annotated_img)
    plt.show()

def color_patch(patch, color, blend):
    return (1 - blend) * patch + blend * np.expand_dims(color, (1, 2))

def color_img(img, coord, stride, color, color_radius_factor=1.0, blend=0.5):
    stride = int(stride * color_radius_factor)
    x_l = int(coord[0]) - stride // 2
    x_u = x_l + stride
    y_l = int(coord[1]) - stride // 2
    y_u = y_l + stride

    img[:, x_l: x_u, y_l: y_u] = np.expand_dims(color, (1, 2)) * blend + (1 - blend) * img[:, x_l: x_u, y_l: y_u]

if __name__ == "__main__":
    run_test()