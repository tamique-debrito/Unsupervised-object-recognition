
from model import BasicNet
from data_gen import PatchDataset, get_dataloader, get_num_patches
import torch
from torch import Tensor, nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_img(filepath):
    img = Image.open(filepath)
    return np.array(img.resize((512, 512))).transpose((2, 0, 1))

def run_test():
    img = load_img("./room_img.jpeg")
    #plt.imshow(img.transpose(1, 2, 0))
    #plt.show()

    device = "cuda"

    patch_size = 32
    
    loss_fn = nn.CrossEntropyLoss()
    num_patches = get_num_patches(img, patch_size)[0]

    model = BasicNet(num_patches).to(device) # each patch is a separate class
    opt = torch.optim.Adam(model.parameters(), lr=0.0001) #type:ignore

    dataloader, dataset = get_dataloader(img, patch_size, 8)

    annot_imgs = []
    train_metrics = []
    for i in range(20):
        annot_img = eval_on_img(dataset, model, device, loss_fn, True)
        annot_imgs.append(annot_img)
        train_on_img(dataloader, model, opt, loss_fn, 10, device, train_metrics)
    
    annot_img = eval_on_img(dataset, model, device, loss_fn, True)
    annot_imgs.append(annot_img)

    for annot_img in annot_imgs:
        display_image(annot_img)
    for metric in train_metrics:
        print(metric)


def run_full(video, patch_size, batch_size, epochs_per_frame):
    loss_fn = nn.CrossEntropyLoss()
    num_patches = get_num_patches(video[0], patch_size)[0]
    model = BasicNet(num_patches) # each patch is a separate class
    opt = torch.optim.adam.Adam(model.parameters(), lr=0.01)
    # TODO

def train_on_img(dataloader, model, opt, loss_fn, num_epochs, device, train_metrics):
    for e in range(num_epochs):
        sum_loss = 0
        n_correct = 0
        n_samples = 0
        for (patches, labels) in dataloader:
            labels: Tensor
            (patches, labels) = (patches.to(device), labels.long().to(device))
            pred = model(patches)
            loss = loss_fn(pred, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_loss += loss
            n_correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
            n_samples += len(pred)
        train_metrics.append({"loss": sum_loss / len(dataloader), "accuracy": n_correct / n_samples})


def eval_on_img(patch_dataset: PatchDataset, model: BasicNet, device, loss_fn, make_img=True, should_show_img=False):
    model.eval()
    annotated_img = None
    blend_color = np.array([0, 255, 0])
    if make_img:
        annotated_img = np.copy(patch_dataset.img)
    for i in range(len(patch_dataset)):
        patch, int_label = patch_dataset[i]
        patch = torch.Tensor(patch).unsqueeze(0)
        label = torch.Tensor([int_label]).long()
        patch, label = patch.to(device), label.to(device)
        pred = model.forward(patch)
        loss: torch.Tensor = loss_fn(pred, label)
        loss_float = -loss.detach().squeeze().cpu().numpy()
        blend = np.exp(loss_float)
        x_l, x_u, y_l, y_u = patch_dataset.coords[int_label]
        if annotated_img is not None:
            view_patch = annotated_img[:, x_l:x_u, y_l:y_u]
            annotated_img[:, x_l:x_u, y_l:y_u] = color_patch(view_patch, blend_color, blend)
    if annotated_img is not None and should_show_img:
        display_image(annotated_img)
    return annotated_img

def display_image(annotated_img):
    plt.imshow(annotated_img.transpose(1, 2, 0))
    plt.show()

def color_patch(patch, color, blend):
    return (1 - blend) * patch + blend * np.expand_dims(color, (1, 2))

if __name__ == "__main__":
    run_test()