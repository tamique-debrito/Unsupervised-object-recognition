import colorsys
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import numpy as np

from data_gen import PatchDataset, PredictionType
from model import ModelType
from training_eval_base import color_patch, display_image, init_objects, load_img, train_on_img


def test():
    prediction_type = PredictionType.PATCH_INDEX
    epochs_per_measure = 150
    train_metrics = []
    n_clusters = 16
    device, model, opt, dataloader, dataset = init_objects(prediction_type)

    train_on_img(dataloader, model, opt, prediction_type, epochs_per_measure, device, train_metrics)

    get_clusters(dataset, model, device, n_clusters, True)

    
    device, model, opt, _, _ = init_objects(prediction_type, n_clusters)

    dataset.use_cluster = True

    train_on_img(dataloader, model, opt, prediction_type, epochs_per_measure, device, train_metrics)

    get_clusters(dataset, model, device, n_clusters, True)

def get_cluster_color(cluster, n_clusters):
    return np.array(colorsys.hsv_to_rgb(cluster/n_clusters, 1, 1)) * 255

def proj_logits(logits):
    pca = PCA(n_components=8)
    pca.fit(logits)
    proj1 = pca.components_.transpose((1, 0))
    proj2 = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(proj1)
    return proj2



def get_clusters(patch_dataset: PatchDataset, model: ModelType, device, n_clusters=5, show_clusters_and_img=False):
    model.eval()
    logits = [] # Logits from the model
    locations = [] # Indices in the dataset
    for index in range(len(patch_dataset)):
        patch, target = patch_dataset[index]
        patch = torch.Tensor(patch).unsqueeze(0)
        target = torch.Tensor(np.expand_dims(target, 0))
        patch, target = patch.to(device), target.to(device)
        pred = model.forward(patch)
        logit = pred.squeeze().detach().cpu().numpy()
        location = target.detach().cpu().numpy().item()
        logits.append(np.concatenate([logit, patch_dataset.coords[index]]))
        locations.append(location)
    
    logits = np.array(logits)
    projected = proj_logits(logits)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans = kmeans.fit(projected)
    cluster_labels = kmeans.predict(projected)

    for cluster_label, loc in zip(cluster_labels, locations):
        patch_dataset.set_cluster_for_sample(loc, cluster_label)
    
    if show_clusters_and_img:
        plt.scatter(projected[:, 0], projected[:, 1], c=cluster_labels)
        plt.show()
        annotated_img = np.copy(patch_dataset.img)
        for index in range(len(patch_dataset)):
            x_l, x_u, y_l, y_u = patch_dataset.patch_center_bounds[index]
            cluster = patch_dataset.cluster_map[index]
            view_patch = annotated_img[:, x_l:x_u, y_l:y_u]
            annotated_img[:, x_l:x_u, y_l:y_u] = color_patch(view_patch, get_cluster_color(cluster, n_clusters), 1)
        display_image(annotated_img)
        
    
if __name__ == "__main__":
    test()