from typing import Union
from torch import isin, nn, Tensor
import torch
import torchvision

from data_gen import PredictionType

def get_model(n_classes, patch_size, n_cluster, prediction_type):
    if prediction_type == PredictionType.PATCH_INDEX: return BasicNetIndex(n_classes, patch_size)
    elif prediction_type == PredictionType.COORD: return BasicNetCoord()
    elif prediction_type == PredictionType.CLUSTERED_COORD: return BasicNetCoordCluster(n_cluster)
    if prediction_type == PredictionType.CLOSE_TO_TARGET: return BasicNetIndex(2, patch_size)
    assert False, "invalid prediction type"

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=None, use_max_pool=True):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3)
        self.use_max_pool = use_max_pool
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()
        self.use_dropout = dropout is not None
        self.dropout = nn.Dropout(dropout if dropout is not None else 0)

    def forward(self, x):
        x = self.conv(x)
        if self.use_max_pool:
            x = self.max_pool(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

    def get_activation_maps(self, x):
        activations = []
        x = self.forward(x)
        x_np = x[0].detach().cpu().numpy()
        for i in range(self.out_dim):
            act = x_np[i]
            activations.append(act)
        return x, activations
        

class Convolutions(nn.Module):
    def __init__(self, dims, dropout=None, use_max_pool=True) -> None:
        super().__init__()
        components = []
        for i, (d1, d2) in enumerate(zip(dims, dims[1:])):
            if isinstance(use_max_pool, list):
                use_max_pool_for_block = use_max_pool[i]
            else:
                use_max_pool_for_block = use_max_pool
            conv_block = ConvBlock(d1, d2, dropout, use_max_pool_for_block)
            components.append(conv_block)
        
        self.components = nn.ModuleList(components)
    
    def forward(self, x):
        for block in self.components:
            x = block(x)
        return x

    def get_activation_maps(self, x):
        block_activations = []
        for block in self.components: # type: ignore
            block: ConvBlock
            x, block_activation = block.get_activation_maps(x)
            block_activations.append(block_activation)
        return block_activations




class BasicNetCoordCluster(nn.Module):
    def __init__(self, n_cluster = 8):
        super(BasicNetCoordCluster, self).__init__()
        self.mse = nn.MSELoss()
        #
        self.clusters = nn.Parameter(torch.zeros((1, n_cluster, 2)))
        #
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2= nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.adaptive_maxpool = nn.AdaptiveMaxPool2d(1)
        #
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.relu4 = nn.ReLU()
        self.fc_temp = nn.Linear(in_features=64, out_features=2)
        self.fc_cluster = nn.Linear(in_features=64, out_features=n_cluster)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor):
        #
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        #
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        #
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.adaptive_maxpool(x)
        x = x.squeeze(2, 3)
        #
        x = self.relu4(self.fc1(x))
        #
        output = self.fc_temp(x)
        cluster_weights = self.fc_cluster(x)
        cluster_weights = self.softmax(cluster_weights).unsqueeze(2) # cluster_weights: (batch, n_cluster, 1). self.clusters: (1, n_cluster, 2)
        output = (cluster_weights * self.clusters).sum(dim=1) # sum out the cluster dimension
        #
        return output

    def get_loss(self, output, targets):
        return self.mse(output, targets)
        



class BasicNetCoord(nn.Module):
    def __init__(self):
        super(BasicNetCoord, self).__init__()
        #
        self.mse = nn.MSELoss()
        #
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2= nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.AdaptiveMaxPool2d(1)
        #
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x: Tensor):
        #
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        #
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        #
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.squeeze(2, 3)
        #
        x = self.relu4(self.fc1(x))
        output = self.fc2(x)
        return output

    def get_loss(self, output, targets):
        return self.mse(output, targets)

class BasicNetIndex(nn.Module):
    def __init__(self, classes, patch_size, dims=None):
        super(BasicNetIndex, self).__init__()
        #
        self.cross_entropy = nn.CrossEntropyLoss()
        #
        if dims is None: dims = [3, 16, 32, 64]
        #
        self.dims = dims
        self.patch_size = patch_size
        self.convs = Convolutions(dims)
        final_pool_size = self.get_final_maxpool_size()
        self.final_pool = nn.MaxPool2d(final_pool_size)
        #self.final_pool = nn.AvgPool2d(final_pool_size)
        #
        self.fc1 = nn.Linear(in_features=dims[-1], out_features=64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=classes)
        #
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=3) #used for generating activation maps

    def get_final_maxpool_size(self):
        return self.convs(torch.zeros(1, self.dims[0], self.patch_size, self.patch_size)).shape[2]
        

    def forward(self, x: Tensor):
        #
        x = self.convs(x)
        x = self.final_pool(x)
        x = x.squeeze(2, 3)
        #
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        #
        output = self.log_softmax(x)
        return output

    def get_loss(self, output, targets):
        targets = targets.long()
        return self.cross_entropy(output, targets)
    
    def get_activation_map(self, frames, class_id):
        # returns: (frame, h, w)
        x: Tensor
        x = self.convs(frames)
        x = self.final_pool(x)
        x = x.permute(0, 2, 3, 1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        activation_map = x[:, :, :, class_id]
        return activation_map


class BasicNetIndexFullImg(nn.Module):
    def __init__(self, h, w, dims=None, target_based=False, dropout=None, use_max_pool=True):
        super(BasicNetIndexFullImg, self).__init__()
        #
        self.cross_entropy = nn.CrossEntropyLoss()
        #
        if dims is None: dims = [3, 16, 32, 64]
        #
        self.dims = dims
        self.convs = Convolutions(dims, dropout=dropout, use_max_pool=use_max_pool)
        self.post_conv_shape = self.convs(torch.zeros(1, dims[0], h, w)).shape[1:]
        self.post_conv_dims = self.post_conv_shape[1:3]
        self.target_based = target_based
        self.n_loc = self.post_conv_dims[0] * self.post_conv_dims[1]
        self.n_classes = self.n_loc if not target_based else 2 # "target_based" means it's predicting whether or not a location is close to a target location
        self.h = h
        self.w = w
        self.final_conv = nn.Sequential(
            *([nn.Conv2d(in_channels=dims[-1], out_channels=dims[-1], kernel_size=1),
            nn.ReLU(),]
            + ([nn.Dropout(dropout)] if dropout is not None else [])
            + [nn.Conv2d(in_channels=dims[-1], out_channels=self.n_classes, kernel_size=1)])
        )
        #
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.softmax = nn.Softmax(dim=3) #used for generating activation maps
    
    def get_labels(self, c_y_norm=None, c_x_norm=None, r_norm=None):
        if self.target_based: return self.get_labels_target(c_y_norm, c_x_norm, r_norm)
        else: return self.get_labels_index()

    @staticmethod
    def in_target_range(c_y, c_x, i, j, r):
        return (c_y - r <= i <= c_y + r) and (c_x - r <= j <= c_x + r)

    def get_labels_target(self, c_y_norm, c_x_norm, r_norm):
        post_conv_h, post_conv_w = self.post_conv_dims
        labels = torch.zeros(post_conv_h, post_conv_w)
        c_y, c_x, r = post_conv_h * c_y_norm, post_conv_w * c_x_norm, (post_conv_h + post_conv_w) / 2 * r_norm
        for i in range(post_conv_h):
            for j in range(post_conv_w):
                if self.in_target_range(c_y, c_x, i, j, r):
                    labels[i, j] = 1
                else:
                    labels[i, j] = 0
        labels = labels.reshape((self.n_loc,))
        return labels.detach()
    
    def get_labels_index(self):
        labels = torch.arange(0, self.n_classes)
        labels = torch.reshape(labels, (self.n_loc,))
        return labels.detach()

    def get_classes_in_center(self, c_y_norm, c_x_norm, r_norm):
        classes = []
        post_conv_h, post_conv_w = self.post_conv_dims
        labels = torch.arange(0, self.n_loc).reshape(post_conv_h, post_conv_w)
        c_y, c_x, r = post_conv_h * c_y_norm, post_conv_w * c_x_norm, (post_conv_h + post_conv_w) / 2 * r_norm
        for i in range(post_conv_h):
            for j in range(post_conv_w):
                if self.in_target_range(c_y, c_x, i, j, r):
                    classes.append(labels[i, j])
        return classes

    def get_class_at_center(self, c_y_norm, c_x_norm):
        if self.target_based: return 1
        post_conv_h, post_conv_w = self.post_conv_dims
        c_y, c_x = post_conv_h * c_y_norm, post_conv_w * c_x_norm
        post_conv_h, post_conv_w = self.post_conv_dims
        labels = torch.arange(0, self.n_classes).reshape((post_conv_h, post_conv_w))
        return int(labels[int(c_y), int(c_x)].item())

    def conv_forward(self, x: Tensor):
        x = self.convs(x)
        x = self.final_conv(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward(self, x: Tensor):
        x = self.conv_forward(x)
        output: Tensor = self.log_softmax(x)
        output = output.reshape(-1, self.n_classes)
        return output

    def get_loss(self, output, targets):
        targets = targets.long()
        return self.cross_entropy(output, targets)

    def get_target_activation_map(self, frames, class_id):
        # returns: (n_frames, h, w)
        x: Tensor
        x = self.conv_forward(frames)
        x: Tensor = self.softmax(x)
        activation_map = x[:, :, :, class_id]
        return activation_map

    def get_target_activation_map_for_classes(self, frames, c_y_norm, c_x_norm, r_norm):
        assert not self.target_based
        # returns: (n_frames, h, w)
        classes = self.get_classes_in_center(c_y_norm, c_x_norm, r_norm)
        x: Tensor
        x = self.conv_forward(frames)
        x: Tensor = self.softmax(x)
        activation_map = x[:, :, :, classes].mean(3)
        return activation_map
    
    def get_activation_maps(self, x):
        return self.convs.get_activation_maps(x)


class BasicNetIndexFullImgMultiFrame(nn.Module):
    # Has a separate final projection layer for each frame
    def __init__(self, h, w, n_frames, dims=None, target_based=False, dropout=None, use_max_pool=True):
        super().__init__()
        #
        self.cross_entropy = nn.CrossEntropyLoss()
        #
        if dims is None: dims = [3, 16, 32, 64]
        #
        self.n_frames = n_frames
        self.dims = dims
        self.convs = Convolutions(dims, dropout=dropout, use_max_pool=use_max_pool)
        self.post_conv_shape = self.convs(torch.zeros(1, dims[0], h, w)).shape[1:]
        self.post_conv_dims = self.post_conv_shape[1:3]
        self.target_based = target_based
        self.n_loc = self.post_conv_dims[0] * self.post_conv_dims[1]
        self.n_classes = self.n_loc if not target_based else 2 # "target_based" means it's predicting whether or not a location is close to a target location
        self.h = h
        self.w = w
        self.pre_final_conv = nn.Sequential(
            *([nn.Conv2d(in_channels=dims[-1], out_channels=dims[-1], kernel_size=1),
            nn.ReLU(),]
            + ([nn.Dropout(dropout)] if dropout is not None else []))
        )
        self.final_convs = nn.ModuleList([nn.Conv2d(in_channels=dims[-1], out_channels=self.n_classes, kernel_size=1) for _ in range(n_frames)])
        #
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.softmax = nn.Softmax(dim=3) #used for generating activation maps
    
    def get_labels(self):
        labels = torch.arange(0, self.n_classes)
        labels = torch.reshape(labels, (self.n_loc,))
        return labels.detach()

    def conv_forward(self, x: Tensor, frame_num):
        final_conv = self.final_convs[frame_num]
        x = self.convs(x)
        x = self.pre_final_conv(x)
        x = final_conv(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward(self, x: Tensor, frame_num):
        x = self.conv_forward(x, frame_num)
        output: Tensor = self.log_softmax(x)
        output = output.reshape(-1, self.n_classes)
        return output

    def get_loss(self, output, targets):
        targets = targets.long()
        return self.cross_entropy(output, targets)
    
    def get_activation_maps(self, x):
        return self.convs.get_activation_maps(x)


ModelType = Union[BasicNetIndex, BasicNetCoord, BasicNetCoordCluster]    

class MyResnet(torchvision.models.ResNet):
    #https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    pass