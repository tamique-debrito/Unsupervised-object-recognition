# Takes the averaging/ensembling algorithm in signle_frame_no_patches and makes it so it trains and evals each model in a separate process and therefore doesn't run out of cuda memory

from dataclasses import dataclass
from typing import Optional

from numpy import save
from sympy import comp
from torch import Tensor
from single_frame_no_patches import *
from activation_map_training import get_activation_map_regression_model, get_heatmap_for_model, show_heatmap
from utils import run_in_new_process

@dataclass
class ComponentModelParams:
    # Params for the model that learns to recognize the object
    source_filename: str
    source_frame_num: int
    activation_map_filename: str # What file to write the component activation map to
    model_filename: str
    c_y_norm: float
    c_x_norm: float
    r_norm: float
    resize_dim: int
    n_steps: int
    net_dims: list[int]
    lr: float
    use_aug: bool
    dropout: Optional[float]
    weight: float

@dataclass
class AvgMapCreationParams:
    obj_info: tuple[float, float, float]
    single_map_prefix: str
    avg_map_filepath: str

def get_component_params_list(source_filename, source_frame_num, dest_filename_prefix, c_y_norm, c_x_norm, r_norm, resize_dim):
    def get_dest_map_filename():
        get_dest_map_filename.n += 1
        return dest_filename_prefix + "map_" + str(get_dest_map_filename.n) + ".npy"
    def get_dest_model_filename():
        return dest_filename_prefix + "model_" + str(get_dest_map_filename.n) + ".npy"
        
    get_dest_map_filename.n = 0
    return [
        ComponentModelParams(
            source_filename=source_filename, source_frame_num=source_frame_num,
            activation_map_filename=get_dest_map_filename(),
            model_filename=get_dest_model_filename(),
            c_y_norm=c_y_norm, c_x_norm=c_x_norm, r_norm=r_norm,
            resize_dim=resize_dim,
            net_dims=[3, 8, 8, 16, 16, 32, 32, 32, 64, 64],
            n_steps=2500, lr=0.001, use_aug=False, dropout=None, weight=0.8
        ),
        
        ComponentModelParams(
            source_filename=source_filename, source_frame_num=source_frame_num,
            activation_map_filename=get_dest_map_filename(),
            model_filename=get_dest_model_filename(),
            c_y_norm=c_y_norm, c_x_norm=c_x_norm, r_norm=r_norm,
            resize_dim=resize_dim,
            net_dims=[3, 8, 8, 16, 16, 32, 32, 32, 64, 64],
            n_steps=2500, lr=0.0005, use_aug=True, dropout=None, weight=0.7
        ),
        
        ComponentModelParams(
            source_filename=source_filename, source_frame_num=source_frame_num,
            activation_map_filename=get_dest_map_filename(),
            model_filename=get_dest_model_filename(),
            c_y_norm=c_y_norm, c_x_norm=c_x_norm, r_norm=r_norm,
            resize_dim=resize_dim,
            net_dims=[3, 8, 8, 16, 16, 32, 32, 32, 64, 64],
            n_steps=2500, lr=0.0002, use_aug=False, dropout=0.1, weight=0.9
        ),
        
        ComponentModelParams(
            source_filename=source_filename, source_frame_num=source_frame_num,
            activation_map_filename=get_dest_map_filename(),
            model_filename=get_dest_model_filename(),
            c_y_norm=c_y_norm, c_x_norm=c_x_norm, r_norm=r_norm,
            resize_dim=resize_dim,
            net_dims=[3, 8, 16, 32, 32, 64],
            n_steps=2500, lr=0.005, use_aug=False, dropout=None, weight=0.7
        ),
        
        ComponentModelParams(
            source_filename=source_filename, source_frame_num=source_frame_num,
            activation_map_filename=get_dest_map_filename(),
            model_filename=get_dest_model_filename(),
            c_y_norm=c_y_norm, c_x_norm=c_x_norm, r_norm=r_norm,
            resize_dim=resize_dim,
            net_dims=[3, 8, 16, 32, 32, 64],
            n_steps=2500, lr=0.0003, use_aug=False, dropout=None, weight=0.9
        ),
        
        ComponentModelParams(
            source_filename=source_filename, source_frame_num=source_frame_num,
            activation_map_filename=get_dest_map_filename(),
            model_filename=get_dest_model_filename(),
            c_y_norm=c_y_norm, c_x_norm=c_x_norm, r_norm=r_norm,
            resize_dim=resize_dim,
            net_dims=[3, 8, 8, 16, 16, 16, 32, 32, 64],
            n_steps=2500, lr=0.0005, use_aug=True, dropout=0.1, weight=0.5
        ),
        
        ComponentModelParams(
            source_filename=source_filename, source_frame_num=source_frame_num,
            activation_map_filename=get_dest_map_filename(),
            model_filename=get_dest_model_filename(),
            c_y_norm=c_y_norm, c_x_norm=c_x_norm, r_norm=r_norm,
            resize_dim=resize_dim,
            net_dims=[3, 8, 8, 16, 16, 16, 32, 32, 64, 64],
            n_steps=2500, lr=0.0001, use_aug=False, dropout=None, weight=0.99
        ),
        
        ComponentModelParams(
            source_filename=source_filename, source_frame_num=source_frame_num,
            activation_map_filename=get_dest_map_filename(),
            model_filename=get_dest_model_filename(),
            c_y_norm=c_y_norm, c_x_norm=c_x_norm, r_norm=r_norm,
            resize_dim=resize_dim,
            net_dims=[3, 8, 8, 16, 16, 32, 32, 32, 64, 64],
            n_steps=2500, lr=0.0005, use_aug=False, dropout=None, weight=0.9
        ),
        
        ComponentModelParams(
            source_filename=source_filename, source_frame_num=source_frame_num,
            activation_map_filename=get_dest_map_filename(),
            model_filename=get_dest_model_filename(),
            c_y_norm=c_y_norm, c_x_norm=c_x_norm, r_norm=r_norm,
            resize_dim=resize_dim,
            net_dims=[3, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256],
            n_steps=2500, lr=0.001, use_aug=True, dropout=0.5, weight=0.5
        ),
        
        ComponentModelParams(
            source_filename=source_filename, source_frame_num=source_frame_num,
            activation_map_filename=get_dest_map_filename(),
            model_filename=get_dest_model_filename(),
            c_y_norm=c_y_norm, c_x_norm=c_x_norm, r_norm=r_norm,
            resize_dim=resize_dim,
            net_dims=[3, 8, 8, 16, 16, 16, 32, 32, 64],
            n_steps=2500, lr=0.0005, use_aug=True, dropout=0.1, weight=0.8
        ),
        
        ComponentModelParams(
            source_filename=source_filename, source_frame_num=source_frame_num,
            activation_map_filename=get_dest_map_filename(),
            model_filename=get_dest_model_filename(),
            c_y_norm=c_y_norm, c_x_norm=c_x_norm, r_norm=r_norm,
            resize_dim=resize_dim,
            net_dims=[3, 8, 8, 16, 16, 16, 32, 32, 64, 64, 128],
            n_steps=2500, lr=0.001, use_aug=False, dropout=None, weight=0.99
        )
    ]

@dataclass
class PrintParams:
    major_steps: bool
    ensemble_steps: bool

def run_full_algorithm(print_params: PrintParams):
    # Trains two ensembles to track different objects in the first video and extracts activation maps for each object
    # Then trains a predictive model on the maps
    # Then extracts the activation maps for a second video (without training) and evaluates the predictive model on them

    if print_params.major_steps:
        print("######################## Started")
    
    source_video_filepath = "./multi-motion take 1.mp4"
    source_frame_num = 20
    resize_dim = 128
    n_frames = 128
    activation_map_model_filepath = "./tmp/data/activation_map_model"
    target_video_filepath = "./multi-motion take 2.mp4"

    ensemble_group_paths = [] # List of model ensemble filepaths, where each ensemble filepath is a list of individual model filepaths

    params_list = [
        AvgMapCreationParams(obj_info=get_info_obj_1(), single_map_prefix="./tmp/data/activ_map_a_", avg_map_filepath="./tmp/data/average_activ_map_a.npy"),
        AvgMapCreationParams(obj_info=get_info_obj_2(), single_map_prefix="./tmp/data/activ_map_b_", avg_map_filepath="./tmp/data/average_activ_map_b.npy")
    ]

    if print_params.major_steps:
        print("######################## Training Ensembles Started")
    for i, params in enumerate(params_list):
        if print_params.ensemble_steps:
            print(f"########### Training and Extracting for Ensemble {i+1}/{len(params_list)}")
        c_y_norm, c_x_norm, r_norm = params.obj_info
        component_model_parms_list = get_component_params_list(source_video_filepath, source_frame_num, params.single_map_prefix, c_y_norm, c_x_norm, r_norm, resize_dim)
        # component_model_parms_list = component_model_parms_list[:4]
        # for p in component_model_parms_list:
        #     p.n_steps = 250
        model_paths = train_and_extract_ensembled_activation_map(params, component_model_parms_list, source_video_filepath, source_frame_num, resize_dim, n_frames)
        ensemble_group_paths.append(model_paths)

    if print_params.major_steps:
        print("######################## Training Ensembles Finished")
        print("######################## Training Activation Map Model")
    activ_map_model_frame_period = 8
    activ_map_model_train_steps = 100
    activ_map_model_lr = 0.03
    run_in_new_process(train_activation_map_model, ([param.avg_map_filepath for param in params_list], activation_map_model_filepath, activ_map_model_frame_period, activ_map_model_train_steps, activ_map_model_lr))

    if print_params.major_steps:
        print("######################## Training Activation Map Model Finished")
        print("######################## Extracting Map for Target Video")

    target_maps_filenames = ["./tmp/data/target_activ_map_a.npy", "./tmp/data/target_activ_map_b.npy"]
    for i, (model_paths, target_map_path)  in enumerate(zip(ensemble_group_paths, target_maps_filenames)):
        if print_params.ensemble_steps:
            print(f"########### Extracting Target Map for Ensemble {i+1}/{len(ensemble_group_paths)}")
        extract_ensembled_map(model_paths, target_video_filepath, target_map_path, resize_dim, n_frames)

    if print_params.major_steps:
        print("######################## Maps Extracted")
        print("######################## Showing Heatmaps for Activation Map Model")

    get_heatmap_for_activ_map_model(activation_map_model_filepath, [param.avg_map_filepath for param in params_list], activ_map_model_frame_period)
    get_heatmap_for_activ_map_model(activation_map_model_filepath, target_maps_filenames, activ_map_model_frame_period)

def temp():
    return False
    load_and_play_map("./tmp/data/target_activ_map_b.npy")
    load_and_play_map("./tmp/data/target_activ_map_a.npy")
    return True
    source_avg_map_filepaths = ["./tmp/data/average_activ_map_a.npy", "./tmp/data/average_activ_map_b.npy"]
    target_avg_map_filepaths = ["./tmp/data/target_activ_map_a.npy", "./tmp/data/target_activ_map_b.npy"]
    ensemble_info = [(8, 100, 0.03), (8, 100, 0.03), (8, 100, 0.03), (8, 100, 0.03)] # frame period, train steps, lr
    ensemble_filepaths = [f"./tmp/data/avg_map_model_component_{i + 1}" for i in range(len(ensemble_info))]
    component_heatmap_filepaths = [f"./tmp/data/avg_heatmap_component_{i + 1}.npy" for i in range(len(ensemble_info))]
    averaged_heatmap_filepath = "./tmp/data/averaged_heatmap_b.npy"
    for model_filepath, info in zip(ensemble_filepaths, ensemble_info):
        period, steps, lr = info
        run_in_new_process(train_activation_map_model, (source_avg_map_filepaths, model_filepath, period, steps, lr))
        run_in_new_process(get_heatmap_for_activ_map_model, (model_filepath, source_avg_map_filepaths, period, True))
    
    for model_filepath, heatmap_filepath, info in zip(ensemble_filepaths, component_heatmap_filepaths, ensemble_info):
        period, _, _ = info
        run_in_new_process(get_heatmap_for_activ_map_model, (model_filepath, target_avg_map_filepaths, period, True, heatmap_filepath))
    avg_map = avg_maps(component_heatmap_filepaths, averaged_heatmap_filepath)
    show_heatmap(avg_map)
    return True

def get_heatmap_for_activ_map_model(model_path, maps_paths, frame_period, show=True, save_filepath=None):
    unformatted_frames = [np.load(filepath) for filepath in maps_paths]
    formatted_frames = format_multiple_activation_map_frames_for_model(unformatted_frames, frame_period)
    cuda_frames = Tensor(formatted_frames).to("cuda")
    model = torch.load(model_path, weights_only=False).to("cuda")
    p = get_heatmap_for_model(model, cuda_frames, show=show)
    if save_filepath is not None:
        np.save(save_filepath, p)

def train_activation_map_model(activation_map_filepaths, destination_model_filepath, frame_period, n_steps=150, lr=0.003):
    unformatted_frames = [np.load(filepath) for filepath in activation_map_filepaths]
    formatted_frames = format_multiple_activation_map_frames_for_model(unformatted_frames, frame_period)
    model = get_activation_map_regression_model(formatted_frames, n_steps, lr, False)
    torch.save(model, destination_model_filepath)

def train_and_extract_ensembled_activation_map(params: AvgMapCreationParams, component_model_parms_list: list[ComponentModelParams], source_video_filepath, source_frame_num, resize_dim, n_frames):
    # Trains multiple individual models with each of the specified parameters and then averages them to get an ensembled map, saving the result to a file
    model_num = 0
    all_map_paths = []
    all_model_paths = []
    for component_model_params in component_model_parms_list:
        model_num += 1
        print(f"Training map {model_num}")
        all_map_paths.append(component_model_params.activation_map_filename)
        all_model_paths.append(component_model_params.model_filename)
        
        run_in_new_process(target=train_and_extract_one_activation_map, args=(component_model_params, n_frames))
    
    avg_maps(all_map_paths, params.avg_map_filepath)
    return all_model_paths

def extract_ensembled_map(model_filepaths, video_filepath, destination_avg_map_filepath, resize_dim, n_frames):
    # Given trained models and a source video, will compute the activation maps for each model and average them together into one file
    single_map_paths = []
    for i, model_filepath in enumerate(model_filepaths):
        single_map_filepath = f"./tmp/data/temp_map_{i}.npy"
        run_in_new_process(target=extract_one_activation_map, args=(model_filepath, video_filepath, single_map_filepath, resize_dim, n_frames))
        single_map_paths.append(single_map_filepath)
    
    avg_maps(single_map_paths, destination_avg_map_filepath)

def avg_maps(source_files, dest_file):
    all_maps = [np.load(file) for file in source_files]
    # Preprocessing and validity checks
    shapes = [m.shape for m in all_maps]
    assert all([len(s) == len(shapes[0]) for s in shapes]), "all arryas should have same dims"
    truncated_shape = []
    for idx in range(len(shapes[0])):
        truncated_shape.append(min([s[idx] for s in shapes]))
    truncated_slice = tuple([slice(s) for s in truncated_shape])
    all_maps = [m[truncated_slice] for m in all_maps]
    # Do the average and save
    all_maps = np.array(all_maps)
    averaged_map = all_maps.mean(0)
    np.save(dest_file, averaged_map)
    return averaged_map

def load_and_play_map(filename):
    activ_map = np.load(filename)
    play_video(activation_map_to_rgb(activ_map), swap_dims=False, fps=3, name="Averaged map")

def extract_one_activation_map(model_filepath, video_filepath, destination_map_filepath, resize_dim, n_frames):
    model = torch.load(model_filepath, weights_only=False).to("cuda")
    model.eval()
    frames = load_video(video_filepath, n_frames, frame_skip=1, resize_to=(resize_dim, resize_dim)) / 256
    cuda_frames = torch.Tensor(frames).to("cuda")
    model.eval()
    act_map = model.get_activation_map(cuda_frames, 1).detach().cpu().numpy()
    np.save(destination_map_filepath, act_map)

def train_and_extract_one_activation_map(params: ComponentModelParams, n_frames):
    lr, use_aug, dropout, weight = params.lr, params.use_aug, params.dropout, params.weight
    steps_per_run = params.n_steps
    resize_dim = params.resize_dim
    frames = load_video(params.source_filename, n_frames, frame_skip=1, resize_to=(resize_dim, resize_dim)) / 256
    frame = frames[params.source_frame_num]
    cuda_frames = torch.Tensor(frames).to("cuda")
    h, w = frame.shape[1:]
    c_y_norm, c_x_norm, r_norm = params.c_y_norm, params.c_x_norm, params.r_norm
    model = get_frame_model(frame, h, w, c_y_norm, c_x_norm, r_norm, steps_per_run, lr=lr, use_aug=use_aug, show_heatmaps=False, dropout=dropout, use_max_pool=False, weight_for_class=weight)
    model.eval()
    act_map = model.get_target_activation_map(cuda_frames, 1).detach().cpu().numpy()
    np.save(params.activation_map_filename, act_map)
    torch.save(model, params.model_filename)

if __name__ == "__main__":
    if temp():
        print("temp")
    else:
        print("main")
        print_params = PrintParams(True, True)
        run_full_algorithm(print_params)
    #train_and_extract_multiple_activation_maps()