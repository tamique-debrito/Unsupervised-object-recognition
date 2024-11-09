def temp(): # Re-extract the maps
    #return False
    source_video_filepath = "./multi-motion take 1.mp4"
    source_frame_num = 20
    resize_dim = 128
    n_frames = 512
    activation_map_model_filepath = "./tmp/data/activation_map_model"
    target_video_filepath = "./multi-motion take 2.mp4"
    params = AvgMapCreationParams(obj_info=get_info_obj_2(), single_map_prefix="./tmp/data/activ_map_b_", avg_map_filepath="./tmp/data/average_activ_map_b.npy")
    c_y_norm, c_x_norm, r_norm = params.obj_info
    component_params = get_component_params_list(source_video_filepath, source_frame_num, params.single_map_prefix, c_y_norm, c_x_norm, r_norm, resize_dim)
    target_map_path = "./tmp/data/target_activ_map_b.npy"
    model_paths = [c_params.model_filename for c_params in component_params]
    extract_ensembled_map(model_paths, target_video_filepath, target_map_path, resize_dim, n_frames)
    activation_map_model_filepath = "./tmp/data/activation_map_model"
    activ_map_model_frame_period = 8
    target_maps_filenames = ["./tmp/data/target_activ_map_a.npy", "./tmp/data/target_activ_map_b.npy"]
    show_heatmap_for_activ_map_model(activation_map_model_filepath, target_maps_filenames, activ_map_model_frame_period)
    return True