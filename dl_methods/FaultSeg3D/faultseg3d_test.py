'''
Module for easier testing of FaultSeg3D. 
Functions are modified from code from the FaultSeg3D repository (https://github.com/xinwucwp/faultSeg)
'''

import segyio
import numpy as np
import matplotlib.pyplot as plt

def extract_cube_from_segy(filename, cube_size, random_crop=False):
    """Extract a cube of data of specified size from a SEGY file"""
    with segyio.open(filename, 'r') as f:
        min_size = min(f.ilines.size, f.xlines.size, len(f.samples))

        if min_size < cube_size:
            print(f"Warning: Requested cube size ({cube_size}) is larger than the input size. "
                  f"Using max available size {min_size} and padding with average value.")

            max_ilines = f.ilines.size
            max_xlines = f.xlines.size
            max_samples = len(f.samples)
        else:
            max_ilines = f.ilines.size - cube_size
            max_xlines = f.xlines.size - cube_size
            max_samples = len(f.samples) - cube_size

        random_ilines = 0 if not random_crop else random.randint(0, max_ilines)
        random_xlines = 0 if not random_crop else random.randint(0, max_xlines)
        random_sample = 0 if not random_crop else random.randint(0, max_samples)

        cube = np.empty((cube_size, cube_size, cube_size), dtype=np.single)
        
        for i in range(min(cube_size, max_ilines - random_ilines)):
            for j in range(min(cube_size, max_xlines - random_xlines)):
                end_sample = min(random_sample + cube_size, max_samples)
                cube_data = f.trace[
                    (random_ilines + i) * f.xlines.size + random_xlines + j
                ][random_sample:end_sample]
                
                if cube_data.size < cube_size:
                    avg_val = np.average(cube_data)
                    cube_data = np.pad(cube_data, 
                                       (0, cube_size - cube_data.size), 
                                       'constant', 
                                       constant_values=(avg_val, avg_val))
                
                cube[i, j, :] = cube_data

        avg_cube_val = np.average(cube)
        cube = np.pad(cube, 
                      (
                          (0, cube_size - cube.shape[0]), 
                          (0, cube_size - cube.shape[1]), 
                          (0, 0)
                      ), 
                      'constant', 
                      constant_values=(avg_cube_val, avg_cube_val))

    return cube


def getMask(overlap_width, model_h=128, model_w=128, model_d=128):
    weights_mask = np.zeros((model_h, model_w, model_d),dtype=np.single)
    weights_mask = weights_mask + 1
    gaussian_weights = np.zeros((overlap_width),dtype=np.single)
    sig = overlap_width / 4
    sig = 0.5 / (sig*sig)
    
    for voxel in range(overlap_width):
        voxel_dist = voxel - overlap_width + 1
        gaussian_weights[voxel] = np.exp(-voxel_dist * voxel_dist * sig)
        
    for i in range(overlap_width):
        for j in range(model_w):
            for k in range(model_d):
                weights_mask[i][j][k] = gaussian_weights[i]
                weights_mask[model_h-i-1][j][k] = gaussian_weights[i]
    for i in range(model_h):
        for j in range(overlap_width):
            for k in range(model_d):
                weights_mask[i][j][k] = gaussian_weights[j]
                weights_mask[i][model_w-j-1][k] = gaussian_weights[j]
    for i in range(model_h):
        for j in range(model_w):
            for k in range(overlap_width):
                weights_mask[i][j][k] = gaussian_weights[k]
                weights_mask[i][j][model_d-k-1] = gaussian_weights[k]
                
    return weights_mask


def faultseg_run(input_path, output_path, loaded_model, image_h, image_w, image_d, model_h=128, model_w=128, model_d=128):
    overlap_width = 12  # overlap width

    if image_h % model_h != 0 or image_w % model_w != 0 or image_d % model_d != 0:
        raise ValueError(f"Each dimension should be a multiple of {model_h}")

    image_arr = np.fromfile(input_path, dtype=np.single)
    image_arr = np.reshape(image_arr, (image_h, image_w, image_d))
    
    n_chunks_h = int(np.round((image_h+overlap_width) / (model_h-overlap_width) + 0.5))
    n_chunks_w = int(np.round((image_w+overlap_width) / (model_w-overlap_width) + 0.5))
    n_chunks_d = int(np.round((image_d+overlap_width) / (model_d-overlap_width) + 0.5))
    
    padded_h = (model_h-overlap_width) * n_chunks_h + overlap_width
    padded_w = (model_w-overlap_width) * n_chunks_w + overlap_width
    padded_d = (model_d-overlap_width) * n_chunks_d + overlap_width
    
    padded_image = np.zeros((padded_h, padded_w, padded_d), dtype=np.single)
    pred_arr = np.zeros((padded_h, padded_w, padded_d), dtype=np.single)
    overlap_mask = np.zeros((padded_h, padded_w, padded_d), dtype=np.single)
    input_arr = np.zeros((1, model_h, model_w, model_d, 1), dtype=np.single)

    padded_image[0:image_h, 0:image_w, 0:image_d] = image_arr
    mask = getMask(overlap_width)

    for i in range(n_chunks_h):
        for j in range(n_chunks_w):
            for k in range(n_chunks_d):
                start_idx_i = i*model_h - i*overlap_width
                end_idx_i = start_idx_i + model_h
                
                start_idx_j = j*model_w - j*overlap_width
                end_idx_j = start_idx_j + model_w
                
                start_idx_k = k*model_d - k*overlap_width
                end_idx_k = start_idx_k + model_d
                
                input_arr[0, :, :, :, 0] = padded_image[start_idx_i:end_idx_i, start_idx_j:end_idx_j, start_idx_k:end_idx_k]
                input_arr = input_arr - np.min(input_arr)
                input_arr = input_arr / np.max(input_arr)
                input_arr = input_arr * 255

                pred = loaded_model.predict(input_arr, verbose=1)
                pred = np.array(pred)
                pred_arr[start_idx_i:end_idx_i, start_idx_j:end_idx_j, start_idx_k:end_idx_k] += pred[0, :, :, :, 0] * mask
                overlap_mask[start_idx_i:end_idx_i, start_idx_j:end_idx_j, start_idx_k:end_idx_k] += mask

    pred_arr = pred_arr / overlap_mask
    pred_arr = pred_arr[0:image_h, 0:image_w, 0:image_d]
    pred_arr.tofile(output_path, format="%4")
    
    return None


def faultseg_plot_slices(image_path, pred_path, plot_x, plot_i, plot_t, image_h, image_w, image_d):
    image_arr = np.fromfile(image_path, dtype=np.single)
    pred_arr = np.fromfile(pred_path, dtype=np.single)
    image_arr = np.reshape(image_arr, (image_h, image_w, image_d))
    pred_arr = np.reshape(pred_arr, (image_h, image_w, image_d))
    
    image_x = np.transpose(image_arr[plot_x, :, :])
    pred_x = np.transpose(pred_arr[plot_x, :, :])
    image_i = np.transpose(image_arr[:, plot_i, :])
    pred_i = np.transpose(pred_arr[:, plot_i, :])
    image_t = np.transpose(image_arr[:, :, plot_t])
    pred_t = np.transpose(pred_arr[:, :, plot_t])

    fig = plt.figure(figsize=(9, 9))
    p1 = plt.subplot(2, 1, 1)
    p1.imshow(image_x, aspect='auto', cmap='seismic')
    p2 = plt.subplot(2, 1, 2)
    p2.imshow(pred_x, aspect='auto', interpolation="bilinear", vmin=0.4, vmax=1.0, cmap=plt.cm.gray)

    fig = plt.figure(figsize=(12, 12))
    p1 = plt.subplot(2, 1, 1)
    p1.imshow(image_i, aspect='auto', cmap='seismic')
    p2 = plt.subplot(2, 1, 2)
    p2.imshow(pred_i, aspect='auto', interpolation="bilinear", vmin=0.4, vmax=1.0, cmap=plt.cm.gray)

    fig = plt.figure(figsize=(12, 12))
    p1 = plt.subplot(2, 1, 1)
    p1.imshow(image_t, aspect='auto', cmap='seismic')
    p2 = plt.subplot(2, 1, 2)
    p2.imshow(pred_t, aspect='auto', interpolation="bilinear", vmin=0.4, vmax=1.0, cmap=plt.cm.gray)

    plt.show()
    
    return image_arr, pred_arr