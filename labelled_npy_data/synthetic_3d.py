'''Module for generating 3D synthetic seismic data with different facies, salt, channels, folds, and faults.'''

import meanderpy as mp
import numpy as np
import matplotlib.pyplot as plt
import noise, datetime
from noise import snoise3
import scipy.ndimage
from scipy import signal
from scipy.special import erf
from scipy.interpolate import RegularGridInterpolator, griddata, interp1d
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter1d

### Suggested Class Labels
### Class 0: Channels
### Class 1: Faults
### Class 2: Background (Parallel Reflectors)
### Class 3: High-Amplitude Parallel Reflectors
### Class 4: Subparallel Reflectors 
### Class 5: Chaotic Reflectors
### Class 6: Salt
### Class 7: Coastal Onlaps
### Class 8: Parallel Onlaps


############ Initialization
def init_reflectivity_model(min_reflect, max_reflect, dimen_size):
    '''
    Initialize a random 3D horizontally layered reflectivity model.
    
    Input Parameters
    ----------------------------------------
    min_reflect: float
        minimum reflectance

    max_reflect: float
        maximum reflectance
        
    dimen_size: int
        shape of model cube output
        
    Output
    ----------------------------------------
    reflect_arr: 3D np.array
        3D reflectivity model
    '''
    reflect_arr = np.tile(np.random.uniform(min_reflect, max_reflect, size=(dimen_size, 1)), (1, dimen_size))
    reflect_arr = np.repeat(reflect_arr[:, :, np.newaxis], dimen_size, axis=2)   
    return reflect_arr


############ Channel Features
def generate_river_topo(nit = 2000,                   # number of iterations
                        W = 400.0,                    # channel width (m),
                        D = 6.0,                      # channel depth (m) 
                        pad = 100,                    # padding (number of nodepoints along centerline)
                        Cfs_w = 0.011,                # dimensionless Chezy friction factor weight
                        deltas = 50.0,                # sampling distance along centerline    
                        crdist_w = 2,                 # cutoff threshold distance weight
                        kl = 60.0/(365*24*60*60.0),   # migration rate constant (m/s)
                        kv =  1.0e-12,                # vertical slope-dependent erosion rate constant (m/s)
                        dt = 2*0.05*365*24*60*60.0,   # time step (s)
                        dens = 1000,                  # density of water (kg/m3)
                        saved_ts = 20,                # which time steps will be saved
                        n_bends = 30,                 # approximate number of bends you want to model
                        Sl = 0.0,                     # initial slope (matters more for submarine channels than rivers)
                        t1 = 500,                     # time step when incision starts
                        t2 = 700,                     # time step when lateral migration starts
                        t3 = 1200,                    # time step when aggradation starts
                        aggr_factor = 2e-9,           # aggradation factor (m/s, about 0.18 m/year, it kicks in after t3)
                        h_mud_w = 1.0,                # overbank deposit thickness weight
                        dx = 10.0,                    # gridcell size in meters
                        v_coarse = 10.0,              # deposition rate of coarse overbank sediment, in m/year (excluding times of no flooding)
                        v_fine = 0.0                  # deposition rate of fine overbank sediment, in m/year (excluding times of no flooding)
                       ):

    '''
    Wrapper function for generating 2D fluvial channel topography 
    array with timestep depth axis using the meanderpy package.
    Code is from the meanderpy repository python notebook showcase 
    (https://github.com/zsylvester/meanderpy).
    Output is transposed to match desired input orientation for 
    generating 3D seismic images (timeslice, xline, inline).
    '''
                        
    depths = D * np.ones((nit,))        # channel depths for different iterations 
    Cfs = Cfs_w * np.ones((nit,))       # dimensionless Chezy friction factor
    crdist = crdist_w * W               # threshold distance at which cutoffs occur

    ch = mp.generate_initial_channel(W, depths[0], Sl, deltas, pad, n_bends) # initialize channel
    chb = mp.ChannelBelt(channels=[ch], cutoffs=[], cl_times=[0.0], cutoff_times=[]) # create channel belt object

    chb.migrate(nit,saved_ts,deltas,pad,crdist,depths,Cfs,kl,kv,dt,dens,t1,t2,t3,aggr_factor) # channel migration

    h_mud = h_mud_w * np.ones((len(chb.channels),)) # thickness of overbank deposit for each time step
    diff_scale = 2.0 * W/dx
    
    # create 3d model
    chb_3d, xmin, xmax, ymin, ymax, dists, zmaps = mp.build_3d_model(chb, 'fluvial', 
            h_mud=h_mud, h=12.0, w=W, 
            bth=0.0, dcr=10.0, dx=dx, delta_s=deltas, dt=dt, starttime=chb.cl_times[0], endtime=chb.cl_times[-1],
            diff_scale=diff_scale, v_fine=v_fine, v_coarse=v_coarse, 
            xmin=9000, xmax=15000, ymin=-3500, ymax=3500)
    
    return np.flip(np.transpose(chb_3d.topo, (2, 1, 0)), axis=0)


def generate_3d_mask_from_2d(seismic_cube, mask_2d, shift_weight=3):
    '''
    Function to generate a 3D mask for the creation of channel features
    for a given seismic cube input, from an input 2D channel topography
    array. The depth in which the channel will be positioned within
    the seismic array is set to between 25-75% depth, to account for 
    folding/faulting deformations in the synthetic data generation workflow.
    
    Input Parameters
    -----------------------------------------
    sesimic_cube: 3D np.array
        The 3D reflectivity seismic array.
        
    mask_2d: 2D np.array
        A 2D channel topography slice from the chb_3d.topo
        3D array generated using meanderpy.
        
    shift_weight: float
        
    Output
    -----------------------------------------
    output_mask: 3D np.array
        A generated 3D channel mask array with same shape as
        the input seismic array, with 1 denoting channel facies,
        0 denoting background facies.
        
    start_layer: int
        The depth at which the channel is added within the seismic
        cube.
    '''
    output_mask = np.zeros_like(seismic_cube)
    height = seismic_cube.shape[0]

    mask_2d_shifted = (mask_2d - mask_2d.max()) * shift_weight

    start_layer = np.random.randint(height // 4, 3 * height // 4)

    for x in range(mask_2d_shifted.shape[0]):
        for y in range(mask_2d_shifted.shape[1]):
            layers_to_remove = -mask_2d_shifted[x, y]
            
            for layer in range(int(layers_to_remove)):
                target_layer = start_layer + layer
                if 0 <= target_layer < height:
                    output_mask[target_layer, x, y] = 1

    return output_mask, start_layer


############ Faults
def generate_faults(seismic_cubes, n_faults, fault_types=['listric', 'planar']):
    '''
    Function to apply same faulting to an input list of 3d numpy array cubes.
    
    Input Parameters
    -------------------------------------
    sesimic_cubes: list of np.array
        List of 3D np.array cubes to be applied faulting
        
    n_faults: int
        Number of faults to be generated. Note: more faults
        might appear in the output than expected as older 
        faults will be segmented by newer faults.
        
    fault_types: list of str
        Fault types to generate. Only options are listric 
        (faults with varying dip angle) and planar.
        
    Output
    -------------------------------------
    seismic_cubes_faulted: list of np.array
        Return list of faulted 3D np.array cubes
        
    fault_planes: 3D np.array
        3D array mask with same shape as the input arrays 
        denoting the location of fault planes. 0 denotes 
        background, 1 denotes fault.
    '''
    z, y, x = np.meshgrid(np.arange(seismic_cubes[0].shape[0]), 
                          np.arange(seismic_cubes[0].shape[1]), 
                          np.arange(seismic_cubes[0].shape[2]), 
                          indexing='ij')

    fault_planes = np.zeros_like(seismic_cubes[0], dtype=int)
    out_of_bounds_arr = np.zeros_like(seismic_cubes[0], dtype=int)
    seismic_cubes_faulted = [seismic_cube.copy() for seismic_cube in seismic_cubes]
    seismic_cubes_faulted.append(fault_planes)

    for _ in range(n_faults):
        fault_start_depth = np.random.randint(0, 300)
        fault_shift_x = np.random.randint(-100, 100)
        fault_shift_y = np.random.randint(-100, 100)
        min_fault_dip, max_fault_dip = 60, 80 
        min_fault_slip, max_fault_slip = 15, 30

        fault_dip_deg_x = np.random.uniform(min_fault_dip, max_fault_dip) * np.random.choice([-1, 1])
        fault_dip_deg_y = np.random.uniform(min_fault_dip, max_fault_dip) * np.random.choice([-1, 1])
        fault_dip_rad_x = np.radians(fault_dip_deg_x)
        fault_dip_rad_y = np.radians(fault_dip_deg_y)

        fault_slip = np.int(np.round(np.random.choice([-1, 1]) * np.random.uniform(min_fault_slip, max_fault_slip)))

        # Calculate fault line coordinates
        if np.random.choice(fault_types) == 'listric':
            listric_factor = 1 - z / seismic_cubes[0].shape[0] 
            fault_line = fault_start_depth + listric_factor * (np.tan(fault_dip_rad_x) * (x + fault_shift_x) +\
                                                               np.tan(fault_dip_rad_y) * (y + fault_shift_y))
        else:
            fault_line = fault_start_depth + (np.tan(fault_dip_rad_x) * (x + fault_shift_x) +\
                                              np.tan(fault_dip_rad_y) * (y + fault_shift_y))

        above_fault_mask = z > fault_line
        below_fault_mask = z < fault_line

        displacement_mask = above_fault_mask

        # Apply deformation
        for i, seismic_cube_faulted in enumerate(seismic_cubes_faulted):
            z_displace, y_displace, x_displace = z[displacement_mask], y[displacement_mask], x[displacement_mask]
            z_displace_new = z_displace + fault_slip
            z_displace_new = np.clip(z_displace_new, 0, seismic_cubes[0].shape[0] - 1)

            seismic_cube_faulted[z_displace, y_displace, x_displace] = seismic_cube_faulted[z_displace_new, y_displace, x_displace]
            
            out_of_bounds_mask = np.zeros_like(seismic_cube_faulted, dtype=bool)
            out_of_bounds_coords = (z_displace_new == 0) | (z_displace_new == seismic_cubes[0].shape[0] - 1)
            out_of_bounds_mask[z_displace[out_of_bounds_coords], y_displace[out_of_bounds_coords], x_displace[out_of_bounds_coords]] = True

            seismic_cube_faulted[out_of_bounds_mask] = 0
            out_of_bounds_arr[out_of_bounds_mask] = 1 
            
        fault_planes[np.abs(z - fault_line) < 3] = 1

    height_midpoint = seismic_cubes[0].shape[0] // 2
    out_of_bounds_arr[height_midpoint:, :, :] = np.where(out_of_bounds_arr[height_midpoint:, :, :] == 1, 
                                                         2, out_of_bounds_arr[height_midpoint:, :, :])

    return seismic_cubes_faulted[:-1], seismic_cubes_faulted[-1], out_of_bounds_arr


############ Folds
def generate_folding(seismic_cube, x_transform, y_transform, fold_2d=True):
    '''
    Input Parameters
    ----------------------------------
    seismic_cube: np.array
        The 3D seismic reflectance array
        
    x_transform: np.array
        Transformed x coordinates
    
    y_transform: np.array
        Transformed y coordinates
    
    fold_2d: boolean
        Enable two dimensional folding
        
    Output
    ----------------------------------
    seismic_cube: np.array
        Folded array
    '''
    dimen_size = seismic_cube.shape[0]
    x, y, z = np.mgrid[-10:10:dimen_size*1j, -10:10:dimen_size*1j, -10:10:dimen_size*1j]
    
    interpolator = RegularGridInterpolator((x[:, 0, 0], y[0, :, 0], z[0, 0, :]), 
                                           seismic_cube)
    seismic_cube = interpolator((x_transform, y_transform, z))
    
    if fold_2d:
        interpolator = RegularGridInterpolator((x[:, 0, 0], y[0, :, 0], z[0, 0, :]), 
                                               np.transpose(seismic_cube, (2, 0, 1)))
        seismic_cube = interpolator((x_transform, y_transform, z))
        seismic_cube = np.transpose(seismic_cube, (1, 0, 2))
    
    return seismic_cube


def smooth_edge_transformation_tanh(transformed_coord, original_coord, window_size=30):
    """
    Apply a smooth transformation to the edges using a tanh function,
    with a hard cutoff outside the original coordinate range to remove
    vertical artifacts from the original folding operation.
    Smoothing is applied only at the top and bottom edges.

    Input Parameters
    -----------------
    transformed_coord: np.array
        The transformed coordinates.

    original_coord: np.array
        The original coordinates.
        
    window_size: int
        The size of the window used for smoothing.

    Returns
    -------
    smoothed_coord: np.array
        The coordinates after applying the smooth edge transformation.
    """
    edge_dist = np.minimum(transformed_coord - original_coord.min(), original_coord.max() - transformed_coord)

    edge_dist = np.where((edge_dist != transformed_coord - original_coord.min()) & 
                         (edge_dist != original_coord.max() - transformed_coord), 
                         np.inf, edge_dist)

    edge_dist_perc = edge_dist / (original_coord.max() - original_coord.min())

    smooth_factor = 0.5 * (1 + np.tanh((edge_dist_perc - 0.5)))
    
    smoothed_coord = original_coord + (transformed_coord - original_coord) * smooth_factor

    smoothed_coord[transformed_coord < original_coord.min()] = 0
    smoothed_coord[transformed_coord > original_coord.max()] = 0
    
    # Detect where the change occurs
    change_indices = np.where(np.diff(smoothed_coord == 0))[0]
    
    # Vectorized smoothing with a uniform filter (windowed mean)
    for change_index in change_indices:
        smoothing_range = np.arange(max(0, change_index-window_size//2), min(len(smoothed_coord), change_index+window_size//2))
        smoothed_coord[smoothing_range] = uniform_filter1d(smoothed_coord[smoothing_range], size=window_size)

    return smoothed_coord


############ Facies
def generate_salt_domes(seismic_arrs, compression_ratio=0.5, salt_labels=[0, 4],
                        num_seeds=100, seed_value_range=(0.1, 0.3), 
                        shape_factor_range=(0.5, 1.5), sharpness_factor_range=(0.5, 2)):
    '''
    Function to introduce salt domes into the input reflectivity array.
    
    Input Parameters
    ------------------------------------------
    seismic_arrs: list of np.array
        List of input seismic arrays
        
    compression_ratio: float
    
    salt_reflect: list of int
        List of values assigned to salt. Default is [0, 4] for
        an expected input array list containing the reflectivity
        model and facies label arrays. 0 denotes the salt reflectivity,
        4 denotes the class label for salt.
        
    num_seeds: int
        Number of seeds to generate
        
    seed_value_range: tuple of float
        Controls size of salt domes
        
    shape_factor_range: tuple of float
        Controls irregularity of salt domes (more elliptical)
        
    sharpness_factor_range: tuple of float
        Controls sharpness of salt domes
        
    Output
    ----------------------------------------
    seismic_arr_compressed: np.array
        3d reflectivity array with salt domes bulging into 
        the original seismic array 
    '''
    height, width, length = seismic_arrs[0].shape
    for i, seismic_arr in enumerate(seismic_arrs):
        seismic_arr[height//2:] = salt_labels[i]
    x, y, z = np.arange(width), np.arange(height), np.arange(length)
    seismic_arrs_compressed = []

    distance_map = np.full((height, width, length), np.inf)

    # Ensure seeds originate near the bottom quarter
    seeds = np.random.rand(num_seeds, 3) * np.array([height/4, width, length])
    seeds[:, 0] += height * 3 / 4
    seed_values = np.random.uniform(seed_value_range[0], seed_value_range[1], num_seeds)

    shape_factors = np.random.uniform(shape_factor_range[0], shape_factor_range[1], num_seeds)
    sharpness_factors = np.random.uniform(sharpness_factor_range[0], sharpness_factor_range[1], num_seeds)

    # Calculate distance to nearest seed
    for i, (seed, seed_value, shape_factor, sharpness_factor) in enumerate(zip(seeds, seed_values, shape_factors, sharpness_factors)):
        distance_map = np.minimum(distance_map, ((x[None, :, None] * shape_factor - seed[1])**2 + 
                                                 (y[:, None, None] * shape_factor - seed[0])**2 + 
                                                 (z[None, None, :] * shape_factor - seed[2])**2)**(0.5 * sharpness_factor) * seed_value)

    distance_map = (distance_map - np.min(distance_map)) / (np.max(distance_map) - np.min(distance_map))

    compression = distance_map * compression_ratio

    new_indices = np.round(y[:, None, None] * (1 - compression)).astype(int)
    new_indices = np.clip(new_indices, 0, height - 1)

    for seismic_arr in seismic_arrs:
        seismic_arr_compressed = seismic_arr[new_indices, x[None, :, None], z[None, None, :]]
        seismic_arrs_compressed.append(seismic_arr_compressed)

    return seismic_arrs_compressed


def generate_coastal_onlap(seismic_arr):
    dimen_size = seismic_arr.shape[-1]
    height = seismic_arr.shape[0]
    reflect_arr = init_reflectivity_model(-1, 1, dimen_size)
    curve_h, curve_x, curve_w, curve_tiltx, curve_tilty, curve_z = 1, 0, 0.4, np.random.uniform(-1, 1), np.random.uniform(-1, 1), 1
    x, y, z = np.mgrid[-10:10:dimen_size*1j, -10:10:dimen_size*1j, -10:10:dimen_size*1j]
    
    fold_transform_small = erf(curve_w * (x - np.mean(x))) + erf(curve_w * (y - np.mean(y)))
    fold_transform_large = curve_tiltx * x + curve_tilty * y + curve_z
    fold_transform = fold_transform_small + fold_transform_large

    x_transformed = np.clip(x + fold_transform, x.min(), x.max())
    y_transformed = np.clip(y + fold_transform, x.min(), x.max())
    
    x_transformed = smooth_edge_transformation_tanh(x_transformed, x)
    y_transformed = smooth_edge_transformation_tanh(y_transformed, y)  
    
    reflect_arr_shifted = generate_folding(reflect_arr, x_transformed, y_transformed)
    
    return reflect_arr_shifted[(dimen_size-height)//2:(dimen_size+height)//2]


def generate_subparallel_terrain(seismic_arr, compression_ratio=0.5, scale=0.1, 
                                 octaves=1, persistence=0.5, lacunarity=2.0):
    height, width, length = seismic_arr.shape
    x, y, z = np.arange(height), np.arange(width), np.arange(length)

    noise_map = np.zeros((height, width, length))
    for i in x:
        for j in y:
            for k in z:
                noise_map[i, j, k] = snoise3(i * scale, j * scale, k * scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

    noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))

    compression = noise_map * compression_ratio

    new_indices = np.round(x[:, None, None] * (1 - compression)).astype(int)
    new_indices = np.clip(new_indices, 0, height - 1)

    seismic_arr_distorted = seismic_arr[new_indices, y[None, :, None], z[None, None, :]]

    return seismic_arr_distorted


def insert_facies_variation(seismic_cube, n_facies, facies_type, min_h=20, max_h=30):
    '''
    Introduce variation of seismic signature within the input seismic 
    reflectance array (choatic, uneven, onlap layers)
    
    Input Parameters
    -------------------------------------
    seismic_cube: np.array
        The seismic reflectance cube
        
    n_facies: int
        Number of facies to introduce
        
    min_facies_h: int
        Minimum height of facies layer
    
    Max_facies_h: int
        Maximum height of facies layer
        
    Output
    ------------------------------------
    seismic_cube: np.array
        The output seismic reflectance array with introduced facies variation
        
    facies_arr: np.array
        3D array with same shape as seismic_cube with labels of the different 
        facies.
    '''
    dimen_size = seismic_cube.shape[2]
    facies_arr = np.zeros((dimen_size, dimen_size, dimen_size))
    
    for facies in range(n_facies):
        facies_label = np.random.choice(facies_type)
        depth_limit = dimen_size - 50
        
        if facies_label == 4:
            min_facies_h = min_h + 50
            max_facies_h = max_h + 50
        else:
            max_facies_h, min_facies_h = max_h, min_h
            
        depth_1 = np.random.randint(0, depth_limit-max_facies_h+1)
        depth_2 = np.random.randint(depth_1+min_facies_h, min(depth_limit-1, depth_1+max_facies_h+1))
        facies_mask = (slice(depth_1,depth_2), slice(None), slice(None))
        
        if facies_label == 1:
            seismic_cube[facies_mask] = seismic_cube[facies_mask]*2
        
        elif facies_label == 2:
            seismic_cube[facies_mask] = generate_subparallel_terrain(seismic_cube[facies_mask])
            
        elif facies_label == 3:
            seismic_cube[facies_mask] = gaussian_filter(2*np.random.rand(depth_2-depth_1, dimen_size, dimen_size) - 1, 
                                                        sigma=np.random.uniform(0.5, 2)
                                                       )
            
        elif facies_label == 4:
            compressed_arrs = generate_salt_domes([seismic_cube[facies_mask], facies_arr[facies_mask]])
            seismic_cube[facies_mask] = compressed_arrs[0]
            facies_arr[facies_mask] = compressed_arrs[1]
        
        elif facies_label == 5:
            seismic_cube[facies_mask] = generate_coastal_onlap(seismic_cube[facies_mask])
            
        
        if facies_label != 4:
            facies_arr[facies_mask] = facies_label
        
    return seismic_cube, facies_arr


############ Miscellaneous
def ricker_3d_gen(points, a):
    """
    Generates a 3D Ricker wavelet.
    
    Input Parameters
    --------------------------------
    points: int
        Number of points in vector. Will be centered around 0.
    
    a: int
        Width parameter of the wavelet.
        
    Output
    --------------------------------
    ricker_3d: 3D np.array
        3D array of length points in shape of ricker curve.
    """
    ricker_1d = signal.ricker(points, a)
    ricker_3d = ricker_1d[:, np.newaxis, np.newaxis] *\
                ricker_1d[np.newaxis, :, np.newaxis] *\
                ricker_1d[np.newaxis, np.newaxis, :]
    return ricker_3d


def apply_depth_dependent_attenuation(seismic_cube, attenuation_factor=0.007):
    '''
    Introduce attenuation to reduce amplitude at lower depths.
    
    Input Parameters
    ---------------------------------------
    seismic_cube: np.array
        
    '''
    depth = np.arange(seismic_cube.shape[0])[:, np.newaxis, np.newaxis]
    return seismic_cube * np.exp(-depth * attenuation_factor)



############ Main Function
def generate_synthetic_3d(river_arr, dimen_size=300, n_faults=3, n_facies=3, n_rivers=2, facies_type=[2,3,4,5], folding=True):
    '''
    Input Parameters:
    -----------------------------------------------
    river_arr: 3D np.array
        topography 2D array with timestep z-axis generated from meanderpy (chb_3d.topo)
        
    dimen_size: int
        Size of seismic cube. Default is 300 for 300*300*300 cube.
        
    n_faults: int
        Number of faults to generate
        
    n_facies: int
        Number of facies variations to introduce. Note: Output may
        not contain less different facies layers than expected as
        the generated facies may be of the same label, and may be 
        on top of one another to form a thicker layer.
        
    folding: boolean
        
    Output
    -----------------------------------------------
    image: 3D np.array
        Synthetic Seismic Image
    
    reflect_arr_shifted: 3D np.array
        Reflectivity array (before ricker wavelet convolution)
        
    river_arr: 3D np.array
        3D river mask. 1 denotes channel facies, 0 denotes non-channel facies.
        
    faults_arr: 3D np.array
        3D fault mask. 1 denotes fault plane, 0 denotes background.
        
    facies_arr: 3D np.array
        3D facies mask. 0 denotes background facies, 1,2,3,etc. denotes
        additional facies added.
    '''
    tic = datetime.datetime.now()
    
    # Not suggested to use high absolute values (> 0.5) for curve_tiltx and curve_tilty if using the smooth transformation function
    curve_h, curve_x, curve_w, curve_tiltx, curve_tilty, curve_z = 1, 0, 0.2, np.random.uniform(-1, 1), np.random.uniform(-1, 1), 1
    min_reflect, max_reflect = -1, 1
    x, y, z = np.mgrid[-10:10:dimen_size*1j, -10:10:dimen_size*1j, -10:10:dimen_size*1j]
    
    print('Initializing parameters and reflectivity arrays ... ')
    print(f'|| curve_h: {curve_h} | curve_x: {curve_x} | curve_w: {curve_w} | curve_z: {curve_z} ||')
    print(f'|| curve_tiltx: {curve_tiltx} | curve_tilty: {curve_tilty} ||')
    
    reflect_arr = init_reflectivity_model(min_reflect, max_reflect, dimen_size)
    river_reflect_arr = init_reflectivity_model(min_reflect, max_reflect, dimen_size)
    reflect_arr_2 = init_reflectivity_model(min_reflect, max_reflect, dimen_size)
    
    print('Generating facies texture ... ')
    reflect_arr, facies_arr = insert_facies_variation(reflect_arr, n_facies, facies_type)
     
    print('Generating river mask ... ')
    river_mask_3d = np.zeros(reflect_arr.shape)
    for river in range(n_rivers):
        river_arr_slice_z = np.random.randint(0, river_arr.shape[0])
        single_river_mask_2d = river_arr[river_arr_slice_z,:,:]
        single_river_mask_3d, river_z = generate_3d_mask_from_2d(reflect_arr, single_river_mask_2d)
        print(f'|| river_arr_slice_z: {river_arr_slice_z} | river_z: {river_z} ||')
        river_mask_3d = np.logical_or(river_mask_3d, single_river_mask_3d).astype(int)
    river_nodes = river_mask_3d > 0
    non_river_nodes = river_mask_3d == 0
    river_field = distance_transform_edt(river_mask_3d)
    
    if n_rivers != 0:
        reflect_arr[river_nodes] = ((river_field[river_nodes] - np.mean(river_field[river_nodes])) / np.max(river_field) +\
                                    river_reflect_arr[river_nodes]) / 2


    print('Calculating folds ... ')
    fold_transform_small = curve_h * np.sin(curve_x + curve_w * x + curve_w * y)
    fold_transform_large = curve_tiltx * x + curve_tilty * y + curve_z
    asymmetry = np.random.uniform(0.5, 1.5) 
    fold_transform = fold_transform_small + fold_transform_large + asymmetry * np.sin(np.pi * z / dimen_size)

    x_transformed = np.clip(x + fold_transform, x.min(), x.max())
    y_transformed = np.clip(y + fold_transform, x.min(), x.max())
    
    fold_transform[river_nodes] = np.sin(fold_transform[river_nodes] * 1.1 * river_field[river_nodes])
    fold_transform[non_river_nodes] = 0
    
    x_transformed_river = np.clip(x + fold_transform, x.min(), x.max())
    y_transformed_river = np.clip(y + fold_transform, x.min(), x.max())
    
    x_transformed = smooth_edge_transformation_tanh(x_transformed, x)
    y_transformed = smooth_edge_transformation_tanh(y_transformed, y)
    
    print('Generating folds ...')
    reflect_arr_shifted = generate_folding(reflect_arr, x_transformed_river, y_transformed_river, fold_2d=False)
    
    if folding:
        reflect_arr_shifted = generate_folding(reflect_arr_shifted, x_transformed, y_transformed)
        river_mask_3d = generate_folding(river_mask_3d, x_transformed, y_transformed)
        facies_arr = generate_folding(facies_arr, x_transformed, y_transformed)
    
    print('Generating faults ... ')
    shifted_arrs, faults_arr, out_of_bounds_arr = generate_faults([reflect_arr_shifted, river_mask_3d, facies_arr], n_faults)
    reflect_arr_shifted = shifted_arrs[-3]
    river_mask_3d = shifted_arrs[-2]
    facies_arr = shifted_arrs[-1]
    
    print('Generating unconformity deposits ... ')
    out_of_bounds_mask = out_of_bounds_arr == 1
    reflect_arr_shifted[out_of_bounds_mask] = reflect_arr_2[out_of_bounds_mask]
    river_mask_3d[out_of_bounds_mask] = 0
    facies_arr[out_of_bounds_mask] = 6
    faults_arr[out_of_bounds_mask] = 0
    
    print('Applying Perlin noise ... ')
    scale_x, scale_y, scale_z = 0.05, 0.05, 0.05
    min_noise, max_noise = 0, 0.3
    for i in range(reflect_arr_shifted.shape[0]):
        for j in range(reflect_arr_shifted.shape[1]):
            for k in range(reflect_arr_shifted.shape[2]):
                reflect_arr_shifted[i, j, k] += np.random.uniform(min_noise, max_noise) *\
                snoise3(i * scale_x, j * scale_y, k * scale_z)

    print('Convolving using 3D Ricker wavelet ... ')
    ricker_wavelet_3d = ricker_3d_gen(10, 3)
    image = signal.convolve(reflect_arr_shifted, ricker_wavelet_3d, mode='same')
    image = signal.convolve(image, ricker_wavelet_3d, mode='same')
    
    image = apply_depth_dependent_attenuation(image)

    image += np.random.normal(scale=0.02, size=image.shape)
    
    toc = datetime.datetime.now()
    print(f'Done :) \n\nTime for image generation: {(toc - tic)}')

    return image, reflect_arr_shifted, river_mask_3d, faults_arr, facies_arr


def save_arrs_to_file(image, reflect_arr, river_arr, faults_arr, facies_arr):
    filename = 'syndata' + datetime.datetime.now().strftime("%Y%m%d_%H%M")
    np.savez(filename, image=image, reflect_arr=reflect_arr, river_arr=river_arr, faults_arr=faults_arr, facies_arr=facies_arr)
    return None


def compile_arrs(river_arr, faults_arr, facies_arr):
    assert river_arr.shape == faults_arr.shape == facies_arr.shape, 'Input arrays not in same shape :('
    combined_arr = np.array(facies_arr) + 2
    combined_arr[river_arr == 1] = 0
    combined_arr[faults_arr == 1] = 1
    return combined_arr