import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import ndimage


class fault_fold_augment():
    def __init__(self):
        self.curve_h, self.curve_x, self.curve_w = 1, 0, 1
        self.curve_rand = np.random.uniform(-0.5, 0.5)
        self.curve_tilt, self.curve_y = np.random.uniform(-1, 1), 1
        
        self.max_fault_dip, self.min_fault_dip = 86, 63
        self.max_fault_x, self.min_fault_x = 5, 2
        self.min_fault_slip, self.max_fault_slip = 2, 6
        self.fault_chance = np.random.uniform()
        self.fault_shift = np.random.choice([-1, 1]) * np.random.uniform(self.min_fault_x, self.max_fault_x)
        self.fault_slip = np.random.choice([-1, 1]) *\
                            np.random.uniform(self.min_fault_slip, self.max_fault_slip)

    def augment(self, seismic_batch, fault_label=False, fault_value=1, fill_value=-1):
        if len(seismic_batch.shape) == 3:
            seismic_batch = seismic_batch.unsqueeze(1) 
            added_channel_dim = True
        else:
            added_channel_dim = False
            
        batch_size, num_classes, height, width = seismic_batch.shape
        x, y = np.meshgrid(np.linspace(-10, 10, height), np.linspace(-10, 10, width))

        augmented_batch = torch.empty_like(seismic_batch)

        for i in range(batch_size):
            for n in range(num_classes):
                seismic_slice = seismic_batch[i, n].cpu().detach().numpy()

                fold_transform_small = self.curve_h * np.sin(
                    self.curve_x + self.curve_w * self.curve_rand * x
                )
                fold_transform_large = self.curve_tilt * x + self.curve_y
                fold_transform = fold_transform_small + fold_transform_large

                points = np.vstack((y.ravel(), x.ravel())).T
                seismic_slice_shifted = griddata(points, seismic_slice.ravel(), (y + fold_transform, x),
                                                 method='linear', 
                                                 fill_value=fill_value
                                                 ).reshape(y.shape)

                if self.fault_chance <= 0.7:
                    fault_line = x + self.fault_shift
                    seismic_slice_faulted = np.empty_like(seismic_slice)

                    for j in range(seismic_slice.shape[1]):
                        if fault_line[0, j] < 0:  # Pixel is on the right side of the fault
                            shift_amount = int(self.fault_slip)
                            seismic_slice_faulted[:, j] = np.roll(seismic_slice_shifted[:, j], shift_amount, axis=0)
                            if shift_amount > 0:
                                seismic_slice_faulted[:shift_amount, j] = fill_value
                            else:
                                seismic_slice_faulted[shift_amount:, j] = fill_value

                        else:  # Pixel is on the left side of the fault
                            shift_amount = -int(self.fault_slip)
                            seismic_slice_faulted[:, j] = np.roll(seismic_slice_shifted[:, j], shift_amount, axis=0)
                            if shift_amount > 0:
                                seismic_slice_faulted[:shift_amount, j] = fill_value
                            else:
                                seismic_slice_faulted[shift_amount:, j] = fill_value
                                
                    if fault_label:
                        if added_channel_dim:
                            assert n == 0
                            seismic_slice_faulted[np.abs(fault_line) == np.min(np.abs(fault_line))] = fault_value
                        elif n == fault_value:
                            seismic_slice_faulted[np.abs(fault_line) == np.min(np.abs(fault_line))] = 1    # probability, not for logit

                else:
                    seismic_slice_faulted = seismic_slice_shifted

                augmented_batch[i, n] = torch.tensor(seismic_slice_faulted, dtype=torch.float32)

        if added_channel_dim:
            augmented_batch = augmented_batch.squeeze(1)

        return augmented_batch
    
    
class enlarge_and_crop_augment():
    def __init__(self):
        self.enlarge_factor = np.random.uniform(1.1, 2.0)
        self.first_run = True
        
    def augment(self, array_batch):
        if len(array_batch.shape) == 3:
            array_batch = array_batch.unsqueeze(1) 
            added_channel_dim = True
        else:
            added_channel_dim = False
            
        batch_size, num_classes, height, width = array_batch.shape
        enlarged_height, enlarged_width = int(height * self.enlarge_factor), int(width * self.enlarge_factor)

        augmented_batch = torch.empty_like(array_batch)

        for i in range(batch_size):
            for n in range(num_classes):
                array_slice = array_batch[i, n].cpu().detach().numpy()

                enlarged_array_slice = ndimage.zoom(array_slice, self.enlarge_factor, order=0)

                if self.first_run:
                    self.crop_x = np.random.randint(0, enlarged_array_slice.shape[1] - width)
                    self.crop_y = np.random.randint(0, enlarged_array_slice.shape[0] - height)
                    self.first_run = False

                cropped_array_slice = enlarged_array_slice[self.crop_y:self.crop_y+height, self.crop_x:self.crop_x+width]

                augmented_batch[i, n] = torch.tensor(cropped_array_slice, dtype=torch.float32)
                
        if added_channel_dim:
            augmented_batch = augmented_batch.squeeze(1)

        return augmented_batch
    

class horizontal_flip_augment():
    def __init__(self):
        self.flip_chance = np.random.uniform()

    def augment(self, array_batch):
        if len(array_batch.shape) == 3:
            array_batch = array_batch.unsqueeze(1) 
            added_channel_dim = True
        else:
            added_channel_dim = False
            
        batch_size, num_classes, height, width = array_batch.shape

        augmented_batch = torch.empty_like(array_batch)

        for i in range(batch_size):
            for n in range(num_classes):
                array_slice = array_batch[i, n].cpu().detach().numpy()

                if self.flip_chance <= 0.5:
                    flipped_array_slice = np.fliplr(array_slice).copy()
                else:
                    flipped_array_slice = array_slice

                augmented_batch[i, n] = torch.tensor(flipped_array_slice, dtype=torch.float32)
                
        if added_channel_dim:
            augmented_batch = augmented_batch.squeeze(1)

        return augmented_batch
    
    
    
def visualize_augment(augmentation):
    reflect_arr = np.tile(np.random.uniform(-1, 1, size=(100, 1)), (1, 100))
    labels_arr = np.zeros((100, 100))
    
    reflect_arr[50:100] *= 2
    labels_arr[50:100] += 1
    
    reflect_arr[0:10, 90:100] = 1
    labels_arr[0:10, 90:100] = 2

    reflect_arr_batch = torch.tensor(reflect_arr[None, None, :, :])
    labels_arr_batch = torch.tensor(labels_arr[None, :, :])

    augmented_reflect_arr_batch = augmentation.augment(reflect_arr_batch)
    augmented_labels_arr_batch = augmentation.augment(labels_arr_batch)

    augmented_reflect_arr_slice = augmented_reflect_arr_batch[0, 0].cpu().detach().numpy()
    augmented_labels_arr_slice = augmented_labels_arr_batch[0].cpu().detach().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(reflect_arr, cmap='gray')
    axs[0, 0].set_title('Original')
    axs[0, 1].imshow(labels_arr, cmap='gray')
    axs[0, 1].set_title('Original Labels')
    axs[1, 0].imshow(augmented_reflect_arr_slice, cmap='gray')
    axs[1, 0].set_title('Aug Seismic')
    axs[1, 1].imshow(augmented_labels_arr_slice, cmap='gray')
    axs[1, 1].set_title('Aug Labels')
    plt.show()
    
    return None
    