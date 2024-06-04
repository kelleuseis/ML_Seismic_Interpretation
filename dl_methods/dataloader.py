import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt 
from torch.utils.data import Dataset


def labels2distmaps(labels, resolution=None, dtype=None):
    assert labels.dtype == np.int

    K = np.max(labels) + 1

    if resolution is None:  # Same resolution in all dimensions
        resolution = (1,) * labels.ndim

    res = np.zeros((K,) + labels.shape, dtype=dtype)
    for k in range(K):
        mask = (labels == k)

        if mask.any():
            res[k] = distance_transform_edt(~mask, sampling=resolution)
            
    return res


class SeismicDatasetv2(Dataset):
    def __init__(self, seismic_data_path, labels_path, 
                 n_stack=3, relabel=False, compute_distmap=False,
                 train_size=0.7, train=True, 
                 padding=False, n_stack_padding=False,
                 cropped=True, random_crop=False,
                 patch_size=224, pad_value=-1, 
                 orientation=2, enforce_cube=False):
        
        '''
        Pytorch Seismic Dataset.
        
        Input Parameters
        ----------------------------------
        seismic_data_path: str
            Filepath containing seismic data array in 
            shape of (timeslice, xline, inline).
            
        labels_path: str
            Filepath containing labels array in shape 
            of (timeslice, xline, inline).
            
        n_stack: int
            For 2.5D input. Number of channels as model 
            input. Odd numbers only.
            
        relabel: boolean
            Relabel the labels array. For instances when
            the labels array does not start with 0 or when
            the label classes are incomplete.
            
        compute_distmap: boolean
            Compute distance map for boundary loss. Note 
            that __getitem__ will output 3 variables 
            (seismic, labels, distmap) instead of just 2
            if set to True.
            
        train_size: float
            Determines the split index for training and 
            validation set. Range from 0 to 1.
            
        train: boolean
            Set to True if initiating dataset for training.
            
        padding: boolean
            Enables padding to ensure the dataset height and
            width dimensions are divisible by patch size.
            
        n_stack_padding: boolean
            Enables padding in the depth dimension by 
            repeating the first and last slices of the 
            array in regards to chosen n_stack. If set to 
            false, __getitem__ will not start from the first
            slice (depending on chosen n_stack).
            
        cropped: boolean
            Enables cropping of the arrays to patch size 
            dimensions. Not to be used with padding.
            
        random_crop: boolean
            Enables random cropping. 
            
        patch_size: int
            Patch size (Deafult is 224).
            
        pad_value: float
            Set value for empty pixels during padding.
            
        orientation: int
            Determines the slice orientation {2: inline 
            (default); 1: xline; 0: timeslice}. 
            
        enforce_cube: boolean
            For enabling full dataset input. Makes sure all data 
            array dimensions are divisible by patch size. Overwrites
            both cropped and n_stack_padding options.
        '''
        
        self.seismic_data_path = seismic_data_path
        self.labels_path = labels_path

        self.seismic_data = np.load(seismic_data_path)
        if labels_path is None:
            self.labels = np.zeros(self.seismic_data.shape).astype(int)
        else:
            self.labels = np.load(labels_path).astype(int)
            
        assert self.labels.shape == self.seismic_data.shape, "Seismic array and labels not in the same shape!"
        assert len(self.seismic_data.shape) == 3, "Seismic array not in 3D"
        print(f'Input Shape: {self.seismic_data.shape}')
        
        self.patch_size = patch_size
        self.pad_value = pad_value
        self.n_buffer = n_stack//2
        self.n_stack_padding = n_stack_padding
        self.orientation = orientation
        
        splitidx = int(self.seismic_data.shape[2]*train_size)
        if train:
            self.seismic_data = self.seismic_data[:,:,:splitidx]
            self.labels = self.labels[:,:,:splitidx]
        else:
            self.seismic_data = self.seismic_data[:,:,splitidx:]
            self.labels = self.labels[:,:,splitidx:]
            
            
        if padding:
            self.padding = ((0, patch_size - self.seismic_data.shape[0] % patch_size),
                            (0, patch_size - self.seismic_data.shape[1] % patch_size), 
                            (0, 0))

            self.seismic_data = np.pad(self.seismic_data, self.padding, mode='constant', constant_values=self.pad_value)
            self.labels = np.pad(self.labels, self.padding, mode='constant', constant_values=self.pad_value)

        if cropped and not enforce_cube:
            if random_crop:
                start_h = np.random.randint(0, self.seismic_data.shape[0]-patch_size)
                start_w = np.random.randint(0, self.seismic_data.shape[1]-patch_size)
            else:
                start_h, start_w = 0, 0
            self.seismic_data = self.seismic_data[start_h:start_h+patch_size, start_w:start_w+patch_size, :]
            self.labels = self.labels[start_h:start_h+patch_size, start_w:start_w+patch_size, :]
            
        self.height, self.width, self.depth = self.seismic_data.shape
        
        if enforce_cube:
            self.height = (self.height // patch_size) * patch_size
            self.width = (self.width // patch_size) * patch_size
            self.depth = (self.depth // patch_size) * patch_size
            
            assert self.height >= 224 and self.width >= 224 and self.depth >= 224,\
            f"Not enough data for initializing {patch_size}*{patch_size}*{patch_size} cubes (may want to change the train size parameter)"

            self.seismic_data = self.seismic_data[:self.height, :self.width, :self.depth]
            self.labels = self.labels[:self.height, :self.width, :self.depth]
            
            self.n_stack_padding = True
        
        if orientation == 1:
            self.seismic_data = np.transpose(self.seismic_data, (0, 2, 1))
            self.labels = np.transpose(self.labels, (0, 2, 1))  
            self.height, self.width, self.depth = self.seismic_data.shape
        elif orientation == 0:
            self.seismic_data = np.transpose(self.seismic_data, (2, 1, 0))
            self.labels = np.transpose(self.labels, (2, 1, 0))    
            self.height, self.width, self.depth = self.seismic_data.shape
            
        if self.n_stack_padding:
            buffer_slices_front = np.repeat(self.seismic_data[:,:,0][:, :, np.newaxis], self.n_buffer, axis=2)
            buffer_slices_end = np.repeat(self.seismic_data[:,:,-1][:, :, np.newaxis], self.n_buffer, axis=2)
            self.seismic_data = np.concatenate((buffer_slices_front, self.seismic_data, buffer_slices_end), axis=2)
        else:
            self.depth -= self.n_buffer*2
        
        # Extract unique labels
        single_label_slice = self.labels[:, :, 0]
        self.unique_labels = np.unique(single_label_slice)
        self.relabel = relabel
        self.new_labels = {label: i for i, label in enumerate(self.unique_labels)}     

        # Normalization
        mean = np.mean(self.seismic_data)
        std = np.std(self.seismic_data)
        self.seismic_data = (self.seismic_data - mean) / std      
            
        self.compute_distmap = compute_distmap
        
        print(f'Output Shape: {(self.height, self.width, self.depth)} | n_buffer: {self.n_buffer} | Class List: {np.unique(self.labels)}')
        

    def __getitem__(self, index):
        depth = index % self.depth
        col = np.minimum(index // self.depth, (index // self.depth) % (self.width // 224))
        row = index // (self.depth * (self.width // 224))
            
        inline_slice = [torch.from_numpy(self.seismic_data[row*self.patch_size:row*self.patch_size+self.patch_size, 
                                                           col*self.patch_size:col*self.patch_size+self.patch_size, 
                                                           depth+self.n_buffer+i]) for i in range(-self.n_buffer, self.n_buffer+1)]
        
        inline_slice = torch.stack(inline_slice, dim=0)

        if not self.n_stack_padding:
            label_slice = torch.from_numpy(self.labels[row*self.patch_size:row*self.patch_size+self.patch_size,
                                                       col*self.patch_size:col*self.patch_size+self.patch_size,
                                                       depth+self.n_buffer])
        else: 
            label_slice = torch.from_numpy(self.labels[row*self.patch_size:row*self.patch_size+self.patch_size, 
                                                       col*self.patch_size:col*self.patch_size+self.patch_size, 
                                                       depth])
            
        if self.relabel:
            # Transform the labels using the dictionary mapping
            label_slice = np.vectorize(lambda x: self.new_labels.get(x, 0))(label_slice)
        
        if self.compute_distmap:
            label_slice_distmap = labels2distmaps(self.labels[:, :, index+self.n_buffer])
            return inline_slice, label_slice, label_slice_distmap
        else:
            return inline_slice, label_slice

    def __len__(self):
        return self.depth * (self.width//self.patch_size) * (self.height//self.patch_size)

    @property
    def num_classes(self):
        mclabel = np.max(self.unique_labels) + 1
        uqlabel = len(self.unique_labels)
        if mclabel != uqlabel:
            print(f'WARNING: Some labels are missing. Number of unique labels ({mclabel}) not equal to max class label ({uqlabel}). First label should be 0.')
            print(f'Current Class Labels (Without Relabelling): {self.unique_labels}')
            
        if self.relabel:
            return uqlabel
        else:
            return mclabel


def reconstruct_data(dataarr, dataset):
    '''
    Reshape model outputs/inputs into original dataset shape.
    For use after enforce_cube option in SeismicDatasetv2.
    
    Input Parameters
    ----------------------------------------
    dataarr: np.array or list
        Data to reshape
        
    dataset: SeismicDatasetv2
        Pytorch Seismic Dataset
        
    Output
    ----------------------------------------
    dataarr: np.array
        4D array (inline, channels, timeslice, xline) 
    '''
    if isinstance(dataarr, list):
        dataarr = np.stack(dataarr)
        
    print(f'Data Array Shape: {dataarr.shape}')
    dataarr = dataarr.reshape(dataset.height//dataset.patch_size, dataset.width//dataset.patch_size, dataset.depth, -1, dataset.patch_size, dataset.patch_size)
    dataarr = np.transpose(dataarr, (2, 3, 0, 4, 1, 5))
    dataarr = dataarr.reshape(dataset.depth, -1, dataset.height, dataset.width)
    
    if dataset.orientation == 1:
        dataarr = np.transpose(dataarr, (3, 1, 2, 0))
    elif dataset.orientation == 0:
        dataarr = np.transpose(dataarr, (2, 1, 0, 3))
        
    print(f'Final Data Array Shape: {dataarr.shape} | Original Dataset Shape: {dataset.labels.shape}')
        
    return dataarr


def fuse_segmentations(xline_logits, inline_logits, fault_logit_boost=0, channel_logit_boost=0, epsilon=1e-8):
    """
    Fuse segmentations from inline and xline logits.

    Input Parameters
    -------------------------------------
    xline_logits: np.array
        3D array of logits from crossline segmentation
        
    inline_logits: np.array
        3D array of logits from inline segmentation
        
    fault_logit_boost: float
        logit boost for fault features
    
    channel_logit_boost: float
        logit boost for channel features
        
    Output
    -------------------------------------
    fused_segmentation: np.array
        3D array of the fused segmentation (inline, timeslice, xline)
        
    combined_probs: np.array
        4D array of fused logits (inline, channels, timeslice, xline)
    """
    if isinstance(inline_logits, list):
        inline_logits = np.stack(inline_logits)
    if isinstance(xline_logits, list):
        xline_logits = np.transpose(np.stack(xline_logits), (3, 1, 2, 0))

    inline_logits[:, 1, :, :] += fault_logit_boost
    xline_logits[:, 1, :, :] += fault_logit_boost

    inline_logits[:, 0, :, :] += channel_logit_boost
    xline_logits[:, 0, :, :] += channel_logit_boost
    
    inline_probs = np.exp(inline_logits) / np.sum(np.exp(inline_logits), axis=1, keepdims=True)
    xline_probs = np.exp(xline_logits) / np.sum(np.exp(xline_logits), axis=1, keepdims=True)

    inline_confidence = np.max(inline_probs, axis=1)
    xline_confidence = np.max(xline_probs, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(xline_confidence.ravel(), bins=50, color='blue')
    plt.title('Crossline Confidence')
    plt.subplot(1, 2, 2)
    plt.hist(inline_confidence.ravel(), bins=50, color='red')
    plt.title('Inline Confidence')
    plt.tight_layout()
    plt.show()

    inline_confidence = inline_confidence[:, np.newaxis, :, :]
    xline_confidence = xline_confidence[:, np.newaxis, :, :]

    combined_probs = (inline_probs * inline_confidence + xline_probs * xline_confidence) / (inline_confidence + xline_confidence + epsilon)
    
    fused_segmentation = np.argmax(combined_probs, axis=1)

    return fused_segmentation, combined_probs
        
    
        
def plot_batch(dataloader, batch_idx):
    for i, data in enumerate(dataloader):
        if i == batch_idx:
            if len(data) == 3:
                images, labels, distmaps = data
            elif len(data) == 2:
                images, labels = data
            break
            
    print(images.shape, labels.shape)
    
    images = images[:,images.shape[1]//2,:,:]
    labels = labels.numpy()

    fig, axes = plt.subplots(1, images.shape[0], figsize=(16, 8))

    for ax, img, lbl in zip(axes, images, labels):
        ax.imshow(img, cmap='gray')
        ax.contour(lbl, colors='r', alpha=0.5)
        ax.axis('off')
    
    plt.show()
    
    return None
    
    

def plot_slice(dataset, slice_idx):
    batch = dataset[slice_idx]
    
    if len(batch) == 3:
        image, label, distmap = batch
    elif len(batch) == 2:
        image, label = batch
    
    print(image.shape, label.shape)
    
    image = image[image.shape[0]//2,:,:]

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.imshow(image, cmap='gray')
    axes.contour(label, colors='r', alpha=0.5) 
    axes.axis('off')
    
    plt.show()
    
    return None