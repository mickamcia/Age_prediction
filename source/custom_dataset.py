import torch
import numpy as np
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        #image = (image-np.min(image))/(np.max(image)-np.min(image))
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        return image, label

def class_labels_reassign(age):
    return age

