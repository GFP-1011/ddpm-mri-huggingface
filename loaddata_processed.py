import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NpzDataset(Dataset):
    """
    Custom Dataset to load .npz files for PyTorch.
    """

    def __init__(self, folder_path, transform=None):
        """
        Initialize the dataset.

        Parameters:
        - folder_path: str, path to the folder containing .npz files.
        - transform: callable, optional transformations to apply to the data (default: None).
        """
        self.folder_path = folder_path
        self.file_names = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".npz")],
            key=lambda x: int(x.split('.')[0])  # Sort by numeric order
        )
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Load an image and label pair from the .npz file.
        """
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        data = np.load(file_path)
        image = data["image"].astype(np.float32)  # Convert to float32 for PyTorch
        label = data["label"].astype(np.float32)  # Convert to int64 for classification

        # Apply optional transforms
        if self.transform:
            image = self.transform(image)

        # Convert to PyTorch tensors
        image_tensor = torch.tensor(image)
        label_tensor = torch.tensor(label)


        return image_tensor, label_tensor


# Example usage:
def create_dataloader(folder_path, batch_size=8, shuffle=True, num_workers=0):
    """
    Create a PyTorch DataLoader for the .npz dataset.

    Parameters:
    - folder_path: str, path to the folder containing .npz files.
    - batch_size: int, number of samples per batch.
    - shuffle: bool, whether to shuffle the data.
    - num_workers: int, number of subprocesses to use for data loading.

    Returns:
    - dataloader: PyTorch DataLoader.
    """
    dataset = NpzDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
