import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import pandas as pd
from typing import Tuple, List, Dict

class WildfireDataset(Dataset):
    def __init__(self, 
                 image_dir: str,
                 metadata_path: str,
                 transform=None,
                 target_transform=None):
        """
        Args:
            image_dir (str): Directory with satellite images
            metadata_path (str): Path to CSV file with metadata (geolocation and labels)
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
        """
        self.image_dir = image_dir
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load image
        img_name = self.metadata.iloc[idx]['image_name']
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Get geolocation data
        lat = self.metadata.iloc[idx]['latitude']
        lon = self.metadata.iloc[idx]['longitude']
        geo_data = torch.tensor([lat, lon], dtype=torch.float32)
        
        # Get label
        label = self.metadata.iloc[idx]['risk_label']
        label = torch.tensor(label, dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, geo_data, label

def get_data_loaders(
    image_dir: str,
    metadata_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    transform=None
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        image_dir: Directory containing satellite images
        metadata_path: Path to metadata CSV file
        batch_size: Batch size for data loaders
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        transform: Optional transform to apply to images
        
    Returns:
        Dictionary containing train, validation, and test data loaders
    """
    # Create dataset
    dataset = WildfireDataset(
        image_dir=image_dir,
        metadata_path=metadata_path,
        transform=transform
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 