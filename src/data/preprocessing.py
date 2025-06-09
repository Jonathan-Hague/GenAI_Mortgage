import torch
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List
import cv2

def get_image_transforms(
    image_size: Tuple[int, int] = (64, 64),
    normalize: bool = True
) -> transforms.Compose:
    """
    Get standard image transforms for satellite images
    
    Args:
        image_size: Target size for images (height, width)
        normalize: Whether to normalize images
        
    Returns:
        Composition of transforms
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
    
    if normalize:
        # Use ImageNet normalization as a starting point
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return transforms.Compose(transform_list)

def normalize_geolocation(
    lat: float,
    lon: float,
    lat_range: Tuple[float, float] = (-90, 90),
    lon_range: Tuple[float, float] = (-180, 180)
) -> Tuple[float, float]:
    """
    Normalize geolocation coordinates to [0, 1] range
    
    Args:
        lat: Latitude
        lon: Longitude
        lat_range: Range of latitude values
        lon_range: Range of longitude values
        
    Returns:
        Normalized (latitude, longitude) tuple
    """
    norm_lat = (lat - lat_range[0]) / (lat_range[1] - lat_range[0])
    norm_lon = (lon - lon_range[0]) / (lon_range[1] - lon_range[0])
    return norm_lat, norm_lon

def denormalize_geolocation(
    norm_lat: float,
    norm_lon: float,
    lat_range: Tuple[float, float] = (-90, 90),
    lon_range: Tuple[float, float] = (-180, 180)
) -> Tuple[float, float]:
    """
    Denormalize geolocation coordinates from [0, 1] range
    
    Args:
        norm_lat: Normalized latitude
        norm_lon: Normalized longitude
        lat_range: Range of latitude values
        lon_range: Range of longitude values
        
    Returns:
        Denormalized (latitude, longitude) tuple
    """
    lat = norm_lat * (lat_range[1] - lat_range[0]) + lat_range[0]
    lon = norm_lon * (lon_range[1] - lon_range[0]) + lon_range[0]
    return lat, lon

def preprocess_satellite_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess satellite image for model input
    
    Args:
        image: Input image as numpy array
        target_size: Target size for image (height, width)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed image
    """
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Convert to float and normalize if requested
    image = image.astype(np.float32)
    if normalize:
        image = image / 255.0
        
    return image

def create_data_augmentation() -> transforms.Compose:
    """
    Create data augmentation transforms for training
    
    Returns:
        Composition of augmentation transforms
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        )
    ]) 