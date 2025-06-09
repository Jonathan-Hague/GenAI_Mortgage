import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torchvision.utils import make_grid

def plot_training_curves(
    train_metrics: List[Dict[str, float]],
    val_metrics: List[Dict[str, float]],
    save_path: str = None
):
    """
    Plot training and validation curves
    
    Args:
        train_metrics: List of training metrics per epoch
        val_metrics: List of validation metrics per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot VAE loss
    plt.subplot(1, 2, 1)
    plt.plot([m['vae_loss'] for m in train_metrics], label='Train VAE Loss')
    plt.plot([m['vae_loss'] for m in val_metrics], label='Val VAE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Loss')
    plt.legend()
    
    # Plot classifier loss
    plt.subplot(1, 2, 2)
    plt.plot([m['classifier_loss'] for m in train_metrics], label='Train Classifier Loss')
    plt.plot([m['classifier_loss'] for m in val_metrics], label='Val Classifier Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Classifier Loss')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(
    predictions: List[int],
    labels: List[int],
    save_path: str = None
):
    """
    Plot confusion matrix
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_risk_distribution(
    risk_scores: List[float],
    save_path: str = None
):
    """
    Plot distribution of risk scores
    
    Args:
        risk_scores: List of risk scores
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(risk_scores, bins=50)
    plt.xlabel('Risk Score')
    plt.ylabel('Count')
    plt.title('Distribution of Risk Scores')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_reconstructions(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    num_images: int = 8,
    save_path: str = None
):
    """
    Visualize original and reconstructed images
    
    Args:
        original_images: Batch of original images
        reconstructed_images: Batch of reconstructed images
        num_images: Number of images to display
        save_path: Path to save the plot
    """
    # Select random images
    indices = np.random.choice(len(original_images), num_images, replace=False)
    original = original_images[indices]
    reconstructed = reconstructed_images[indices]
    
    # Create grid of images
    original_grid = make_grid(original, nrow=4, normalize=True)
    reconstructed_grid = make_grid(reconstructed, nrow=4, normalize=True)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_grid.permute(1, 2, 0).cpu())
    plt.title('Original Images')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_grid.permute(1, 2, 0).cpu())
    plt.title('Reconstructed Images')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_geographic_risk(
    latitudes: List[float],
    longitudes: List[float],
    risk_scores: List[float],
    save_path: str = None
):
    """
    Plot risk scores on a geographic map
    
    Args:
        latitudes: List of latitude values
        longitudes: List of longitude values
        risk_scores: List of risk scores
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        longitudes,
        latitudes,
        c=risk_scores,
        cmap='RdYlBu_r',
        alpha=0.6
    )
    plt.colorbar(scatter, label='Risk Score')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution of Wildfire Risk')
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 