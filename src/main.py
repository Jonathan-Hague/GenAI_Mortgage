import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from typing import Dict, Any
import yaml
import logging
from tqdm import tqdm

from models.vae import VAE
from models.classifier import WildfireRiskClassifier
from data.data_loader import get_data_loaders
from data.preprocessing import get_image_transforms, create_data_augmentation
from utils.metrics import calculate_metrics
from utils.visualization import plot_training_curves

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE-Logit model for wildfire risk assessment')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                      help='Mode to run the script in')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(output_dir: str):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def train_epoch(
    vae: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    vae_optimizer: optim.Optimizer,
    classifier_optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch"""
    vae.train()
    classifier.train()
    
    total_vae_loss = 0
    total_classifier_loss = 0
    total_samples = 0
    
    for images, geo_data, labels in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        geo_data = geo_data.to(device)
        labels = labels.to(device)
        
        # VAE forward pass
        vae_optimizer.zero_grad()
        recon_images, mu, log_var = vae(images)
        vae_loss, recon_loss, kl_loss = vae.loss_function(recon_images, images, mu, log_var)
        vae_loss.backward()
        vae_optimizer.step()
        
        # Classifier forward pass
        classifier_optimizer.zero_grad()
        with torch.no_grad():
            vae_features = vae.encode(images)[0]  # Get mu from VAE
        logits = classifier(vae_features, geo_data)
        classifier_loss = classifier.loss_function(logits, labels)
        classifier_loss.backward()
        classifier_optimizer.step()
        
        # Update statistics
        batch_size = images.size(0)
        total_vae_loss += vae_loss.item() * batch_size
        total_classifier_loss += classifier_loss.item() * batch_size
        total_samples += batch_size
    
    return {
        'vae_loss': total_vae_loss / total_samples,
        'classifier_loss': total_classifier_loss / total_samples
    }

def evaluate(
    vae: nn.Module,
    classifier: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model performance"""
    vae.eval()
    classifier.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, geo_data, labels in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            geo_data = geo_data.to(device)
            
            # Get VAE features
            vae_features = vae.encode(images)[0]
            
            # Get classifier predictions
            logits = classifier(vae_features, geo_data)
            predictions = classifier.predict(vae_features, geo_data)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_labels)
    return metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(args.output_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Get data transforms
    train_transform = get_image_transforms(
        image_size=config['data']['image_size'],
        normalize=True
    )
    train_transform = transforms.Compose([
        create_data_augmentation(),
        train_transform
    ])
    
    val_transform = get_image_transforms(
        image_size=config['data']['image_size'],
        normalize=True
    )
    
    # Get data loaders
    data_loaders = get_data_loaders(
        image_dir=os.path.join(args.data_dir, 'images'),
        metadata_path=os.path.join(args.data_dir, 'metadata.csv'),
        batch_size=config['training']['batch_size'],
        transform=train_transform
    )
    
    # Initialize models
    vae = VAE(
        input_channels=3,
        latent_dim=config['model']['vae']['latent_dim'],
        hidden_dims=config['model']['vae']['hidden_dims']
    ).to(device)
    
    classifier = WildfireRiskClassifier(
        latent_dim=config['model']['vae']['latent_dim'],
        geo_dim=2,
        hidden_dim=config['model']['classifier']['hidden_dim'],
        num_classes=config['model']['classifier']['num_classes']
    ).to(device)
    
    # Initialize optimizers
    vae_optimizer = optim.Adam(
        vae.parameters(),
        lr=config['training']['vae_lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    classifier_optimizer = optim.Adam(
        classifier.parameters(),
        lr=config['training']['classifier_lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    if args.mode == 'train':
        # Training loop
        best_val_accuracy = 0
        train_metrics = []
        val_metrics = []
        
        for epoch in range(config['training']['num_epochs']):
            logging.info(f'Epoch {epoch+1}/{config["training"]["num_epochs"]}')
            
            # Train
            train_epoch_metrics = train_epoch(
                vae, classifier,
                data_loaders['train'],
                vae_optimizer,
                classifier_optimizer,
                device
            )
            
            # Evaluate
            val_epoch_metrics = evaluate(
                vae, classifier,
                data_loaders['val'],
                device
            )
            
            # Log metrics
            logging.info(f'Train metrics: {train_epoch_metrics}')
            logging.info(f'Validation metrics: {val_epoch_metrics}')
            
            # Save best model
            if val_epoch_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_epoch_metrics['accuracy']
                torch.save({
                    'vae_state_dict': vae.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                    'classifier_optimizer_state_dict': classifier_optimizer.state_dict(),
                    'epoch': epoch,
                    'val_metrics': val_epoch_metrics
                }, os.path.join(args.output_dir, 'best_model.pth'))
            
            # Save training curves
            train_metrics.append(train_epoch_metrics)
            val_metrics.append(val_epoch_metrics)
            plot_training_curves(
                train_metrics,
                val_metrics,
                save_path=os.path.join(args.output_dir, 'training_curves.png')
            )
    
    elif args.mode == 'evaluate':
        # Load best model
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
        vae.load_state_dict(checkpoint['vae_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        # Evaluate on test set
        test_metrics = evaluate(
            vae, classifier,
            data_loaders['test'],
            device
        )
        
        logging.info(f'Test metrics: {test_metrics}')

if __name__ == '__main__':
    main() 