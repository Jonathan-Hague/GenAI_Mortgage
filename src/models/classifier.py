import torch
import torch.nn as nn
import torch.nn.functional as F

class WildfireRiskClassifier(nn.Module):
    def __init__(self, latent_dim, geo_dim=2, hidden_dim=64, num_classes=2):
        super(WildfireRiskClassifier, self).__init__()
        
        # Combine VAE latent features with geolocation data
        self.input_dim = latent_dim + geo_dim
        
        # Neural network layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, vae_features, geo_features):
        # Concatenate VAE features with geolocation data
        x = torch.cat([vae_features, geo_features], dim=1)
        
        # Forward pass through the network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def predict_proba(self, vae_features, geo_features):
        """Get probability predictions"""
        with torch.no_grad():
            logits = self.forward(vae_features, geo_features)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, vae_features, geo_features):
        """Get class predictions"""
        probs = self.predict_proba(vae_features, geo_features)
        return torch.argmax(probs, dim=1)
    
    def loss_function(self, logits, targets):
        """Calculate cross entropy loss"""
        return F.cross_entropy(logits, targets) 