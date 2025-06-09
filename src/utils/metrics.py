import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import List, Dict

def calculate_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        
    Returns:
        Dictionary containing various metrics
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1': f1_score(labels, predictions, average='weighted')
    }
    
    # Calculate ROC-AUC if binary classification
    if len(np.unique(labels)) == 2:
        metrics['roc_auc'] = roc_auc_score(labels, predictions)
    
    return metrics

def calculate_risk_scores(
    predictions: List[int],
    probabilities: List[float],
    confidence_threshold: float = 0.8
) -> Dict[str, float]:
    """
    Calculate risk assessment scores
    
    Args:
        predictions: List of predicted labels
        probabilities: List of prediction probabilities
        confidence_threshold: Threshold for high confidence predictions
        
    Returns:
        Dictionary containing risk assessment metrics
    """
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Calculate high confidence predictions
    high_confidence_mask = probabilities >= confidence_threshold
    high_confidence_predictions = predictions[high_confidence_mask]
    
    # Calculate risk metrics
    risk_metrics = {
        'high_risk_ratio': np.mean(predictions == 1),
        'high_confidence_ratio': np.mean(high_confidence_mask),
        'high_risk_high_confidence_ratio': np.mean(
            (predictions == 1) & high_confidence_mask
        )
    }
    
    return risk_metrics 