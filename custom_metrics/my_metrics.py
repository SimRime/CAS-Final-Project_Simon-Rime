# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:43:47 2024

@author: srime
"""

# my_metrics.py
import numpy as np
from autogluon.core.metrics import make_scorer

def custom_loss_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Set fixed values for threshold and penalty weight
    threshold = 1.0
    penalty_weight = 1.0
    
    # Calculate mean squared error
    mse = ((y_true - y_pred) ** 2).mean()
    
    # Calculate penalty for underfitting high values
    penalty = ((y_true - y_pred)[y_true > threshold] ** 2).sum()
    
    # Combine mse with the penalty
    return mse + penalty_weight * penalty

# Create the AutoGluon scorer
custom_scorer = make_scorer(name='custom_loss',
                            score_func=custom_loss_func,
                            optimum=0,
                            greater_is_better=False)