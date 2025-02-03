# utils/prediction.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C

def create_features(input_data):
    if isinstance(input_data, dict):
        data = pd.DataFrame([input_data])
    else:
        data = input_data.copy()
    
    X = pd.DataFrame({
        'ServiceQuality': [data['ServiceQuality']],
        'Price': [data['Price']],
        'Innovation': [data['Innovation']]
    })
    
    return X

def get_prediction_model():
    kernel = (C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) +
             RBF(length_scale=1.0) +
             WhiteKernel(noise_level=1.0))

    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.2,
        n_restarts_optimizer=20,
        normalize_y=True,
        random_state=42
    )
