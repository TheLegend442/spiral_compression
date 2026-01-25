

## =========================================================================
##  GENERAL IMPORTS
## =========================================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

## =========================================================================
##  GENERAL HELPER FUNCTIONS
## =========================================================================

# Saves a dictionary of numpy arrays to a .npz file
def save_npz_dict(save_path: str, data: dict, *, compress: bool = True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # (optional) make sure values are numpy-friendly
    payload = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            payload[k] = v
        else:
            payload[k] = np.asarray(v)
    if compress:
        np.savez_compressed(save_path, **payload)
    else:
        np.savez(save_path, **payload)

# Generates a default spiral dictionary with zeroed severity and type labels
def generate_spiral_dict(total_steps):
    """Return a default spiral dictionary with zeroed severity and type labels."""
    return {
        'theta': None,                          # Angle values
        'r': None,                              # r values
        'flat_onehot': total_steps * [0],       # One-hot labels for flat segments (1 means flat)
        'tight_onehot': total_steps * [0],      # One-hot labels for tight segments (1 means tight)
        'tightness': total_steps * [0],         # Tightness severity values - based on distribution (0 not tight, 1 very tight)
        'normality': total_steps * [0],         # Normality severity values - based on deviaton from normal spiral (0 normal, 1 very abnormal)
        'k': None,                              # Underlying Archimedes spiral growth rate
        'embedding': None,                      # Placeholder for Chronos embedding
        'k_tight': None,                        # Tightness growth rate (if applicable)
        'theta_tight': None,                    # Tightness angle (if applicable)
        'n_tight': None,                        # Tightness n parameter (if applicable)
    }


## =========================================================================
##  PLOTTING
## =========================================================================



## =========================================================================
##  PREPROCESSING
## =========================================================================

from chronos_emb import ChronosEmbedder
from sklearn.linear_model import RANSACRegressor, LinearRegression

# Used to preprocess spiral - robust line fitting
def fit_robust_line(x, y):
    """
    Fit a robust line y = a*x + b using RANSAC.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
    y : array-like, shape (n_samples,)

    Returns
    -------
    a : float
        Slope of the fitted line.
    b : float
        Intercept of the fitted line.
    model : object
        Fitted RANSACRegressor model with a LinearRegression base.
    """
    x = np.asarray(x).reshape(-1, 1)  # sklearn wants 2D for features
    y = np.asarray(y)

    base_model = LinearRegression()
    ransac = RANSACRegressor(
        estimator=base_model,
        min_samples=0.5,      # fraction or absolute number of points
        residual_threshold=None,  # auto threshold
        max_trials=100,
        random_state=42,
    )
    ransac.fit(x, y)

    a = ransac.estimator_.coef_[0]
    b = ransac.estimator_.intercept_
    return a, b, ransac

# Resamples spiral data based on a fitted line for initial segment, uses linear interpolation afterwards
def resample_spiral(theta, r, a, b, angle2cut, pts_per_rotation=100):
    """
        theta : array-like, shape (n_samples,)
        r : array-like, shape (n_samples,)
        a : float
            Slope of the fitted line for initial segment.
        b : float
            Intercept of the fitted line for initial segment. (corrected to fit r at angle2cut)
        angle2cut : float
            Angle at which to switch from line model to interpolation.
        pts_per_rotation : int, optional
            Number of points to sample per full rotation (2*pi). Default is 100.
    """
    
    new_r, new_theta = [], []
    
    current_theta = 0
    while current_theta <= theta[-1]:
        new_theta.append(current_theta)
        
        if current_theta < angle2cut:
            new_r.append(a * current_theta + b)
            current_theta += 2 * np.pi / pts_per_rotation
            continue
        
        up_idx = np.searchsorted(theta, current_theta, side="right")

        # Clamp indices
        if up_idx == 0:
            new_r.append(r[0])
            continue
        if up_idx >= len(theta):
            new_r.append(r[-1])
            continue
        
        lw_idx = up_idx - 1

        t0 = theta[lw_idx]
        t1 = theta[up_idx]
        r0 = r[lw_idx]
        r1 = r[up_idx]
        
        if t1 == t0:
            temp_r = r0
        else:
            w = (current_theta - t0) / (t1 - t0)
            temp_r = (1 - w) * r0 + w * r1
        
        
        new_r.append(temp_r)
        current_theta += 2 * np.pi / pts_per_rotation
        
    return np.array(new_theta), np.array(new_r)

# Prepares input vector for Chronos embedding
def prepare_chronos_input(r, theta):
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    return torch.stack([r, sin_theta, cos_theta], dim=0)

# Main preprocessing function for spiral data - reads CSV, centers, converts to polar, fits line, resamples
def preprocess_spiral(file_path, pts_per_rotation=100, smoothing=False, mirror_over_x=False):
    
    # Read CSV and remove duplicates (due to overlapping points)
    df = pd.read_csv(file_path)
    df.drop_duplicates(subset=['x', 'y'], keep='first', inplace=True)
    
    # Center the points
    df['x'] = df['x'] - df['x'].iloc[0]
    df['y'] = df['y'] - df['y'].iloc[0]

    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    if mirror_over_x: y = -y ## MINUS KER JE V INPUT CSV-JU SLIKA ZRCALJENA ČEZ X-OS (idk why...)
    
    # Convert to polar coordinates
    org_theta = np.arctan2(y, x)
    org_theta = np.unwrap(org_theta)
    org_r = np.sqrt(x**2 + y**2)
    
    theta = org_theta.copy()
    r = org_r.copy()
    
    # Assert counter-clockwise orientation
    INVERSION = False
    theta_sum = np.sum(theta)
    if theta_sum < 0:
        theta = -theta
        INVERSION = True
        
    # Optional smoothing
    if smoothing:
        r = gaussian_filter(r, sigma=5.0, mode='nearest', radius=3)
        
    # Shift to start from minimum theta and re-center
    min_theta_idx = np.argmin(theta)
    theta_shifted = theta[min_theta_idx:].copy()
    SHIFT_ANGLE = theta_shifted[0]
    theta_shifted -= SHIFT_ANGLE
    theta_shifted = np.asarray(theta_shifted)
    
    r_shifted = r[min_theta_idx:].copy()
    r_shifted = np.asarray(r_shifted)
    
    # RANSAC line fitting - fit only to points with theta > angle2cut
    angle2cut = np.pi
    theta_shifted_pi = theta_shifted[theta_shifted > angle2cut]
    r_shifted_pi = r_shifted[theta_shifted > angle2cut]
    _a, _b, _ = fit_robust_line(theta_shifted_pi, r_shifted_pi) # RANSAC
    
    theta_cut_idx = np.where(theta_shifted > angle2cut)[0][0]
    corrected_b = r_shifted[theta_cut_idx] - _a * theta_shifted[theta_cut_idx]

    # Generate new theta and r values, evenly spaced
    new_theta, new_r = resample_spiral(
        theta_shifted,
        r_shifted,
        _a,
        corrected_b,
        angle2cut,
        pts_per_rotation=pts_per_rotation
    )
    
    # Move spiral radius to start at zero
    r_idx = np.argmin(new_r)
    R_SHIFT = new_r[r_idx]
    new_r = new_r - R_SHIFT

    return new_theta, new_r, org_theta, org_r, _a, corrected_b, SHIFT_ANGLE, INVERSION, R_SHIFT

def preprocess_save_spiral(input_csv_path, output_npz_path, pts_per_rotation=100, smoothing=False, chronos_pipeline=None, mirror_over_x=False):
    new_theta, new_r, org_theta, org_r, _, _, shift_angle, inversion, r_shift = preprocess_spiral(
        input_csv_path,
        pts_per_rotation=pts_per_rotation,
        smoothing=smoothing,
        mirror_over_x=mirror_over_x,
    )
    
    spiral_dict = generate_spiral_dict(len(new_theta))
    
    spiral_dict['theta'] = new_theta
    spiral_dict['r'] = new_r
    
    # Store original data and performed transformations
    spiral_dict['org_theta'] = org_theta
    spiral_dict['org_r'] = org_r
    spiral_dict['shift_angle'] = shift_angle
    spiral_dict['inversion'] = inversion
    spiral_dict['r_shift'] = r_shift
    
    if chronos_pipeline is None:
        chronos_pipeline = ChronosEmbedder()
    x = prepare_chronos_input(torch.from_numpy(new_r), torch.from_numpy(new_theta))
    emb = chronos_pipeline.embed_single(x)
    spiral_dict['embedding'] = emb
    
    save_npz_dict(output_npz_path, spiral_dict)