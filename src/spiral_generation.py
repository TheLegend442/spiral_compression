
import random
import numpy as np
from chronos_emb import ChronosEmbedder
from scipy.ndimage import gaussian_filter


import torch
import os
import argparse
from pathlib import Path

## ===========================================================
## HELPER FUNCTIONS
## ===========================================================

def prepare_chronos_input(r, theta):
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    return torch.stack([r, sin_theta, cos_theta], dim=0)

def apply_local_shaping(r_data, theta_data, local_minimas_theta, local_minimas_r, 
                        local_maximas_theta, local_maximas_r, tightness_width, flat_precentage):
    """
    Applies local shaping (tightening for minimas, loosening for maximas) to the radial data.

    Args:
        r_data (np.ndarray): The radial data array to be modified.
        theta_data (np.ndarray): The corresponding angular data array.
        local_minimas_theta (np.ndarray): Angles of local minimas.
        local_minimas_r (np.ndarray): Radius of local minimas.
        local_maximas_theta (np.ndarray): Angles of local maximas.
        local_maximas_r (np.ndarray): Radii of local maximas.
        tightness_width (float): The total angular width of the shaping window.
        flat_precentage (float): The percentage of the width that will be flat (0 to 1).

    Returns:
        np.ndarray: The modified radial data array.
    """
    
    # Create a copy of the radial data to modify
    r_shaped = r_data.copy()
    
    HALF_FLAT_WIDTH = (tightness_width * flat_precentage) / 2
    
    # Combine minimas and maximas into a list of tasks
    # Each task is (theta_list, r_list, application_function)
    tasks = [
        (local_minimas_theta, local_minimas_r, np.minimum),  # Minimas: Use np.minimum to pull down
        (local_maximas_theta, local_maximas_r, np.maximum)   # Maximas: Use np.maximum to push up
    ]
    
    # Process both minimas and maximas in a single loop structure
    for lt_thetas, lt_rs, application_func in tasks:
        
        if len(lt_thetas) == 0:
            continue

        for i in range(len(lt_thetas)):
            lt_theta = lt_thetas[i]
            lt_r = lt_rs[i]
            
            # 1. Define the full angular window (W)
            full_start_angle = lt_theta - tightness_width / 2
            full_end_angle = lt_theta + tightness_width / 2
            
            # 2. Define the FLAT region boundaries (F)
            flat_start_angle = lt_theta - HALF_FLAT_WIDTH
            flat_end_angle = lt_theta + HALF_FLAT_WIDTH
            
            # 3. Get the radius at the edges of the full window from the CURRENT r_shaped array
            # These are the points the ramps connect *to*.
            # Use theta_data for the x-axis and r_shaped for the y-axis (the current curve)
            r_at_start = np.interp(full_start_angle, theta_data, r_shaped)
            r_at_end = np.interp(full_end_angle, theta_data, r_shaped)
            
            # 4. Initialize the target radius array for the entire window
            mask_full = (theta_data >= full_start_angle) & (theta_data <= full_end_angle)
            theta_segment = theta_data[mask_full]
            r_target_segment = np.empty_like(theta_segment)
            
            # --- A. Flat Central Region ---
            mask_flat = (theta_segment >= flat_start_angle) & (theta_segment <= flat_end_angle)
            r_target_segment[mask_flat] = lt_r
            
            # --- B. Left Ramp Region ---
            mask_left = theta_segment < flat_start_angle
            theta_left = theta_segment[mask_left]
            
            # Ramp connects r_at_start (at full_start_angle) to lt_r (at flat_start_angle)
            denominator_left = flat_start_angle - full_start_angle
            
            if denominator_left > 1e-9: 
                # Formula: r = r_start + (r_end - r_start) * ( (theta - theta_start) / (theta_end - theta_start) )
                r_target_left = r_at_start + (lt_r - r_at_start) * (
                    (theta_left - full_start_angle) / denominator_left
                )
            else: 
                r_target_left = np.full_like(theta_left, lt_r)

            r_target_segment[mask_left] = r_target_left
            
            # --- C. Right Ramp Region ---
            mask_right = theta_segment > flat_end_angle
            theta_right = theta_segment[mask_right]

            # Ramp connects lt_r (at flat_end_angle) to r_at_end (at full_end_angle)
            denominator_right = full_end_angle - flat_end_angle
            
            if denominator_right > 1e-9: 
                # Formula: r = r_start + (r_end - r_start) * ( (theta - theta_start) / (theta_end - theta_start) )
                r_target_right = lt_r + (r_at_end - lt_r) * (
                    (theta_right - flat_end_angle) / denominator_right
                )
            else: 
                r_target_right = np.full_like(theta_right, lt_r)
                
            r_target_segment[mask_right] = r_target_right
            
            # 5. Apply the shaping: use the stored application_func (np.minimum for minimas, np.maximum for maximas)
            r_shaped[mask_full] = application_func(r_shaped[mask_full], r_target_segment)
            
    return r_shaped

def check_for_intersections(theta, r):
    """
    Checks for self-intersection in a polar curve (r, theta) over a 2*pi rotation.
    Self-intersection occurs if r(theta + 2*pi) < r(theta).
    
    Args:
        theta (np.ndarray): Array of angles (must span at least 2*pi).
        r (np.ndarray): Array of radii.

    Returns:
        bool: True if self-intersection is detected, False otherwise.
    """
    
    # 1. Ensure theta is sorted (required for np.interp)
    # This is generally true for spiral generation, but good practice.
    sort_indices = np.argsort(theta)
    theta_sorted = theta[sort_indices]
    r_sorted = r[sort_indices]

    # 2. Iterate only over the angles where theta + 2*pi is still within the domain
    # This optimization avoids unnecessary checks and handles bounds better.
    theta_limit = theta_sorted[-1] - 2 * np.pi
    
    # Use a mask to get the angles in the current domain (theta) that are also in the 
    # range [theta_start, theta_end - 2*pi]
    mask = theta_sorted <= theta_limit
    theta_current = theta_sorted[mask]
    r_current = r_sorted[mask]

    # 3. Use NumPy's highly efficient np.interp to find r(theta + 2*pi)
    # The 'theta_current + 2 * np.pi' array serves as the x-coordinates to interpolate.
    theta_plus_2pi = theta_current + 2 * np.pi
    
    # r_plus_2pi is the interpolated radius at the next rotation
    r_plus_2pi = np.interp(theta_plus_2pi, theta_sorted, r_sorted)

    # 4. Check the condition for self-intersection across all relevant points
    # Intersection occurs if the radius is decreasing too fast: r(theta + 2pi) < r(theta)
    # if np.any(r_plus_2pi < r_current):
    #     return True
    
    # return False 
    
    return np.sum(r_plus_2pi < r_current)

def generete_flat_params(spiral_params, flat_params_input):
    
    flat_params = flat_params_input.copy()
    flat_params['insertions'] = []
    flat_params['extensions'] = []
    
    max_ins_per_rotation = flat_params['max_ins_per_rotation']
    max_ext_per_rotation = flat_params['max_ext_per_rotation']
    ins_len_range = flat_params['ins_len_range']
    start_offset_range = flat_params['start_offset_range']
    end_offset_range = flat_params['end_offset_range']
    disp_fac_r_range = flat_params['disp_fac_r_range']
    disp_fac_len_range = flat_params['disp_fac_len_range']
    
    num_rotations = (spiral_params['theta_end'] - spiral_params['theta_start']) / (2 * np.pi)
    num_insertions = np.random.randint(1, max(2, int(max_ins_per_rotation * num_rotations)))
    num_extensions = np.random.randint(0, max(1, int(max_ext_per_rotation * num_rotations)))
    
    def sample_with_min_distance(a, b, n, r, pts=None, max_att = 10000):
        """Sample n points in [a, b] such that any two points are at least r apart."""
        
        if pts is not None:
            points = pts.copy()
        points = []
        attempts = 0
        max_attempts = max_att

        while len(points) < n and attempts < max_attempts:
            x = random.uniform(a, b)
            if all(abs(x - p) >= r for p in points):
                points.append(x)
            attempts += 1

        if len(points) < n:
            raise RuntimeError("Could not place all points; reduce n or r.")
        return sorted(points)
    
    insertion_angles = sample_with_min_distance(spiral_params['theta_start'] + 2*np.pi, spiral_params['theta_end'], num_insertions, 0.5*np.pi)
    extension_angles = sample_with_min_distance(spiral_params['theta_start'] + 2*np.pi, spiral_params['theta_end'], num_extensions, 0.5*np.pi, pts=insertion_angles)
    
    for ang in insertion_angles:
        insertion = {
            "ins_angle": ang,
            "ins_len": np.random.uniform(ins_len_range[0], ins_len_range[1])
        }
        flat_params['insertions'].append(insertion)
        
    for ang in extension_angles:
        extension = {
            "mid_angle": ang,
            "start_offset": np.random.uniform(start_offset_range[0], start_offset_range[1]),
            "end_offset": np.random.uniform(end_offset_range[0], end_offset_range[1]),
            "displacement_fac_r": np.random.uniform(disp_fac_r_range[0], disp_fac_r_range[1]),
            "displacement_fac_len": np.random.uniform(disp_fac_len_range[0], disp_fac_len_range[1])
        }
        flat_params['extensions'].append(extension)
    
    return flat_params

def hellinger_gaussians(mu1, sigma1, mu2, sigma2, eps=1e-12):
    s1 = max(float(sigma1), eps)
    s2 = max(float(sigma2), eps)

    denom = s1*s1 + s2*s2
    coeff = np.sqrt((2.0*s1*s2) / denom)
    expo = np.exp(-((mu1 - mu2)**2) / (4.0*denom))

    h2 = 1.0 - coeff * expo
    # numerical safety
    h2 = float(np.clip(h2, 0.0, 1.0))
    return np.sqrt(h2)  # in [0,1]

def radius_distribution(theta_values, r_values, view_width):
    
    def get_divergence(_theta, _r, view_angle, view_width=view_width):
        r_tight = []
        r_normal = []
        
        tight_slots = []
        ang = view_angle
        while ang < _theta[-1] + view_width / 2:
            tight_slots.append((ang - view_width/2, ang + view_width/2))
            ang += 2*np.pi
            
        normal_slots = []
        ang = view_angle + np.pi
        while ang < _theta[-1] + view_width / 2:
            normal_slots.append((ang - view_width/2, ang + view_width/2))
            ang += 2*np.pi
            
        for ts in tight_slots:
            idxs = np.where((_theta >= ts[0]) & (_theta <= ts[1]))[0]
            r_tight.extend(_r[idxs])
            
        for ns in normal_slots:
            idxs = np.where((_theta >= ns[0]) & (_theta <= ns[1]))[0]
            r_normal.extend(_r[idxs])
            
        return np.array(r_tight), np.array(r_normal)
    
    divgs = []
    idx = 0
    while theta_values[idx] - theta_values[0] < np.pi:
        r_tight, r_normal = get_divergence(theta_values, r_values, theta_values[idx])
        
        # Calculate Hellinger Gaussian distance
        mu_tight = np.mean(r_tight)
        sigma_tight = np.std(r_tight)
        mu_normal = np.mean(r_normal)
        sigma_normal = np.std(r_normal)
        
        hell = hellinger_gaussians(mu_tight, sigma_tight, mu_normal, sigma_normal)
        divgs.append(hell)
        idx += 1
        
    target_len = len(r_values)
    repeats = int(np.ceil(target_len / len(divgs)))
    divgs = (divgs * repeats)[:target_len]
    divgs = np.array(divgs)
    
    return divgs

def normality_calculation(r_original, r_modified, eps=1e-6, tau=0.15):
    r_diff = np.abs(r_original - r_modified)
    denom = np.maximum(r_original, eps)
    rel_dev = r_diff / denom
    
    return 1.0 - np.exp(-rel_dev/tau)

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

## ===========================================================
## SPIRAL GENERATION FUNCTIONS
## ===========================================================

def generate_archimedes(spiral_params):
    
    theta_start = spiral_params['theta_start']
    theta_end = spiral_params['theta_end']
    k = spiral_params['k']
    steps_per_rotation = spiral_params['steps_per_rotation']
    
    total_steps = int((theta_end - theta_start) / (2 * np.pi) * steps_per_rotation)
    spiral_dict = generate_spiral_dict(total_steps)
    
    theta = np.linspace(theta_start, theta_end, total_steps)
    r = k * theta
    
    spiral_dict['theta'] = theta
    spiral_dict['r'] = r
    spiral_dict['k'] = k
    spiral_dict['k_tight'] = k
    spiral_dict['n_tight'] = 0.0
    
    spiral_dict['tightness'] = radius_distribution(theta, r, view_width=5*np.pi/50)
    
    return spiral_dict

def generate_spiky(spiky_params):
    
    theta_start = spiky_params['theta_start']
    theta_end = spiky_params['theta_end']
    k = spiky_params['k']
    steps_per_rotation = spiky_params['steps_per_rotation']
    
    total_steps = int((theta_end - theta_start) / (2 * np.pi) * steps_per_rotation)
    spiral_dict = generate_spiral_dict(total_steps)
    
    theta = np.linspace(theta_start, theta_end, total_steps)
    r = k * theta
    
    # Spiky parameters
    curved_base_probability = spiky_params['curved_base_probability']
    period = spiky_params['period']
    curved_spiky_sigma_range = spiky_params['curved_spiky_sigma_range']
    curved_smoothing_sigma = spiky_params['curved_smoothing_sigma']
    curved_smoothing_radius = spiky_params['curved_smoothing_radius']
    sharp_spiky_sigma_range = spiky_params['sharp_spiky_sigma_range']
    spiky_smoothing_sigma = spiky_params['spiky_smoothing_sigma']
    spiky_smoothing_radius = spiky_params['spiky_smoothing_radius']
    
    r_spiky = r.copy()
    if np.random.rand() > curved_base_probability:
        for idx in range(len(r_spiky)):
            scale = np.random.uniform(curved_spiky_sigma_range[0], curved_spiky_sigma_range[1])
            r_spiky[idx] *= np.random.normal(loc=1.0, scale=scale) * np.abs(np.sin(period*theta[idx]))
        r_spiky = gaussian_filter(r_spiky, sigma=curved_smoothing_sigma, mode='nearest', radius=curved_smoothing_radius)

        if np.random.rand() < 0.5:
            for idx in range(len(r_spiky)):
                scale = np.random.uniform(sharp_spiky_sigma_range[0], sharp_spiky_sigma_range[1])
                r_spiky[idx] += np.random.normal(loc=0.0, scale=scale) * r_spiky[idx]
            r_spiky = gaussian_filter(r_spiky, sigma=spiky_smoothing_sigma, mode='nearest', radius=spiky_smoothing_radius)
    else:
        for idx in range(len(r_spiky)):
            scale = np.random.uniform(sharp_spiky_sigma_range[0], sharp_spiky_sigma_range[1])
            r_spiky[idx] += np.random.normal(loc=0.0, scale=scale) * r_spiky[idx]
        r_spiky = gaussian_filter(r_spiky, sigma=spiky_smoothing_sigma, mode='nearest', radius=spiky_smoothing_radius)
    
    
    spiral_dict['theta'] = theta
    spiral_dict['r'] = r_spiky
    spiral_dict['k'] = k
    spiral_dict['k_tight'] = k
    spiral_dict['n_tight'] = 0.0
    
    spiral_dict['tightness'] = radius_distribution(theta, r, view_width=5*np.pi/50)
    
    return spiral_dict

def generate_spiky_tight(spiral_params, tight_params, spiky_params, sigma=10.0, radius=6):
    
    # Archimedes spiral base
    theta_start = spiral_params['theta_start']
    theta_end = spiral_params['theta_end']
    k = spiral_params['k']
    steps_per_rotation = spiral_params['steps_per_rotation']
    
    total_steps = int((theta_end - theta_start) / (2 * np.pi) * steps_per_rotation)
    spiral_dict = generate_spiral_dict(total_steps)
    
    theta = np.linspace(theta_start, theta_end, total_steps)
    r = k * theta
    
    spiral_dict['theta'] = theta
    spiral_dict['k'] = k

    max_r = np.max(r)

    # Tightening parameters
    tightness_angle = tight_params['tight_angle']
    n_ratio = tight_params['n_ratio']
    max_k2k_ratio = tight_params['max_k2k_ratio']
    tightness_k = np.min([tight_params['tight_k'], max_k2k_ratio * k])
    tightness_width = tight_params['tight_width']
    
    def get_local_minimas(tightness_angle, tightness_k, n_ratio, max_r, theta_end):
        """Function that computes  the local minimas positions (theta, r) for tightening."""
        local_minimas_theta = []
        local_minimas_r = []
        minima_theta = tightness_angle
        while minima_theta < theta_end + 2*np.pi:
            local_minimas_theta.append(minima_theta)
            local_minimas_r.append(tightness_k * minima_theta + n_ratio * max_r)
            minima_theta += 2*np.pi
        local_minimas_theta = np.array(local_minimas_theta)
        local_minimas_r = np.array(local_minimas_r)
        
        return local_minimas_theta, local_minimas_r
    
    minimas_theta, minimas_r = get_local_minimas(tightness_angle, tightness_k, n_ratio, max_r, theta_end) 

    r_tightened = r.copy()
    r_tightened = apply_local_shaping(r_tightened, theta, minimas_theta, minimas_r, [], [], tightness_width, tight_params['flat_precentage'])
    r_tightened = gaussian_filter(r_tightened, sigma=sigma, mode='nearest', radius=radius)
    
    # Spiky parameters
    curved_base_probability = spiky_params['curved_base_probability']
    period = spiky_params['period']
    curved_spiky_sigma_range = spiky_params['curved_spiky_sigma_range']
    curved_smoothing_sigma = spiky_params['curved_smoothing_sigma']
    curved_smoothing_radius = spiky_params['curved_smoothing_radius']
    sharp_spiky_sigma_range = spiky_params['sharp_spiky_sigma_range']
    spiky_smoothing_sigma = spiky_params['spiky_smoothing_sigma']
    spiky_smoothing_radius = spiky_params['spiky_smoothing_radius']
    
    r_spiky = r_tightened.copy()
    if np.random.rand() > curved_base_probability:
        for idx in range(len(r_spiky)):
            scale = np.random.uniform(curved_spiky_sigma_range[0], curved_spiky_sigma_range[1])
            r_spiky[idx] *= np.random.normal(loc=1.0, scale=scale) * np.abs(np.sin(period*theta[idx]))
        r_spiky = gaussian_filter(r_spiky, sigma=curved_smoothing_sigma, mode='nearest', radius=curved_smoothing_radius)

        if np.random.rand() < 0.5:
            for idx in range(len(r_spiky)):
                scale = np.random.uniform(sharp_spiky_sigma_range[0], sharp_spiky_sigma_range[1])
                r_spiky[idx] += np.random.normal(loc=0.0, scale=scale) * r_spiky[idx]
            r_spiky = gaussian_filter(r_spiky, sigma=spiky_smoothing_sigma, mode='nearest', radius=spiky_smoothing_radius)
    else:
        for idx in range(len(r_spiky)):
            scale = np.random.uniform(sharp_spiky_sigma_range[0], sharp_spiky_sigma_range[1])
            r_spiky[idx] += np.random.normal(loc=0.0, scale=scale) * r_spiky[idx]
        r_spiky = gaussian_filter(r_spiky, sigma=spiky_smoothing_sigma, mode='nearest', radius=spiky_smoothing_radius)
    
    
    spiral_dict['r'] = r_spiky
    spiral_dict['k'] = k
    spiral_dict['k_tight'] = tightness_k
    spiral_dict['n_tight'] = 0.0
    
    spiral_dict['tightness'] = radius_distribution(theta, r, view_width=5*np.pi/50)
    
    return spiral_dict

def generate_tight(spiral_params, tight_params, sigma=10.0, radius=6):
    
    # Archimedes spiral base
    theta_start = spiral_params['theta_start']
    theta_end = spiral_params['theta_end']
    k = spiral_params['k']
    steps_per_rotation = spiral_params['steps_per_rotation']
    
    total_steps = int((theta_end - theta_start) / (2 * np.pi) * steps_per_rotation)
    spiral_dict = generate_spiral_dict(total_steps)
    
    theta = np.linspace(theta_start, theta_end, total_steps)
    r = k * theta
    
    spiral_dict['theta'] = theta
    spiral_dict['k'] = k

    max_r = np.max(r)

    # Tightening parameters
    tightness_angle = tight_params['tight_angle']
    n_ratio = tight_params['n_ratio']
    max_k2k_ratio = tight_params['max_k2k_ratio']
    tightness_k = np.min([tight_params['tight_k'], max_k2k_ratio * k])
    tightness_width = tight_params['tight_width']
    
    def get_local_minimas(tightness_angle, tightness_k, n_ratio, max_r, theta_end):
        """Function that computes  the local minimas positions (theta, r) for tightening."""
        local_minimas_theta = []
        local_minimas_r = []
        minima_theta = tightness_angle
        while minima_theta < theta_end + 2*np.pi:
            local_minimas_theta.append(minima_theta)
            local_minimas_r.append(tightness_k * minima_theta + n_ratio * max_r)
            minima_theta += 2*np.pi
        local_minimas_theta = np.array(local_minimas_theta)
        local_minimas_r = np.array(local_minimas_r)
        
        return local_minimas_theta, local_minimas_r
    
    minimas_theta, minimas_r = get_local_minimas(tightness_angle, tightness_k, n_ratio, max_r, theta_end) 

    r_tightened = r.copy()
    r_tightened = apply_local_shaping(r_tightened, theta, minimas_theta, minimas_r, [], [], tightness_width, tight_params['flat_precentage'])
    r_tightened = gaussian_filter(r_tightened, sigma=sigma, mode='nearest', radius=radius)
    
    spiral_dict['r'] = r_tightened
    
    r_diff = list(np.abs(r - r_tightened))
    spiral_dict['tight_onehot'] = [1 if rd > r * 0.08 else 0 for r, rd in zip(r, r_diff)]
    spiral_dict['tightness'] = radius_distribution(theta, r_tightened, view_width=5*np.pi/50)
    spiral_dict['normality'] = normality_calculation(r, r_tightened)
    
    spiral_dict['k_tight'] = tightness_k
    spiral_dict['theta_tight'] = tightness_angle
    spiral_dict['n_tight'] = n_ratio * max_r
    
    return spiral_dict

def generate_flat(spiral_params, flat_params, type="insert_line", sigma_ins=2.0, radius_ins=2, sigma_ext=20.0, radius_ext=1):
    
    # Archimedes spiral base
    theta_start = spiral_params['theta_start']
    theta_end = spiral_params['theta_end']
    k = spiral_params['k']
    steps_per_rotation = spiral_params['steps_per_rotation']
    
    total_steps = int((theta_end - theta_start) / (2 * np.pi) * steps_per_rotation)
    spiral_dict = generate_spiral_dict(total_steps)
    
    theta = np.linspace(theta_start, theta_end, total_steps)
    r = k * theta
    
    spiral_dict['theta'] = theta
    spiral_dict['k'] = k
    
    def insert_flat2radius(_theta, _r, start_idx, end_idx, start_point, end_point):
        """Calculate new radius values for points between start_idx and end_idx that form a straight line connecting start and end point"""
        modified_r = _r.copy()
        
        for ang_idx in range(start_idx, end_idx):
            ang = _theta[ang_idx]
            t = (np.tan(ang) * start_point[0] - start_point[1]) / (end_point[1] - start_point[1] - np.tan(ang) * (end_point[0] - start_point[0]))
            x = start_point[0] + t * (end_point[0] - start_point[0])
            y = start_point[1] + t * (end_point[1] - start_point[1])
            
            r_temp = np.sqrt(x**2 + y**2)
            modified_r[ang_idx] = r_temp
            
        return modified_r
    
    # OPTION 1 - Insert line segment
    def insert_line_segment(_theta, _r, _start_ang, _flat_len_rad):
        start_angle, end_angle = _start_ang, _start_ang + _flat_len_rad # Define flat segment angles
        start_idx = np.argmin(abs(theta - start_angle))
        end_idx = np.argmin(abs(theta - end_angle))
        
        start_ang, end_ang = _theta[start_idx], _theta[end_idx] # Exact angles that exist in _theta
        start_r, end_r = _r[start_idx], _r[start_idx]           # Corresponding radius values
        
        start_point = start_r * np.cos(start_ang), start_r * np.sin(start_ang)  # Cartesian coords
        end_point = end_r * np.cos(end_ang), end_r * np.sin(end_ang)            # Cartesian coords
        
        modified_r = insert_flat2radius(_theta, _r, start_idx + 1, end_idx, start_point, end_point)
        
        return modified_r, start_idx, end_idx
    
    # OPTION 2 - Extended flat regions
    def extend_radius_region(_theta, _r, mid_angle, start_offset, end_offset, displacement_fac_r, displacement_fac_len):
        mid_idx = np.argmin(abs(_theta - mid_angle))
        start_idx = np.argmin(abs(_theta - (mid_angle - start_offset)))
        mid_ang = _theta[mid_idx]
        start_ang = _theta[start_idx]
        mid_r = _r[mid_idx] * (1 + displacement_fac_r) # Displace radius at mid point
        start_r = _r[start_idx]
        
        mid_point = mid_r * np.cos(mid_ang), mid_r * np.sin(mid_ang)            # Cartesian coords
        start_point = start_r * np.cos(start_ang), start_r * np.sin(start_ang)  # Cartesian coords

        def unfold_angle(angle, _mid_ang):
            if angle < 0: angle += 2 * np.pi
            while angle < _mid_ang:
                angle += 2 * np.pi
            return angle

        # Determine the end point of the extended line segment
        end_point_x = mid_point[0] + (mid_point[0] - start_point[0]) * displacement_fac_len # Extend line segment
        end_point_y = mid_point[1] + (mid_point[1] - start_point[1]) * displacement_fac_len # Extend line segment
        end_point = (end_point_x, end_point_y)
        
        end_ang = np.arctan2(end_point[1], end_point[0])
        end_ang = unfold_angle(end_ang, mid_ang)
        end_idx = np.argmin(abs(_theta - end_ang))
        
        # Determine the fourth point (where segments meets the spiral again)
        fourth_angle = end_ang + end_offset
        fourth_idx = np.argmin(abs(_theta - fourth_angle))
        fourth_ang = _theta[fourth_idx]
        fourth_r = _r[fourth_idx]
        fourth_point = fourth_r * np.cos(fourth_ang), fourth_r * np.sin(fourth_ang)

        # Insert the line segment
        modified_r = insert_flat2radius(_theta, _r, start_idx + 1, end_idx, start_point, end_point)
        modified_r = insert_flat2radius(_theta, modified_r, end_idx, fourth_idx, end_point, fourth_point)
        
        return modified_r, start_idx, fourth_idx
        
    # Process flat segments
    r_flat = r.copy()
    insertions = flat_params['insertions']
    for ins in insertions:
        r_flat, start, end = insert_line_segment(theta, r_flat, ins['ins_angle'], ins['ins_len'])
        spiral_dict['flat_onehot'][start:end] = [1]*(end-start) # Mark as flat segment
    r_flat = gaussian_filter(r_flat, sigma=sigma_ins, mode='nearest', radius=radius_ins)
    
    extensions = flat_params['extensions']
    for ext in extensions:
        r_flat, start, end = extend_radius_region(
            theta, r_flat,
            ext['mid_angle'],
            ext['start_offset'],
            ext['end_offset'],
            ext['displacement_fac_r'],
            ext['displacement_fac_len']
        )
        spiral_dict['flat_onehot'][start:end] = [1]*(end-start) # Mark as flat segment
    r_flat = gaussian_filter(r_flat, sigma=sigma_ext, mode='nearest', radius=radius_ext)
    
    spiral_dict['r'] = r_flat
    
    # Add normality
    r_diff = list(np.abs(r - r_flat))
    spiral_dict['normality'] = normality_calculation(r, r_flat)
    spiral_dict['tightness'] = radius_distribution(theta, r_flat, view_width=5*np.pi/50)
    
    spiral_dict['k_tight'] = k
    spiral_dict['n_tight'] = 0.0
    
    return spiral_dict

# ===========================================================
## MAIN FUNCTION
## ===========================================================

def main():
    ## General parameters
    STEPS_PER_ROTATION = 100                    # Number of steps per full rotation (2*pi) 
    MIN_THETA_RANGE = (0, np.pi)                # Start theta angle range for spiral generation
    MAX_THETA_RANGE = (6*np.pi, 12*np.pi)       # End theta angle range for spiral generation
    K_RANGE = (0.5, 2.0)                        # Range for spiral growth rate

    ## Tightness parameters
    N_RATIO_RANGE = (0.0, 0.02)                 # Ratio of n parameter for linear equation in terms of max radius
    TIGHT_K_RANGE = (0.4, 1.3)                  # Range for tightness_k parameter for spiral growth rate
    TIGHT_WIDTH_RANGE = (4.5, 7.5)              # Width of tightness
    MAX_K2K_RATIO = 0.38                        # Max ratio of tightness_k to original k
    FLAT_PERCENTAGE = 0.38                      # Percentage of flat region in tightness window

    ## Spikiness parameters
    curved_base_probability = 0.5
    sharp_spiky_sigma_range = (0.03, 0.12)
    curved_spiky_sigma_range = (0.1, 0.17)
    curved_spiky_period_range = (8, 16)

    sharp_spiky_smoothing_sigma_range = (0.12, 0.3)
    sharp_spiky_smoothing_raduis_range = (1, 2)
    curved_spiky_smoothing_sigma_range = (12, 20)
    curved_spiky_smoothing_raduis_range = (5, 6)

    ## Flatness parameters
    MAX_INS_PER_ROTATION = 1                    # Max number of insertions per full rotation
    MAX_EXT_PER_ROTATION = 0.8                  # Max number of extensions per full rotation
    INS_LEN_RANGE = (0.2*np.pi, 0.55*np.pi)     # Range for insertion length in radians
    START_OFFSET_RANGE = (0.1*np.pi, 0.5*np.pi) # Range for extension start offset in radians
    END_OFFSET_RANGE = (0.1*np.pi, 0.5*np.pi)   # Range for extension end offset in radians
    DISP_FAC_R_RANGE = (0.003, 0.02)            # Range for displacement factor for radius
    DISP_FAC_LEN_RANGE = (0.2, 1.05)            # Range for displacement factor for length
    
    flat_params_deafult = {
        "insertions": [],
        "exstisnejtensions": [],
        "max_ins_per_rotation": MAX_INS_PER_ROTATION,
        "max_ext_per_rotation": MAX_EXT_PER_ROTATION,
        "ins_len_range": INS_LEN_RANGE,
        "start_offset_range": START_OFFSET_RANGE,
        "end_offset_range": END_OFFSET_RANGE,
        "disp_fac_r_range": DISP_FAC_R_RANGE,
        "disp_fac_len_range": DISP_FAC_LEN_RANGE
    }
    
    ## Generation parameters
    BATCH_SIZE = 256
    
    parser = argparse.ArgumentParser(
        description="Synthetic Spiral Generation with Local Tightening and Flattening",
    )
    
    parser.add_argument(
        "--spiral_type", type=str,
        help="Input spiral type to generate.",
        default="normal",
        choices=["normal", "spiky_tight", "tight", "spiky", "flat"]
    )
    
    parser.add_argument(
        "--output_dir", type=str,
        help="Output directory to save the generated spiral CSV.",
        default="../data/"
    )
    
    parser.add_argument(
        "--num_spirals", type=int,
        help="Number of spirals to generate.",
        default=1
    )
    
    args = parser.parse_args()
    spiral_dicts = []
    pipeline = ChronosEmbedder()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    match args.spiral_type:
        case "normal":
            spiral_idx = 0
            spiral_counter = 0
            while spiral_counter < args.num_spirals:
                batch_dicts = []
                batch_series = []
                batch_counter = 0
                while batch_counter < BATCH_SIZE and spiral_counter < args.num_spirals:
                    spiral_params = {
                        'steps_per_rotation': STEPS_PER_ROTATION,
                        'theta_start': np.random.uniform(MIN_THETA_RANGE[0], MIN_THETA_RANGE[1]),
                        'theta_end': np.random.uniform(MAX_THETA_RANGE[0], MAX_THETA_RANGE[1]),
                        'k': np.random.uniform(K_RANGE[0], K_RANGE[1])
                    }

                    spiral_dict = generate_archimedes(spiral_params)
                    x = prepare_chronos_input(torch.from_numpy(spiral_dict['r'].astype(np.float32)), torch.from_numpy(spiral_dict['theta'].astype(np.float32)))
                    batch_dicts.append(spiral_dict)
                    batch_series.append(x)
                    batch_counter += 1
                    spiral_counter += 1
                
                # Embed the batch using Chronos
                embeddings = pipeline.embed_batch(batch_series)
                
                for i, emb in enumerate(embeddings):
                    batch_dicts[i]['embedding'] = emb
                    # spiral_dicts.append(batch_dicts[i])
                
                # Save generated spirals
                for spiral_dict in batch_dicts:
                    idx = spiral_idx
                    spiral_idx += 1
                    save_path = f"{output_dir}/{args.spiral_type}_{idx}.npz"
                    save_npz_dict(save_path, spiral_dict)
    
        case "tight":
            spiral_idx = 0
            spiral_counter = 0
            while spiral_counter < args.num_spirals:
                batch_dicts = []
                batch_series = []
                batch_counter = 0
                while batch_counter < BATCH_SIZE and spiral_counter < args.num_spirals:
                    spiral_params = {
                        'steps_per_rotation': STEPS_PER_ROTATION,
                        'theta_start': np.random.uniform(MIN_THETA_RANGE[0], MIN_THETA_RANGE[1]),
                        'theta_end': np.random.uniform(MAX_THETA_RANGE[0], MAX_THETA_RANGE[1]),
                        'k': np.random.uniform(K_RANGE[0], K_RANGE[1])
                    }
                    
                    tight_params = {
                        "tight_angle": np.random.uniform(0, 2*np.pi),
                        "n_ratio" :  np.random.uniform(N_RATIO_RANGE[0], N_RATIO_RANGE[1]),
                        "tight_k":  np.random.uniform(TIGHT_K_RANGE[0], TIGHT_K_RANGE[1]),
                        "tight_width": np.random.uniform(TIGHT_WIDTH_RANGE[0], TIGHT_WIDTH_RANGE[1]),
                        "max_k2k_ratio": MAX_K2K_RATIO,
                        "flat_precentage": FLAT_PERCENTAGE
                    }
                    
                    spiral_dict = generate_tight(spiral_params, tight_params)
                    x = prepare_chronos_input(torch.from_numpy(spiral_dict['r'].astype(np.float32)), torch.from_numpy(spiral_dict['theta'].astype(np.float32)))
                    batch_dicts.append(spiral_dict)
                    batch_series.append(x)
                    batch_counter += 1
                    spiral_counter += 1
                
                # Embed the batch using Chronos
                embeddings = pipeline.embed_batch(batch_series)
                
                for i, emb in enumerate(embeddings):
                    batch_dicts[i]['embedding'] = emb
                    #spiral_dicts.append(batch_dicts[i])
                    
                # Save generated spirals
                for spiral_dict in batch_dicts:
                    idx = spiral_idx
                    spiral_idx += 1
                    save_path = f"{output_dir}/{args.spiral_type}_{idx}.npz"
                    save_npz_dict(save_path, spiral_dict)
        
        case "spiky":
            spiral_idx = 0
            spiral_counter = 0
            while spiral_counter < args.num_spirals:
                batch_dicts = []
                batch_series = []
                batch_counter = 0
                while batch_counter < BATCH_SIZE and spiral_counter < args.num_spirals:
                    spiky_params = {
                        'steps_per_rotation': STEPS_PER_ROTATION,
                        'theta_start': np.random.uniform(MIN_THETA_RANGE[0], MIN_THETA_RANGE[1]),
                        'theta_end': np.random.uniform(MAX_THETA_RANGE[0], MAX_THETA_RANGE[1]),
                        'k': np.random.uniform(K_RANGE[0], K_RANGE[1]),
                        'curved_base_probability': curved_base_probability,
                        'period': np.random.uniform(curved_spiky_period_range[0], curved_spiky_period_range[1]+1),
                        'curved_spiky_sigma_range': curved_spiky_sigma_range,
                        'curved_smoothing_sigma': np.random.uniform(curved_spiky_smoothing_sigma_range[0], curved_spiky_smoothing_sigma_range[1]),
                        'curved_smoothing_radius': np.random.randint(curved_spiky_smoothing_raduis_range[0], curved_spiky_smoothing_raduis_range[1]+1),
                        'sharp_spiky_sigma_range': sharp_spiky_sigma_range,
                        'spiky_smoothing_sigma': np.random.uniform(sharp_spiky_smoothing_sigma_range[0], sharp_spiky_smoothing_sigma_range[1]),
                        'spiky_smoothing_radius': np.random.randint(sharp_spiky_smoothing_raduis_range[0], sharp_spiky_smoothing_raduis_range[1]+1),
                    }
                
                    spiral_dict = generate_spiky(spiky_params)
                    x = prepare_chronos_input(torch.from_numpy(spiral_dict['r'].astype(np.float32)), torch.from_numpy(spiral_dict['theta'].astype(np.float32)))
                    batch_dicts.append(spiral_dict)
                    batch_series.append(x)
                    batch_counter += 1
                    spiral_counter += 1
                
                # Embed the batch using Chronos
                embeddings = pipeline.embed_batch(batch_series)
                
                for i, emb in enumerate(embeddings):
                    batch_dicts[i]['embedding'] = emb
                    # spiral_dicts.append(batch_dicts[i])
                
                # Save generated spirals
                for spiral_dict in batch_dicts:
                    idx = spiral_idx
                    spiral_idx += 1
                    save_path = f"{output_dir}/{args.spiral_type}_{idx}.npz"
                    save_npz_dict(save_path, spiral_dict)

        case "spiky_tight":
            spiral_idx = 0
            spiral_counter = 0
            while spiral_counter < args.num_spirals:
                batch_dicts = []
                batch_series = []
                batch_counter = 0
                while batch_counter < BATCH_SIZE and spiral_counter < args.num_spirals:
                    spiral_params = {
                        'steps_per_rotation': STEPS_PER_ROTATION,
                        'theta_start': np.random.uniform(MIN_THETA_RANGE[0], MIN_THETA_RANGE[1]),
                        'theta_end': np.random.uniform(MAX_THETA_RANGE[0], MAX_THETA_RANGE[1]),
                        'k': np.random.uniform(K_RANGE[0], K_RANGE[1])
                    }
                    
                    tight_params = {
                        "tight_angle": np.random.uniform(0, 2*np.pi),
                        "n_ratio" :  np.random.uniform(N_RATIO_RANGE[0], N_RATIO_RANGE[1]),
                        "tight_k":  np.random.uniform(TIGHT_K_RANGE[0], TIGHT_K_RANGE[1]),
                        "tight_width": np.random.uniform(TIGHT_WIDTH_RANGE[0], TIGHT_WIDTH_RANGE[1]),
                        "max_k2k_ratio": MAX_K2K_RATIO,
                        "flat_precentage": FLAT_PERCENTAGE
                    }
                    
                    spiky_params = {
                        'steps_per_rotation': STEPS_PER_ROTATION,
                        'theta_start': np.random.uniform(MIN_THETA_RANGE[0], MIN_THETA_RANGE[1]),
                        'theta_end': np.random.uniform(MAX_THETA_RANGE[0], MAX_THETA_RANGE[1]),
                        'k': np.random.uniform(K_RANGE[0], K_RANGE[1]),
                        'curved_base_probability': curved_base_probability,
                        'period': np.random.uniform(curved_spiky_period_range[0], curved_spiky_period_range[1]+1),
                        'curved_spiky_sigma_range': curved_spiky_sigma_range,
                        'curved_smoothing_sigma': np.random.uniform(curved_spiky_smoothing_sigma_range[0], curved_spiky_smoothing_sigma_range[1]),
                        'curved_smoothing_radius': np.random.randint(curved_spiky_smoothing_raduis_range[0], curved_spiky_smoothing_raduis_range[1]+1),
                        'sharp_spiky_sigma_range': sharp_spiky_sigma_range,
                        'spiky_smoothing_sigma': np.random.uniform(sharp_spiky_smoothing_sigma_range[0], sharp_spiky_smoothing_sigma_range[1]),
                        'spiky_smoothing_radius': np.random.randint(sharp_spiky_smoothing_raduis_range[0], sharp_spiky_smoothing_raduis_range[1]+1),
                    }
                    
                    spiral_dict = generate_spiky_tight(spiral_params, tight_params, spiky_params)
                    x = prepare_chronos_input(torch.from_numpy(spiral_dict['r'].astype(np.float32)), torch.from_numpy(spiral_dict['theta'].astype(np.float32)))
                    batch_dicts.append(spiral_dict)
                    batch_series.append(x)
                    batch_counter += 1
                    spiral_counter += 1
                
                # Embed the batch using Chronos
                embeddings = pipeline.embed_batch(batch_series)
                
                for i, emb in enumerate(embeddings):
                    batch_dicts[i]['embedding'] = emb
                    
                # Save generated spirals
                for spiral_dict in batch_dicts:
                    idx = spiral_idx
                    spiral_idx += 1
                    save_path = f"{output_dir}/{args.spiral_type}_{idx}.npz"
                    save_npz_dict(save_path, spiral_dict)
            
        case "flat":
            spiral_idx = 0
            spiral_counter = 0
            while spiral_counter < args.num_spirals:
                batch_dicts = []
                batch_series = []
                batch_counter = 0
                while batch_counter < BATCH_SIZE and spiral_counter < args.num_spirals:
                    spiral_params = {
                        'steps_per_rotation': STEPS_PER_ROTATION,
                        'theta_start': np.random.uniform(MIN_THETA_RANGE[0], MIN_THETA_RANGE[1]),
                        'theta_end': np.random.uniform(MAX_THETA_RANGE[0], MAX_THETA_RANGE[1]),
                        'k': np.random.uniform(K_RANGE[0], K_RANGE[1])
                    }

                    while True:
                        flat_params = generete_flat_params(spiral_params, flat_params_deafult)
                        spiral_dict = generate_flat(spiral_params, flat_params)
                        num_of_intrsections = check_for_intersections(spiral_dict['theta'], spiral_dict['r'])
                        if num_of_intrsections <= 8:
                            break
                
                    x = prepare_chronos_input(torch.from_numpy(spiral_dict['r'].astype(np.float32)), torch.from_numpy(spiral_dict['theta'].astype(np.float32)))
                    batch_dicts.append(spiral_dict)
                    batch_series.append(x)
                    batch_counter += 1
                    spiral_counter += 1
                
                
                # Embed the batch using Chronos
                embeddings = pipeline.embed_batch(batch_series)
                
                for i, emb in enumerate(embeddings):
                    batch_dicts[i]['embedding'] = emb
                
                # Save generated spirals
                for spiral_dict in batch_dicts:
                    idx = spiral_idx
                    spiral_idx += 1
                    save_path = f"{output_dir}/{args.spiral_type}_{idx}.npz"
                    save_npz_dict(save_path, spiral_dict)
                
        case _:
            raise ValueError(f"Unsupported spiral type: {args.spiral_type}")


if __name__ == "__main__":
    main()