import os
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed

# ==========================================
# Configuration and Hyperparameters
# ==========================================
BASE_DATA_DIR = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\simulation\data\IRI_new_experiments"
OUTPUT_DIR = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model\baseline\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Spatial Windowing Constants
WINDOW_SIZE_M = 100.0
STEP_SIZE_M = 10.0
FINAL_SPATIAL_STEPS = 400  # Resolution: 0.25m per step

# Filtering Constants
MIN_SPEED_MS = 1.0  # Threshold to drop near-stationary data

# Data Augmentation (Simulated Sampling Rates)
AUGMENT_HZ = [10, 15, 20, 25, 30, 50]

# Sensor Features
RAW_FEATURES = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']

# Train/Validation/Test Splits by Simulated Route
SPLITS = {
    'train': ['east_coast_usa', 'west_coast_usa'],
    'val': ['automation_test_track_trip_1'],
    'test': ['automation_test_track_trip_2', 'automation_test_track_trip_3']
}

def extract_context_features(raw_data_window, speed_array, distance_m):
    """
    Extracts 13 statistical and pseudo-spectral contextual features.
    raw_data_window: (400, 6) -> ax, ay, az, wx, wy, wz
    """
    ax, ay, az, wx, wy, wz = [raw_data_window[:, i] for i in range(6)]
    N = len(az)
    
    speed_mean = np.mean(speed_array)
    speed_std = np.std(speed_array)
    
    rms_az = np.sqrt(np.mean(az**2))
    rms_ay = np.sqrt(np.mean(ay**2))
    var_az = np.var(az)
    crest_factor_az = np.max(np.abs(az)) / (rms_az + 1e-6)
    
    zero_crossings_az = len(np.where(np.diff(np.sign(az)))[0])
    mcr_az = zero_crossings_az / N
    p2p_az = np.max(az) - np.min(az)
    
    rms_wz = np.sqrt(np.mean(wz**2))
    rms_wy = np.sqrt(np.mean(wy**2))
    mean_abs_ax = np.mean(np.abs(ax))
    
    # Pseudo-Spectral features (converting spatial frequencies to temporal)
    fft_vals = np.fft.rfft(az)
    psd = np.abs(fft_vals)**2
    freqs_spatial = np.fft.rfftfreq(N, d=0.25)
    
    if speed_mean > 0:
        freqs_temporal = freqs_spatial * speed_mean
        band_1_4 = np.sum(psd[(freqs_temporal >= 1.0) & (freqs_temporal <= 4.0)])
        band_4_15 = np.sum(psd[(freqs_temporal > 4.0) & (freqs_temporal <= 15.0)])
        total_energy = np.sum(psd) + 1e-6
        
        energy_ratio_1_4 = band_1_4 / total_energy
        energy_ratio_4_15 = band_4_15 / total_energy
    else:
        energy_ratio_1_4, energy_ratio_4_15 = 0.0, 0.0

    return np.array([
        speed_mean, speed_std, rms_az, rms_ay, var_az, crest_factor_az,
        mcr_az, p2p_az, rms_wz, rms_wy, mean_abs_ax, 
        energy_ratio_1_4, energy_ratio_4_15
    ], dtype=np.float32)

def compute_enriched_tabular_vector(raw_data_window, ctx_vector):
    """
    Computes 7 additional waveform quantiles and concatenates with the 13-dim ctx vector
    to form a 20-dim feature vector optimized for tree-based boosting baselines.
    """
    ax, ay, az, wx, wy, wz, speed_ms = [raw_data_window[:, i] for i in range(7)]
    
    max_abs_az = np.max(np.abs(az))
    p95_abs_az = np.percentile(np.abs(az), 95)
    skew_az = float(skew(az))
    kurt_az = float(kurtosis(az))
    max_abs_gx = np.max(np.abs(wx))
    max_abs_gy = np.max(np.abs(wy))
    
    vert_energy = np.sum(az**2)
    rot_energy = np.sum(wx**2 + wy**2) + 1e-6
    kinetic_energy_ratio = vert_energy / rot_energy
    
    extra_features = np.array([
        max_abs_az, p95_abs_az, skew_az, kurt_az, max_abs_gx, max_abs_gy, kinetic_energy_ratio
    ], dtype=np.float32)
    
    return np.concatenate([ctx_vector, extra_features])

def process_trip_data(df, trip_identifier):
    """Parses trip telemetry, filters slow speeds, and applies spatial windowing."""
    X_raw_list, X_ctx_list, X_tab_list, y_list = [], [], [], []
    
    df = df[df['speed_ms'] >= MIN_SPEED_MS].copy().sort_values('sample_number')
    if len(df) < 50:
        return [], [], [], []
        
    df['time_s'] = df['sample_number'] * 0.01 
    df['dt'] = df['time_s'].diff().fillna(0.01)
    df['dx'] = df['speed_ms'] * df['dt']
    df['cumulative_distance'] = df['dx'].cumsum()
    
    max_dist = df['cumulative_distance'].max()
    
    for start_dist in np.arange(0, max_dist - WINDOW_SIZE_M, STEP_SIZE_M):
        end_dist = start_dist + WINDOW_SIZE_M
        patch = df[(df['cumulative_distance'] >= start_dist) & (df['cumulative_distance'] < end_dist)]
        
        if len(patch) < 20: continue 
        
        target_iri = patch['IRI'].mean()
        start_time, end_time = patch['time_s'].min(), patch['time_s'].max()
        
        # Temporal jitter and augmentation
        target_hz = np.random.choice(AUGMENT_HZ)
        poll_interval = 1.0 / target_hz
        t_polls = np.arange(start_time, end_time, poll_interval)
        if len(t_polls) < 5: continue
        t_polls += np.random.uniform(0, poll_interval * 0.2, size=len(t_polls)) 
        
        augmented_features = []
        for feat in RAW_FEATURES:
            os_sensor = interp1d(patch['time_s'], patch[feat], kind='previous', fill_value="extrapolate")
            augmented_features.append(os_sensor(t_polls))
            
        os_speed = interp1d(patch['time_s'], patch['speed_ms'], kind='linear', fill_value="extrapolate")
        augmented_speed = os_speed(t_polls)
        
        os_dist = interp1d(patch['time_s'], patch['cumulative_distance'], kind='linear', fill_value="extrapolate")
        augmented_dists = os_dist(t_polls)
        
        # Spatial Domain Resampling to fixed grid
        fixed_spatial_grid = np.linspace(start_dist, end_dist, FINAL_SPATIAL_STEPS)
        fixed_patch_features = []
        
        for i in range(len(RAW_FEATURES)):
            spatial_fix = interp1d(
                augmented_dists, 
                augmented_features[i], 
                kind='linear', 
                bounds_error=False, 
                fill_value=(augmented_features[i][0], augmented_features[i][-1])
            )
            fixed_patch_features.append(spatial_fix(fixed_spatial_grid))
            
        spatial_speed = interp1d(
            augmented_dists, 
            augmented_speed, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(augmented_speed[0], augmented_speed[-1])
        )(fixed_spatial_grid)
            
        X_raw_6 = np.column_stack(fixed_patch_features) 
        
        # Physics-informed Speed Normalization of Vertical Acceleration
        az_idx = RAW_FEATURES.index('az')
        v_safe = np.maximum(spatial_speed, 5.0) 
        X_raw_6[:, az_idx] = X_raw_6[:, az_idx] * ((22.22 / v_safe) ** 2)
        
        # Extract 13-dim context vector
        X_ctx = extract_context_features(X_raw_6, spatial_speed, start_dist)
        
        # Append speed as 7th channel for RNN/LSTM sequential baselines -> shape (400, 7)
        X_raw_7 = np.column_stack([X_raw_6, spatial_speed]).astype(np.float32)
        
        # Compute 20-dim enriched tabular vector for XGBoost/RF/LightGBM/CatBoost
        X_tab = compute_enriched_tabular_vector(X_raw_7, X_ctx)
        
        X_raw_list.append(X_raw_7)
        X_ctx_list.append(X_ctx)
        X_tab_list.append(X_tab)
        y_list.append(target_iri)
            
    return X_raw_list, X_ctx_list, X_tab_list, y_list

def worker_process_trip(car, route, trip):
    """Joblib worker function to process individual trips."""
    trip_key = f"{route}_{trip}"
    
    split = 'train'
    if route in SPLITS['train']:
        split = 'train'
    elif route == 'automation_test_track':
        if trip == 'trip_1': split = 'val'
        elif trip in ['trip_2', 'trip_3']: split = 'test'
        else: return None
    else:
        return None
        
    csv_path = os.path.join(BASE_DATA_DIR, car, route, trip, 'readings_2.csv')
    if not os.path.exists(csv_path): return None
        
    df = pd.read_csv(csv_path)
    x_r, x_c, x_t, y_val = process_trip_data(df, trip_key)
    
    if len(y_val) == 0: return None
    return {'split': split, 'raw': x_r, 'ctx': x_c, 'tab': x_t, 'y': y_val}

def build_datasets():
    """Compiles the dataset utilizing parallel processing across CPU cores."""
    datasets = {'train': {'raw': [], 'ctx': [], 'tab': [], 'y': []},
                'val':   {'raw': [], 'ctx': [], 'tab': [], 'y': []},
                'test':  {'raw': [], 'ctx': [], 'tab': [], 'y': []}}
    
    tasks = []
    if os.path.exists(BASE_DATA_DIR):
        for car in sorted(os.listdir(BASE_DATA_DIR)):
            car_p = os.path.join(BASE_DATA_DIR, car)
            if not os.path.isdir(car_p) or car == 'desktop.ini': continue
            for route in sorted(os.listdir(car_p)):
                route_p = os.path.join(car_p, route)
                if not os.path.isdir(route_p) or route == 'desktop.ini': continue
                for trip in sorted(os.listdir(route_p)):
                    if os.path.isdir(os.path.join(route_p, trip)):
                        tasks.append((car, route, trip))
                
    print(f"[*] Dispatching {len(tasks)} trips from {BASE_DATA_DIR} to worker pool...")
    start_time = time.time()
    
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(worker_process_trip)(car, route, trip) for car, route, trip in tasks
    )
    
    for res in results:
        if res:
            split = res['split']
            datasets[split]['raw'].extend(res['raw'])
            datasets[split]['ctx'].extend(res['ctx'])
            datasets[split]['tab'].extend(res['tab'])
            datasets[split]['y'].extend(res['y'])

    for split in ['train', 'val', 'test']:
        raw_arr = np.array(datasets[split]['raw'], dtype=np.float32)
        ctx_arr = np.array(datasets[split]['ctx'], dtype=np.float32)
        tab_arr = np.array(datasets[split]['tab'], dtype=np.float32)
        y_arr = np.array(datasets[split]['y'], dtype=np.float32)
        
        print(f"[{split.upper()}] -> Raw: {raw_arr.shape}, Ctx: {ctx_arr.shape}, Tab: {tab_arr.shape}, y: {y_arr.shape}")
        
        out_file = os.path.join(OUTPUT_DIR, f"{split}_data.npz")
        np.savez_compressed(out_file, raw=raw_arr, ctx=ctx_arr, tab=tab_arr, y=y_arr)
        print(f"    [+] Saved compressed buffer to {out_file} ({os.path.getsize(out_file) / (1024*1024):.2f} MB)")
        
    print(f"[*] Dataset compilation completed in {time.time() - start_time:.2f}s")
    return datasets

if __name__ == '__main__':
    build_datasets()
