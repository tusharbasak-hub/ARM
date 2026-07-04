import os
import time
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import tensorflow as tf

# =====================================================================
# CONFIGURATION & PATHS
# =====================================================================
BASE_DATA_DIR = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\simulation\data"
MODEL_PATH = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model\iri_compliant\iri_background_model.tflite"
LUT_PATH = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model\iri_compliant\mobile_calibration_lut.json"
OUTPUT_DATA_DIR = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model\speed_invariance_test\data"
SAMPLES_TXT_PATH = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model\speed_invariance_test\samples.txt"

WINDOW_SIZE_M = 100.0
STEP_SIZE_M = 10.0
FINAL_SPATIAL_STEPS = 400
RAW_FEATURES = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']

# Hardcoded mapping based on user specifications
TRIPS_CONFIG = {
    'roamer_1': {
        'rel_path': 'IRI_old/roamer/automation_test_track/trip_1/readings.csv',
        'start': 2245,
        'end': 16820,
        'vehicle': 'Roamer SUV',
        'style': 'Safe / Standard'
    },
    'roamer_2': {
        'rel_path': 'IRI_old/roamer/automation_test_track/trip_2/readings.csv',
        'start': 2100,
        'end': 16915,
        'vehicle': 'Roamer SUV',
        'style': 'Dynamic / Fast'
    },
    'roamer_3': {
        'rel_path': 'IRI_old/roamer/automation_test_track/trip_3/readings.csv',
        'start': 1525,
        'end': 5315,
        'vehicle': 'Roamer SUV',
        'style': 'Rash / Aggressive'
    },
    'sunburst2_1': {
        'rel_path': 'IRI_new_experiments/sunburst2/automation_test_track/trip_1/readings_2.csv',
        'start': 2200,
        'end': 19960,
        'vehicle': 'Sunburst2 Sedan',
        'style': 'Safe / Standard'
    },
    'sunburst2_2': {
        'rel_path': 'IRI_new_experiments/sunburst2/automation_test_track/trip_2/readings_2.csv',
        'start': 2255,
        'end': 13235,
        'vehicle': 'Sunburst2 Sedan',
        'style': 'Rash / Aggressive'
    },
    'vivace_1': {
        'rel_path': 'IRI_new_experiments/vivace/automation_test_track/trip_1/readings_2.csv',
        'start': 1945,
        'end': 15915,
        'vehicle': 'Vivace Hatchback',
        'style': 'Safe / Standard'
    },
    'vivace_2': {
        'rel_path': 'IRI_new_experiments/vivace/automation_test_track/trip_2/readings_2.csv',
        'start': 2170,
        'end': 15610,
        'vehicle': 'Vivace Hatchback',
        'style': 'Dynamic / Fast'
    }
}

# =====================================================================
# FEATURE EXTRACTION & CALIBRATION SETUP
# =====================================================================
def extract_context_features(raw_data_window, speed_array):
    """Extracts 13 statistical and pseudo-spectral contextual features."""
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

print("[*] Loading TFLite Model and Calibration LUT...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

ctx_idx = input_details[0]['index']
raw_idx = input_details[1]['index']
out_idx = output_details[0]['index']

with open(LUT_PATH, 'r') as f:
    lut_data = json.load(f)
calibrator = interp1d(lut_data['x_raw'], lut_data['y_calibrated'], kind='linear', fill_value='extrapolate')

# =====================================================================
# PROCESSING TRIP DATA
# =====================================================================
def process_trip(trip_key, config):
    csv_path = os.path.join(BASE_DATA_DIR, config['rel_path'])
    print(f"\n[*] Processing {trip_key} ({config['vehicle']} - {config['style']})...")
    if not os.path.exists(csv_path):
        print(f"    [!] Error: File not found: {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    df_slice = df[(df['sample_number'] >= config['start']) & (df['sample_number'] <= config['end'])].copy().sort_values('sample_number')
    
    if len(df_slice) < 100:
        print(f"    [!] Warning: Slice too short ({len(df_slice)} samples). Skipping.")
        return None
        
    df_slice['time_s'] = (df_slice['sample_number'] - df_slice['sample_number'].min()) * 0.01 
    df_slice['dt'] = df_slice['time_s'].diff().fillna(0.01)
    df_slice['dx'] = df_slice['speed_ms'] * df_slice['dt']
    df_slice['cumulative_distance'] = df_slice['dx'].cumsum()
    
    max_dist = df_slice['cumulative_distance'].max()
    print(f"    -> Extracted {len(df_slice)} samples | Total distance: {max_dist:.1f} meters")
    
    records = []
    
    for start_dist in np.arange(0, max_dist - WINDOW_SIZE_M, STEP_SIZE_M):
        end_dist = start_dist + WINDOW_SIZE_M
        patch = df_slice[(df_slice['cumulative_distance'] >= start_dist) & (df_slice['cumulative_distance'] < end_dist)]
        
        if len(patch) < 20: continue
        
        target_iri = patch['IRI'].mean()
        mean_speed_ms = patch['speed_ms'].mean()
        mean_speed_kmh = mean_speed_ms * 3.6
        
        # Resample onto fixed spatial grid (400 steps over 100m)
        fixed_spatial_grid = np.linspace(start_dist, end_dist, FINAL_SPATIAL_STEPS)
        fixed_patch_features = []
        
        for feat in RAW_FEATURES:
            os_sensor = interp1d(patch['cumulative_distance'], patch[feat], kind='linear', bounds_error=False, fill_value=(patch[feat].iloc[0], patch[feat].iloc[-1]))
            fixed_patch_features.append(os_sensor(fixed_spatial_grid))
            
        os_speed = interp1d(patch['cumulative_distance'], patch['speed_ms'], kind='linear', bounds_error=False, fill_value=(patch['speed_ms'].iloc[0], patch['speed_ms'].iloc[-1]))
        spatial_speed = os_speed(fixed_spatial_grid)
        
        X_raw_6 = np.column_stack(fixed_patch_features).astype(np.float32)
        
        # Physics-informed speed normalization of vertical acceleration
        az_idx = RAW_FEATURES.index('az')
        v_safe = np.maximum(spatial_speed, 5.0)
        X_raw_6[:, az_idx] = X_raw_6[:, az_idx] * ((22.22 / v_safe) ** 2)
        
        X_ctx = extract_context_features(X_raw_6, spatial_speed)
        
        # TFLite Inference
        interpreter.set_tensor(ctx_idx, X_ctx.reshape(1, 13))
        interpreter.set_tensor(raw_idx, X_raw_6.reshape(1, 400, 6))
        interpreter.invoke()
        pred_log = float(interpreter.get_tensor(out_idx)[0, 0])
        pred_raw = float(np.expm1(pred_log))
        pred_calibrated = float(calibrator(pred_raw))
        
        # Center distance of window
        center_dist = start_dist + (WINDOW_SIZE_M / 2.0)
        
        records.append({
            'trip_key': trip_key,
            'vehicle': config['vehicle'],
            'style': config['style'],
            'distance_m': center_dist,
            'speed_kmh': mean_speed_kmh,
            'predicted_iri': pred_calibrated,
            'true_iri': target_iri
        })
        
    out_df = pd.DataFrame(records)
    out_file = os.path.join(OUTPUT_DATA_DIR, f"{trip_key}_aligned_iri.csv")
    out_df.to_csv(out_file, index=False)
    print(f"    [+] Saved {len(out_df)} window predictions to {out_file}")
    return out_df

if __name__ == "__main__":
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    all_dfs = []
    
    start_time = time.time()
    for trip_key, config in TRIPS_CONFIG.items():
        res = process_trip(trip_key, config)
        if res is not None:
            all_dfs.append(res)
            
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DATA_DIR, "all_trips_combined_iri.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"\n[+] Successfully compiled {len(combined_df)} total evaluation windows across all trips in {time.time()-start_time:.2f}s!")
        print(f"    Combined data written to: {combined_path}")
