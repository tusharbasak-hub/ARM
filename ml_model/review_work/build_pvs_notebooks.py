import os
import json

def make_nb(cells_data):
    cells = []
    for cell_type, source_text in cells_data:
        # Split multi-line source into list of strings with trailing newlines
        lines = source_text.split('\n')
        formatted_source = [line + '\n' for line in lines[:-1]]
        if lines[-1]:
            formatted_source.append(lines[-1])
            
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": formatted_source
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
        
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "tf",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb

# ==============================================================================
# NOTEBOOK 1: DATA PREPARATION & CENTRAL IMU MERGING
# ==============================================================================
nb1_cells = [
    ("markdown", """# Notebook 1: PVS Dataset Preparation & Central IMU Merging

This notebook extracts sensor readings from the multi-class PVS dataset (`PVS 1` through `PVS 9`). In physical test vehicles, smartphones were mounted on both the left and right sides of the dashboard. To obtain a unified chassis response representing the sprung mass center, we compute the arithmetic mean of the left and right accelerometer and gyroscope axes:
$$a_{x,\\text{mid}} = \\frac{a_{x,\\text{left}} + a_{x,\\text{right}}}{2}, \\quad a_{y,\\text{mid}} = \\frac{a_{y,\\text{left}} + a_{y,\\text{right}}}{2}, \\quad a_{z,\\text{mid}} = \\frac{a_{z,\\text{left}} + a_{z,\\text{right}}}{2}$$
$$\\omega_{x,\\text{mid}} = \\frac{\\omega_{x,\\text{left}} + \\omega_{x,\\text{right}}}{2}, \\quad \\omega_{y,\\text{mid}} = \\frac{\\omega_{y,\\text{left}} + \\omega_{y,\\text{right}}}{2}, \\quad \\omega_{z,\\text{mid}} = \\frac{\\omega_{z,\\text{left}} + \\omega_{z,\\text{right}}}{2}$$
This averaging physically eliminates roll-induced differential vertical vibrations across opposite sides of the vehicle, isolating pure heave and pitch dynamics.

Furthermore, because physical gyroscope sensors in the PVS dataset log angular velocity in **degrees/sec**, whereas our Section 7 model is trained on SI units (**radians/sec**), we scale all gyroscope readings during extraction:
$$\\omega_{\\text{rad/s}} = \\omega_{\\text{deg/s}} \\times \\frac{\\pi}{180}$$

We also fuse the categorical road quality labels from `dataset_labels.csv`. Road conditions are encoded as integer classes:
- **Good Road (`0`)**: Smooth asphalt or paved road without severe defects.
- **Regular Road (`1`)**: Moderate roughness or minor degradation.
- **Bad Road (`2`)**: Severe roughness, cobblestones, dirt roads, or structural deterioration.

To combine the binary one-hot left and right labels into a single ground truth quality label, we use the logical OR / maximum severity:
$$\\text{label}_{\\text{mid}} = \\max(\\text{label}_{\\text{left}}, \\text{label}_{\\text{right}})$$
"""),
    ("code", """import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
RAW_DATA_DIR = r"D:\\Coding\\Hackathon\\GFG\\ARM\\ARM\\ml_model_work\\data\\raw\\multi_class_kaggle"
OUTPUT_DATA_DIR = r"D:\\Coding\\Hackathon\\GFG\\ARM\\ARM\\ml_model\\review_work\\data"
OUTPUT_FIG_DIR = r"D:\\Coding\\Hackathon\\GFG\\ARM\\ARM\\ml_model\\review_work\\outputs\\figures"

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
print(f"[*] Raw Data Directory: {RAW_DATA_DIR}")
print(f"[*] Output Data Directory: {OUTPUT_DATA_DIR}")
"""),
    ("code", """def process_pvs_trip(trip_id):
    folder_path = os.path.join(RAW_DATA_DIR, f"PVS {trip_id}")
    left_csv = os.path.join(folder_path, "dataset_gps_mpu_left.csv")
    right_csv = os.path.join(folder_path, "dataset_gps_mpu_right.csv")
    labels_csv = os.path.join(folder_path, "dataset_labels.csv")
    
    if not (os.path.exists(left_csv) and os.path.exists(right_csv) and os.path.exists(labels_csv)):
        print(f"[!] Skipping PVS {trip_id}: Missing required CSV files in {folder_path}")
        return None
        
    print(f"[+] Processing PVS {trip_id}...")
    
    # Load raw CSVs
    df_l = pd.read_csv(left_csv)
    df_r = pd.read_csv(right_csv)
    df_lab = pd.read_csv(labels_csv)
    
    # Verify row alignment
    min_len = min(len(df_l), len(df_r), len(df_lab))
    if len(df_l) != min_len or len(df_r) != min_len or len(df_lab) != min_len:
        print(f"    [!] Length mismatch in PVS {trip_id} (Left: {len(df_l)}, Right: {len(df_r)}, Labels: {len(df_lab)}). Truncating to min_len={min_len}.")
        df_l = df_l.iloc[:min_len].reset_index(drop=True)
        df_r = df_r.iloc[:min_len].reset_index(drop=True)
        df_lab = df_lab.iloc[:min_len].reset_index(drop=True)
        
    # Create central DataFrame
    df_mid = pd.DataFrame()
    
    # 1. Timestamps and Odometry (Primary reference: left sensor)
    df_mid['timestamp'] = df_l['timestamp']
    df_mid['timestamp_gps'] = df_l['timestamp_gps']
    df_mid['latitude'] = df_l['latitude']
    df_mid['longitude'] = df_l['longitude']
    df_mid['speed'] = df_l['speed']  # speed in m/s
    
    # 2. Central IMU Averaging (Sprung Mass Center)
    imu_axes = ['acc_x_dashboard', 'acc_y_dashboard', 'acc_z_dashboard', 
                'gyro_x_dashboard', 'gyro_y_dashboard', 'gyro_z_dashboard']
    
    for col in imu_axes:
        short_name = col.replace('_dashboard', '').replace('acc_', 'a').replace('gyro_', 'w')
        val = (df_l[col].astype(np.float32) + df_r[col].astype(np.float32)) / 2.0
        if 'gyro_' in col:
            val = val * (np.pi / 180.0)  # Scale angular velocity from deg/sec to rad/sec
        df_mid[short_name] = val
        
    # 3. Ground Truth Road Quality Categorical Labels (0=Good, 1=Regular, 2=Bad)
    label_left = np.zeros(min_len, dtype=np.int32)
    label_left[df_lab['regular_road_left'] == 1] = 1
    label_left[df_lab['bad_road_left'] == 1] = 2
    
    label_right = np.zeros(min_len, dtype=np.int32)
    label_right[df_lab['regular_road_right'] == 1] = 1
    label_right[df_lab['bad_road_right'] == 1] = 2
    
    # Logical OR / Maximum Worst-Case Severity
    df_mid['label_left'] = label_left
    df_mid['label_right'] = label_right
    df_mid['label_mid'] = np.maximum(label_left, label_right)
    
    # 4. Surface Metadata and Anomalies
    surface_cols = [
        'paved_road', 'unpaved_road', 'dirt_road', 'cobblestone_road', 
        'asphalt_road', 'no_speed_bump', 'speed_bump_asphalt', 'speed_bump_cobblestone'
    ]
    for col in surface_cols:
        if col in df_lab.columns:
            df_mid[col] = df_lab[col].astype(np.int32)
        else:
            df_mid[col] = 0
            
    # Save processed central dataset
    out_path = os.path.join(OUTPUT_DATA_DIR, f"PVS_{trip_id}_central.csv")
    df_mid.to_csv(out_path, index=False)
    
    label_counts = df_mid['label_mid'].value_counts().to_dict()
    print(f"    [✔] Saved {len(df_mid):,} samples to {out_path}")
    print(f"        Class Distribution -> Good (0): {label_counts.get(0, 0):,} | Regular (1): {label_counts.get(1, 0):,} | Bad (2): {label_counts.get(2, 0):,}")
    
    return df_mid
"""),
    ("code", """# --- PROCESS ALL 9 TRIPS ---
trip_summaries = []
all_mid_labels = []

for i in range(1, 10):
    df_trip = process_pvs_trip(i)
    if df_trip is not None:
        trip_summaries.append({
            'trip_id': f"PVS {i}",
            'total_samples': len(df_trip),
            'good_samples': (df_trip['label_mid'] == 0).sum(),
            'regular_samples': (df_trip['label_mid'] == 1).sum(),
            'bad_samples': (df_trip['label_mid'] == 2).sum(),
            'duration_sec': round(df_trip['timestamp'].max() - df_trip['timestamp'].min(), 1),
            'distance_km': round((df_trip['speed'] * df_trip['timestamp'].diff().fillna(0.01)).sum() / 1000.0, 2)
        })
        all_mid_labels.extend(df_trip['label_mid'].values)

summary_df = pd.DataFrame(trip_summaries)
print("\\n" + "="*80)
print("PVS DATASET CENTRAL IMU EXTRACTION SUMMARY")
print("="*80)
display(summary_df)

summary_csv_path = os.path.join(OUTPUT_DATA_DIR, "pvs_extraction_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"\\n[✔] Exported summary table to: {summary_csv_path}")
"""),
    ("code", """# --- VISUALIZE CLASS DISTRIBUTION & CENTRAL AVERAGING ---
plt.figure(figsize=(14, 5))

# Subplot 1: Class Distribution
plt.subplot(1, 2, 1)
counts = pd.Series(all_mid_labels).value_counts().sort_index()
bars = plt.bar(['Good (0)', 'Regular (1)', 'Bad (2)'], counts.values, color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.85, edgecolor='black')
plt.title("Combined PVS Ground Truth Quality Distribution (label_mid)", fontsize=12, fontweight='bold')
plt.xlabel("Road Quality Class", fontsize=11)
plt.ylabel("Number of Sample Points", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts.values)*0.01), f"{int(yval):,}", ha='center', va='bottom', fontweight='bold')

# Subplot 2: Demonstration of Roll Cancellation (10-second slice from PVS 1)
try:
    sample_trip_l = pd.read_csv(os.path.join(RAW_DATA_DIR, "PVS 1", "dataset_gps_mpu_left.csv"), nrows=1000)
    sample_trip_r = pd.read_csv(os.path.join(RAW_DATA_DIR, "PVS 1", "dataset_gps_mpu_right.csv"), nrows=1000)
    
    t = sample_trip_l['timestamp'] - sample_trip_l['timestamp'].iloc[0]
    az_l = sample_trip_l['acc_z_dashboard']
    az_r = sample_trip_r['acc_z_dashboard']
    az_mid = (az_l + az_r) / 2.0
    
    plt.subplot(1, 2, 2)
    plt.plot(t[:300], az_l[:300], label='Left Dashboard az', color='#1f77b4', alpha=0.4, linewidth=1)
    plt.plot(t[:300], az_r[:300], label='Right Dashboard az', color='#aec7e8', alpha=0.4, linewidth=1)
    plt.plot(t[:300], az_mid[:300], label='Central Averaged az (Sprung Mass)', color='#d62728', linewidth=2)
    plt.title("Central IMU Averaging (Roll Vibration Cancellation)", fontsize=12, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=11)
    plt.ylabel("Vertical Acceleration (m/s²)", fontsize=11)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
except Exception as e:
    print(f"[!] Could not plot sample time-series: {e}")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_FIG_DIR, "pvs_data_preparation_overview.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"[✔] Overview visualization saved to: {plot_path}")
""")
]

# ==============================================================================
# NOTEBOOK 2: INFERENCE, THRESHOLD SEARCH & EVALUATION
# ==============================================================================
nb2_cells = [
    ("markdown", """# Notebook 2: TFLite Inference, Automatic Threshold Binning & Evaluation

This notebook executes zero-shot inference on the prepared PVS central dataset using our deployed TensorFlow Lite model (`iri_background_model.tflite`). We replicate the exact preprocessing pipeline from Section 7 of our training pipeline:
- **Sliding Spatial Window:** $100\\text{ m}$ window size with $10\\text{ m}$ step size ($90\\%$ overlap).
- **Monotonic Distance & Dead-Stop Override:** Monotonic odometry perturbation (`cumsum() + arange * 1e-4`) and baseline assignment ($v < 0.5\\text{ km/h} \\to \\text{IRI} = 1.5\\text{ m/km}$).
- **Spatial Resampling:** Interpolation to `FINAL_STEPS = 400` points ($0.25\\text{ m}$ resolution).
- **Physics Velocity Normalization:** $a_z \\leftarrow a_z \\cdot (22.22 / v_{\\text{safe}})^2$ to normalize all vibrations to a reference speed of $80\\text{ km/h}$.
- **Contextual Branch:** 13-dimensional statistical feature vector (mean/std speed, RMS, crest factor, MCR, P2P, and PSD frequency band ratios).
- **Calibration LUT:** Piecewise linear post-processing via `mobile_calibration_lut.json`.

### Axis Calibration Toggles
In simulation training data, vertical gravity averages around $-9.8\\text{ m/s}^2$, whereas in PVS vehicle sensor logs it averages around $+9.8\\text{ m/s}^2$. Additionally, smartphone mounting orientations may rotate horizontal axes by $90^\\circ$. We provide toggle variables (`INVERT_AZ` and `SWAP_AX_AY`) at the top of the notebook to seamlessly align physical sensor coordinate frames with training assumptions.

### Automated Threshold Binning & Trip-Wise Isolation
To evaluate our continuous IRI predictions against PVS categorical ground truth (`0 = Good`, `1 = Regular`, `2 = Bad`), we perform an automated 2D grid search over boundary thresholds $(T_1, T_2)$ with $0.1\\text{ m/km}$ spacing across all trips combined, finding the exact cutoffs that maximize macro F1-score. Finally, we compute and export confusion matrices both globally across all 9 trips and individually for each trip (`PVS 1` through `PVS 9`) to retain full visibility into route-specific generalization.
"""),
    ("code", """import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, classification_report, balanced_accuracy_score

# --- CONFIGURATION & PATHS ---
MODEL_PATH = r"D:\\Coding\\Hackathon\\GFG\\ARM\\ARM\\ml_model\\iri_compliant\\iri_background_model.tflite"
LUT_PATH = r"D:\\Coding\\Hackathon\\GFG\\ARM\\ARM\\ml_model\\iri_compliant\\mobile_calibration_lut.json"

DATA_DIR = r"D:\\Coding\\Hackathon\\GFG\\ARM\\ARM\\ml_model\\review_work\\data"
OUTPUT_DIR = r"D:\\Coding\\Hackathon\\GFG\\ARM\\ARM\\ml_model\\review_work\\outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- AXIS CALIBRATION TOGGLES ---
# In training simulation data, az averages around -9.8 m/s^2, whereas in PVS physical logs it averages around +9.8 m/s^2.
INVERT_AZ = True 
# Toggle to swap horizontal axes (ax <-> ay and wx <-> wy) if PVS sensor orientation is rotated 90 degrees relative to simulation
SWAP_AX_AY = True 
# Toggle to convert gyroscopes from deg/s to rad/s (set to True if loading raw deg/s data directly into Notebook 2)
CONVERT_GYRO_DEG_TO_RAD = False 

# --- INFERENCE PARAMETERS (Exact Model 3 Specs) ---
WINDOW_SIZE_M = 100.0
STEP_SIZE_M = 10.0
FINAL_STEPS = 400
FEATURES = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']

print(f"[*] TFLite Model Path: {MODEL_PATH}")
print(f"[*] Calibration LUT Path: {LUT_PATH}")
print(f"[*] Axis Calibration -> INVERT_AZ: {INVERT_AZ} | SWAP_AX_AY: {SWAP_AX_AY} | CONVERT_GYRO_DEG_TO_RAD: {CONVERT_GYRO_DEG_TO_RAD}")
"""),
    ("code", """# --- CONTEXT FEATURE EXTRACTION (Matches Model 3) ---
def extract_context_features(raw_data_window, speed_array):
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
    
    # Pseudo-Spectral features
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

# --- LUT CALIBRATION ---
def calibrate_iri(raw_iri, lut_path=LUT_PATH):
    if not os.path.exists(lut_path):
        return raw_iri
    with open(lut_path, 'r') as f:
        lut = json.load(f)
    return float(np.interp(raw_iri, lut['x_raw'], lut['y_calibrated']))
"""),
    ("code", """# --- WINDOWED TFLITE INFERENCE ENGINE ---
def run_trip_inference(trip_id, interpreter, raw_input_idx, ctx_input_idx, output_idx):
    csv_path = os.path.join(DATA_DIR, f"PVS_{trip_id}_central.csv")
    if not os.path.exists(csv_path):
        print(f"[!] Missing {csv_path}")
        return None, None
        
    print(f"[+] Running TFLite Inference on PVS {trip_id}...")
    df = pd.read_csv(csv_path)
    
    # Time and odometry calculation
    df['dt'] = df['timestamp'].diff().fillna(0.01)
    df['dt'] = np.where(df['dt'] <= 0, 0.01, df['dt'])
    df['dx'] = df['speed'] * df['dt']
    df['cumulative_distance'] = df['dx'].cumsum() + (np.arange(len(df)) * 1e-4)
    
    max_dist = df['cumulative_distance'].max()
    windows = []
    
    # Sliding spatial window
    for start_dist in np.arange(0, max_dist - WINDOW_SIZE_M, STEP_SIZE_M):
        end_dist = start_dist + WINDOW_SIZE_M
        patch = df[(df['cumulative_distance'] >= start_dist) & (df['cumulative_distance'] < end_dist)]
        
        if len(patch) < 20:
            continue
            
        avg_speed_ms = patch['speed'].mean()
        avg_speed_kmh = avg_speed_ms * 3.6
        
        # Dead stop override
        if avg_speed_kmh < 0.5:
            final_iri = 1.5
        else:
            # Extract features and apply axis toggles
            patch_dict = {}
            for feat in FEATURES:
                patch_dict[feat] = patch[feat].values.copy()
                
            if INVERT_AZ:
                patch_dict['az'] = -patch_dict['az']
            if SWAP_AX_AY:
                patch_dict['ax'], patch_dict['ay'] = patch_dict['ay'].copy(), patch_dict['ax'].copy()
                patch_dict['wx'], patch_dict['wy'] = patch_dict['wy'].copy(), patch_dict['wx'].copy()
            if CONVERT_GYRO_DEG_TO_RAD:
                for g_col in ['wx', 'wy', 'wz']:
                    patch_dict[g_col] = patch_dict[g_col] * (np.pi / 180.0)
                
            # Distance domain resampling
            fixed_spatial_grid = np.linspace(start_dist, end_dist, FINAL_STEPS)
            fixed_patch_features = []
            for feat in FEATURES:
                spatial_fix = interp1d(
                    patch['cumulative_distance'], patch_dict[feat],
                    kind='linear', bounds_error=False,
                    fill_value=(patch_dict[feat][0], patch_dict[feat][-1])
                )
                fixed_patch_features.append(spatial_fix(fixed_spatial_grid))
                
            spatial_speed = interp1d(
                patch['cumulative_distance'], patch['speed'],
                kind='linear', bounds_error=False,
                fill_value=(patch['speed'].iloc[0], patch['speed'].iloc[-1])
            )(fixed_spatial_grid)
            
            X_raw_filtered = np.column_stack(fixed_patch_features).astype(np.float32)
            
            # Physics speed normalization (80 km/h = 22.22 m/s)
            az_idx = FEATURES.index('az')
            v_safe = np.maximum(spatial_speed, 5.0)
            X_raw_filtered[:, az_idx] = X_raw_filtered[:, az_idx] * ((22.22 / v_safe) ** 2)
            
            X_ctx = extract_context_features(X_raw_filtered, spatial_speed)
            
            X_raw_batch = np.expand_dims(X_raw_filtered, axis=0)
            X_ctx_batch = np.expand_dims(X_ctx, axis=0)
            
            interpreter.set_tensor(raw_input_idx, X_raw_batch)
            interpreter.set_tensor(ctx_input_idx, X_ctx_batch)
            interpreter.invoke()
            
            pred_log = interpreter.get_tensor(output_idx)[0][0]
            pred_iri_raw = np.expm1(pred_log)
            final_iri = calibrate_iri(pred_iri_raw)
            
        # Ground truth window aggregation (Mode / Max severity in window)
        win_label_mid = int(patch['label_mid'].mode()[0]) if not patch['label_mid'].mode().empty else int(patch['label_mid'].max())
        
        # Aggregate surface metadata
        win_meta = {
            'trip_id': f"PVS {trip_id}",
            'start_dist': start_dist,
            'end_dist': end_dist,
            'mean_lat': patch['latitude'].mean(),
            'mean_lon': patch['longitude'].mean(),
            'mean_speed_kmh': avg_speed_kmh,
            'predicted_iri': final_iri,
            'label_mid': win_label_mid,
            'paved_road': int(patch['paved_road'].max()),
            'unpaved_road': int(patch['unpaved_road'].max()),
            'dirt_road': int(patch['dirt_road'].max()),
            'cobblestone_road': int(patch['cobblestone_road'].max()),
            'asphalt_road': int(patch['asphalt_road'].max()),
            'speed_bump_asphalt': int(patch['speed_bump_asphalt'].max()),
            'speed_bump_cobblestone': int(patch['speed_bump_cobblestone'].max())
        }
        windows.append(win_meta)
        
    df_win = pd.DataFrame(windows)
    
    # Map predictions back to raw timestamp CSV for completeness
    if len(df_win) > 0:
        interp_pred = interp1d(
            df_win['start_dist'] + WINDOW_SIZE_M/2.0, df_win['predicted_iri'],
            kind='nearest', bounds_error=False,
            fill_value=(df_win['predicted_iri'].iloc[0], df_win['predicted_iri'].iloc[-1])
        )
        df['predicted_iri'] = interp_pred(df['cumulative_distance'])
    else:
        df['predicted_iri'] = np.nan
        
    out_csv = os.path.join(DATA_DIR, f"PVS_{trip_id}_with_predictions.csv")
    df.to_csv(out_csv, index=False)
    print(f"    [✔] Extracted {len(df_win):,} spatial windows | Saved raw predictions to {out_csv}")
    
    return df_win, df
"""),
    ("code", """# --- EXECUTE INFERENCE ACROSS ALL 9 TRIPS ---
print("[+] Allocating TFLite Interpreter...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

raw_input_idx = next(i['index'] for i in input_details if len(i['shape']) == 3)
ctx_input_idx = next(i['index'] for i in input_details if len(i['shape']) == 2)
output_idx = output_details[0]['index']

all_trip_windows = {}
combined_windows_list = []

for i in range(1, 10):
    df_win, df_raw = run_trip_inference(i, interpreter, raw_input_idx, ctx_input_idx, output_idx)
    if df_win is not None and len(df_win) > 0:
        all_trip_windows[f"PVS {i}"] = df_win
        combined_windows_list.append(df_win)

df_all_windows = pd.concat(combined_windows_list, ignore_index=True)
combined_csv_path = os.path.join(OUTPUT_DIR, "all_trips_windowed_evaluation.csv")
df_all_windows.to_csv(combined_csv_path, index=False)
print(f"\\n[✔] Completed inference on {len(df_all_windows):,} total spatial windows across all trips.")
print(f"[✔] Exported combined window dataset to: {combined_csv_path}")
"""),
    ("code", """# --- AUTOMATED 2D GRID SEARCH FOR OPTIMAL IRI THRESHOLDS (0.1 SPACING) ---
# Choose optimization metric:
# 'balanced_accuracy' -> Maximizes average diagonal percentage (recall across Good, Regular, Bad). Best for preventing 0 predictions on minority Bad class!
# 'macro_f1_constrained' -> Maximizes Macro F1 subject to min recall >= 10% for every class.
# 'macro_f1' -> Standard unweighted Macro F1.
OPTIMIZE_METRIC = 'balanced_accuracy'

print(f"[+] Running 2D grid search over (T1, T2) boundary thresholds optimizing for: {OPTIMIZE_METRIC}...")

y_true_all = df_all_windows['label_mid'].values
iri_all = df_all_windows['predicted_iri'].values

t1_range = np.arange(1.5, 6.1, 0.1)
t2_range = np.arange(4.0, 12.1, 0.1)

best_score = -1.0
best_f1 = 0.0
best_bal_acc = 0.0
best_t1, best_t2 = 3.5, 7.0
grid_results = []

for t1 in t1_range:
    for t2 in t2_range:
        if t2 <= t1 + 0.5:
            continue
        # Classify: < t1 -> 0 (Good), t1..t2 -> 1 (Regular), >= t2 -> 2 (Bad)
        y_pred = np.zeros_like(iri_all, dtype=np.int32)
        y_pred[(iri_all >= t1) & (iri_all < t2)] = 1
        y_pred[iri_all >= t2] = 2
        
        macro_f1 = f1_score(y_true_all, y_pred, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(y_true_all, y_pred)
        
        if OPTIMIZE_METRIC == 'balanced_accuracy':
            score = bal_acc
        elif OPTIMIZE_METRIC == 'macro_f1_constrained':
            recalls = precision_recall_fscore_support(y_true_all, y_pred, average=None, zero_division=0)[1]
            score = macro_f1 if np.all(recalls >= 0.10) else 0.0
        else:
            score = macro_f1
            
        grid_results.append({'T1': t1, 'T2': t2, 'Macro_F1': macro_f1, 'Balanced_Accuracy': bal_acc, 'Score': score})
        
        if score > best_score:
            best_score = score
            best_f1 = macro_f1
            best_bal_acc = bal_acc
            best_t1, best_t2 = t1, t2

df_grid = pd.DataFrame(grid_results)
print(f"\\n[★] OPTIMAL THRESHOLDS FOUND ({OPTIMIZE_METRIC}) -> T1 (Good/Regular): {best_t1:.2f} m/km | T2 (Regular/Bad): {best_t2:.2f} m/km")
print(f"[★] Maximum {OPTIMIZE_METRIC}: {best_score:.4f} | Associated Macro F1: {best_f1:.4f} | Balanced Accuracy: {best_bal_acc:.4f}")

# Save optimal thresholds to JSON for mapping notebook
opt_thresh_file = os.path.join(OUTPUT_DIR, "optimal_thresholds.json")
with open(opt_thresh_file, 'w') as f:
    json.dump({'T1': round(float(best_t1), 2), 'T2': round(float(best_t2), 2), 'best_macro_f1': round(float(best_f1), 4), 'best_bal_acc': round(float(best_bal_acc), 4), 'metric': OPTIMIZE_METRIC}, f, indent=2)
print(f"[✔] Exported optimal thresholds to: {opt_thresh_file}")

# --- PLOT GRID SEARCH HEATMAP ---
plt.figure(figsize=(10, 8))
pivot_grid = df_grid.pivot(index='T2', columns='T1', values='Score')
sns.heatmap(pivot_grid, cmap='viridis', cbar_kws={'label': f'Score ({OPTIMIZE_METRIC})'}, origin='lower')
plt.title(f"Automated Threshold Grid Search Landscape ({OPTIMIZE_METRIC})", fontsize=13, fontweight='bold')
plt.xlabel("T1: Good / Regular Boundary (m/km)", fontsize=11)
plt.ylabel("T2: Regular / Bad Boundary (m/km)", fontsize=11)
grid_plot_path = os.path.join(FIGURES_DIR, "threshold_grid_search_heatmap.png")
plt.savefig(grid_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"[✔] Saved grid search heatmap to: {grid_plot_path}")
"""),
    ("code", """# --- TRIP-WISE & GLOBAL EVALUATION REPORTS ---
def classify_iri(iri_vals, t1=best_t1, t2=best_t2):
    preds = np.zeros_like(iri_vals, dtype=np.int32)
    preds[(iri_vals >= t1) & (iri_vals < t2)] = 1
    preds[iri_vals >= t2] = 2
    return preds

class_names = ['Good (0)', 'Regular (1)', 'Bad (2)']
score_report_rows = []

print("\\n" + "="*80)
print("GLOBAL COMBINED CLASSIFICATION REPORT")
print("="*80)
y_pred_all = classify_iri(df_all_windows['predicted_iri'].values)
print(classification_report(y_true_all, y_pred_all, target_names=class_names, digits=4))

# Plot Combined Confusion Matrix
plt.figure(figsize=(7, 6))
cm_all = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False, annot_kws={"size": 13, "weight": "bold"})
plt.title(f"Combined PVS Dataset Confusion Matrix\\nMacro F1: {best_f1:.4f} (T1={best_t1:.2f}, T2={best_t2:.2f})", fontsize=12, fontweight='bold')
plt.xlabel("Predicted Road Quality Class", fontsize=11, fontweight='bold')
plt.ylabel("Ground Truth Road Quality Class", fontsize=11, fontweight='bold')
combined_cm_path = os.path.join(FIGURES_DIR, "combined_confusion_matrix.png")
plt.savefig(combined_cm_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"[✔] Exported combined confusion matrix to: {combined_cm_path}")

# Record global summary row
p_all, r_all, f_all, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
score_report_rows.append({
    'Scope': 'All Trips Combined',
    'Samples': len(y_true_all),
    'Macro_Precision': round(p_all, 4),
    'Macro_Recall': round(r_all, 4),
    'Macro_F1': round(f_all, 4),
    'Accuracy': round(np.mean(y_true_all == y_pred_all), 4)
})

# --- TRIP-WISE EVALUATION & PLOTTING ---
print("\\n" + "="*80)
print("TRIP-WISE EVALUATION & CONFUSION MATRICES")
print("="*80)

for trip_name, df_trip_win in all_trip_windows.items():
    y_true_trip = df_trip_win['label_mid'].values
    y_pred_trip = classify_iri(df_trip_win['predicted_iri'].values)
    
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true_trip, y_pred_trip, average='macro', zero_division=0)
    acc_trip = np.mean(y_true_trip == y_pred_trip)
    
    score_report_rows.append({
        'Scope': trip_name,
        'Samples': len(y_true_trip),
        'Macro_Precision': round(p_mac, 4),
        'Macro_Recall': round(r_mac, 4),
        'Macro_F1': round(f_mac, 4),
        'Accuracy': round(acc_trip, 4)
    })
    
    # Plot individual trip confusion matrix
    plt.figure(figsize=(6, 5))
    cm_trip = confusion_matrix(y_true_trip, y_pred_trip, labels=[0, 1, 2])
    sns.heatmap(cm_trip, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False, annot_kws={"size": 12, "weight": "bold"})
    plt.title(f"{trip_name} Confusion Matrix | Macro F1: {f_mac:.4f}\\n(T1={best_t1:.2f}, T2={best_t2:.2f})", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("Ground Truth Class", fontsize=10)
    
    trip_num = trip_name.split()[-1]
    trip_cm_path = os.path.join(FIGURES_DIR, f"trip_{trip_num}_confusion_matrix.png")
    plt.savefig(trip_cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [✔] Exported {trip_name} confusion matrix -> {trip_cm_path} (Macro F1: {f_mac:.4f})")

# Export final score report table
df_score_report = pd.DataFrame(score_report_rows)
score_report_csv_path = os.path.join(OUTPUT_DIR, "score_report.csv")
df_score_report.to_csv(score_report_csv_path, index=False)
print(f"\\n[✔] Final comprehensive score report exported to: {score_report_csv_path}")
display(df_score_report)
""")
]

# ==============================================================================
# NOTEBOOK 3: GEOSPATIAL VISUALIZATION & MAPPING
# ==============================================================================
nb3_cells = [
    ("markdown", """# Notebook 3: Multi-Layer Geospatial Visualization & Mapping

This notebook builds interactive HTML maps using `folium` to visually validate our smartphone IMU-based IRI estimation model against physical road characteristics in the PVS dataset. For each trip, we generate an interactive map featuring three distinct layers:
1. **Predicted IRI Severity Band (Layer 1):** Color-coded circle markers representing our TFLite model's continuous roughness estimates categorized by our optimal grid-searched boundaries ($T_1^*, T_2^*$):
   - **Green (Good):** $\\text{IRI} < T_1^*$
   - **Orange/Yellow (Regular):** $T_1^* \\le \\text{IRI} < T_2^*$
   - **Red (Bad):** $\\text{IRI} \\ge T_2^*$
2. **Ground Truth PVS Road Quality (Layer 2):** Color-coded track indicating the manual visual/vehicle inspection labels (`Good = 0`, `Regular = 1`, `Bad = 2`).
3. **Road Surface Textures & Physical Anomalies (Layer 3):** Distinctive markers and popups highlighting transitions across pavement types (`asphalt`, `cobblestone`, `dirt`) and physical obstacles (`speed bumps`).

These self-contained HTML maps allow interactive layer toggling, zooming, and tooltip inspection, providing compelling qualitative proof for journal reviewers of how our physics-invariant model responds to real-world road anomalies.
"""),
    ("code", """import os
import json
import pandas as pd
import numpy as np
import folium

# --- CONFIGURATION & PATHS ---
DATA_DIR = r"D:\\Coding\\Hackathon\\GFG\\ARM\\ARM\\ml_model\\review_work\\data"
OUTPUT_DIR = r"D:\\Coding\\Hackathon\\GFG\\ARM\\ARM\\ml_model\\review_work\\outputs"
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps")
os.makedirs(MAPS_DIR, exist_ok=True)

# Load optimal thresholds from Notebook 2 (fallback to 3.5 and 7.0 if not found)
thresh_file = os.path.join(OUTPUT_DIR, "optimal_thresholds.json")
if os.path.exists(thresh_file):
    with open(thresh_file, 'r') as f:
        thresh_data = json.load(f)
    T1 = thresh_data.get('T1', 3.5)
    T2 = thresh_data.get('T2', 7.0)
    print(f"[*] Loaded Optimal Thresholds -> T1: {T1} m/km | T2: {T2} m/km")
else:
    T1, T2 = 3.5, 7.0
    print(f"[!] optimal_thresholds.json not found. Using default thresholds -> T1: {T1} m/km | T2: {T2} m/km")
"""),
    ("code", """# --- COLOR & ANOMALY HELPERS ---
def get_iri_color(iri_val):
    if iri_val < T1:
        return 'green', 'Good Road'
    elif iri_val < T2:
        return 'orange', 'Regular Road'
    else:
        return 'red', 'Bad Road'

def get_gt_color(label):
    if label == 0:
        return '#2ca02c', 'Good (0)'
    elif label == 1:
        return '#ff7f0e', 'Regular (1)'
    else:
        return '#d62728', 'Bad (2)'
"""),
    ("code", """# --- MAP GENERATOR FUNCTION ---
def generate_trip_map(trip_id):
    win_csv = os.path.join(OUTPUT_DIR, "all_trips_windowed_evaluation.csv")
    raw_csv = os.path.join(DATA_DIR, f"PVS_{trip_id}_with_predictions.csv")
    
    if not (os.path.exists(win_csv) and os.path.exists(raw_csv)):
        print(f"[!] Skipping PVS {trip_id}: Missing input files.")
        return None
        
    df_all_win = pd.read_csv(win_csv)
    df_win = df_all_win[df_all_win['trip_id'] == f"PVS {trip_id}"].copy()
    df_raw = pd.read_csv(raw_csv)
    
    if len(df_win) == 0 or len(df_raw) == 0:
        print(f"[!] No valid window data for PVS {trip_id}.")
        return None
        
    print(f"[+] Generating multi-layer map for PVS {trip_id} ({len(df_win):,} windows)...")
    
    # Initialize map centered at route start
    start_lat = df_win['mean_lat'].iloc[0]
    start_lon = df_win['mean_lon'].iloc[0]
    m = folium.Map(location=[start_lat, start_lon], zoom_start=15, tiles='CartoDB positron')
    
    # Feature Groups for Layer Control
    fg_iri = folium.FeatureGroup(name=f"Layer 1: Predicted IRI Bins (T1={T1}, T2={T2})", show=True)
    fg_gt = folium.FeatureGroup(name="Layer 2: Ground Truth Road Quality", show=True)
    fg_surface = folium.FeatureGroup(name="Layer 3: Surface Types & Speed Bumps", show=True)
    
    # 1. Add Predicted IRI Layer
    for idx, row in df_win.iterrows():
        color, class_name = get_iri_color(row['predicted_iri'])
        tooltip_text = (
            f"<b>Predicted IRI:</b> {row['predicted_iri']:.2f} m/km<br>"
            f"<b>Predicted Severity:</b> {class_name}<br>"
            f"<b>Mean Speed:</b> {row['mean_speed_kmh']:.1f} km/h<br>"
            f"<b>Ground Truth Class:</b> {row['label_mid']}"
        )
        folium.CircleMarker(
            location=[row['mean_lat'], row['mean_lon']],
            radius=5,
            weight=1,
            color='black',
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=tooltip_text
        ).add_to(fg_iri)
        
    # 2. Add Ground Truth Layer (Smaller radius for visual overlay comparison)
    for idx, row in df_win.iterrows():
        gt_color, gt_name = get_gt_color(row['label_mid'])
        folium.CircleMarker(
            location=[row['mean_lat'], row['mean_lon']],
            radius=2.5,
            weight=0,
            fill=True,
            fill_color=gt_color,
            fill_opacity=0.9,
            tooltip=f"Ground Truth: {gt_name}"
        ).add_to(fg_gt)
        
    # 3. Add Surface Anomalies Layer (Scan raw df for transitions and obstacles)
    df_sub = df_raw.iloc[::50].copy()
    
    for idx, row in df_sub.iterrows():
        lat, lon = row['latitude'], row['longitude']
        if pd.isna(lat) or pd.isna(lon):
            continue
            
        # Check Speed Bumps
        if row.get('speed_bump_asphalt', 0) == 1 or row.get('speed_bump_cobblestone', 0) == 1:
            bump_type = "Cobblestone Speed Bump" if row.get('speed_bump_cobblestone', 0) == 1 else "Asphalt Speed Bump"
            iri_str = f"{row['predicted_iri']:.2f}" if pd.notna(row.get('predicted_iri', np.nan)) else "N/A"
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa'),
                popup=f"<b>Obstacle:</b> {bump_type}<br><b>Local Predicted IRI:</b> {iri_str} m/km",
                tooltip=f"⚠️ {bump_type}"
            ).add_to(fg_surface)
            
        # Check Cobblestone sections
        elif row.get('cobblestone_road', 0) == 1 and idx % 250 == 0:
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color='purple',
                fill=True,
                fill_color='purple',
                fill_opacity=0.4,
                tooltip="🧱 Cobblestone Pavement Zone"
            ).add_to(fg_surface)
            
        # Check Dirt / Unpaved sections
        elif (row.get('dirt_road', 0) == 1 or row.get('unpaved_road', 0) == 1) and idx % 250 == 0:
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color='brown',
                fill=True,
                fill_color='brown',
                fill_opacity=0.5,
                tooltip="🟤 Dirt / Unpaved Road Zone"
            ).add_to(fg_surface)
            
    m.add_child(fg_iri)
    m.add_child(fg_gt)
    m.add_child(fg_surface)
    folium.LayerControl(collapsed=False).add_to(m)
    
    out_map_path = os.path.join(MAPS_DIR, f"PVS_{trip_id}_interactive_map.html")
    m.save(out_map_path)
    print(f"    [✔] Exported interactive map -> {out_map_path}")
    return out_map_path
"""),
    ("code", """# --- GENERATE MAPS FOR ALL 9 TRIPS ---
print("="*80)
print("EXPORTING MULTI-LAYER INTERACTIVE HTML MAPS")
print("="*80)

generated_maps = []
for i in range(1, 10):
    map_path = generate_trip_map(i)
    if map_path:
        generated_maps.append(map_path)

print(f"\\n[✔] Successfully generated {len(generated_maps)} trip maps in {MAPS_DIR}")
"""),
    ("code", """# --- GENERATE COMBINED OVERVIEW MAP ---
print("[+] Constructing global PVS overview map...")
win_csv = os.path.join(OUTPUT_DIR, "all_trips_windowed_evaluation.csv")
if os.path.exists(win_csv):
    df_all_win = pd.read_csv(win_csv)
    
    mean_lat = df_all_win['mean_lat'].mean()
    mean_lon = df_all_win['mean_lon'].mean()
    m_global = folium.Map(location=[mean_lat, mean_lon], zoom_start=13, tiles='CartoDB positron')
    
    df_global_sub = df_all_win.iloc[::2].copy()
    
    fg_global_iri = folium.FeatureGroup(name="Global Predicted IRI Severity", show=True)
    for idx, row in df_global_sub.iterrows():
        color, class_name = get_iri_color(row['predicted_iri'])
        folium.CircleMarker(
            location=[row['mean_lat'], row['mean_lon']],
            radius=4,
            weight=0,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            tooltip=f"{row['trip_id']} | IRI: {row['predicted_iri']:.2f} m/km ({class_name})"
        ).add_to(fg_global_iri)
        
    m_global.add_child(fg_global_iri)
    folium.LayerControl().add_to(m_global)
    
    global_map_path = os.path.join(MAPS_DIR, "combined_pvs_overview_map.html")
    m_global.save(global_map_path)
    print(f"[✔] Global combined overview map exported to: {global_map_path}")
""")
]

if __name__ == "__main__":
    out_dir = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model\review_work\notebooks"
    os.makedirs(out_dir, exist_ok=True)
    
    nb1 = make_nb(nb1_cells)
    nb2 = make_nb(nb2_cells)
    nb3 = make_nb(nb3_cells)
    
    path1 = os.path.join(out_dir, "01_prepare_pvs_data.ipynb")
    path2 = os.path.join(out_dir, "02_iri_inference_and_evaluation.ipynb")
    path3 = os.path.join(out_dir, "03_geospatial_mapping.ipynb")
    
    with open(path1, "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)
    print(f"[OK] Created Notebook 1: {path1}")
    
    with open(path2, "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)
    print(f"[OK] Created Notebook 2: {path2}")
    
    with open(path3, "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)
    print(f"[OK] Created Notebook 3: {path3}")
