import os
import numpy as np
import pandas as pd
import pywt
from scipy.signal import StateSpace, lsim, detrend
from scipy.interpolate import interp1d

# ==========================================
# 1. CONSTANTS & IRC/WORLD BANK PARAMETERS
# ==========================================
DX = 0.01                # 1cm uniform spatial grid
SEGMENT_LENGTH_M = 1.0   # 1-meter patch for MRI calculation
V_SIM_KMH = 80.0         # Standard IRC testing speed
V_SIM_MS = V_SIM_KMH * (1000.0 / 3600.0) # 22.22 m/s

# Golden Car Parameters (Standardized by World Bank/IRC)
C  = 6.0    
K1 = 63.3   
K2 = 653.0  
MU = 0.15   

A = np.array([
    [0, 1, 0, 0],
    [-K1, -C, K1, C],
    [0, 0, 0, 1],
    [K1/MU, C/MU, -(K1+K2)/MU, -C/MU]
])
B = np.array([[0], [0], [0], [K2/MU]])
C_mat = np.array([[0, 1, 0, -1]]) 
D = np.array([[0]])

golden_car_system = StateSpace(A, B, C_mat, D)

# ==========================================
# 2. WAVELET PROCESSING ENGINE
# ==========================================

def wavelet_time_lpf(signal, time_array, cutoff_hz=1.11, wavelet='db4'):
    """ Acts as a Low-Pass Filter in the Time Domain. """
    dt = np.median(np.diff(time_array))
    fs = 1.0 / dt if dt > 0 else 100.0
    
    # Calculate Level: L = log2(Fs / F_cutoff) - 1
    target_level = int(np.round(np.log2(fs / cutoff_hz))) - 1
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    target_level = np.clip(target_level, 1, max_level)
    
    coeffs = pywt.wavedec(signal, wavelet, level=target_level)
    # Keep only Approximation (cA), zero out Detail coefficients (cD)
    new_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(new_coeffs, wavelet)[:len(signal)]

def wavelet_spatial_bpf(data, dx=0.01, min_wave=5.4, max_wave=25.0, wavelet='db4'):
    """
    Applies a Band-Pass filter in the Spatial Domain using Wavelets.
    Nukes anything shorter than 5.4m and longer than 25m.
    """
    # Each level L corresponds to a scale of approx 2^L * DX
    # L_min (high freq limit) ~ log2(5.4 / 0.01) -> ~9.07
    # L_max (low freq limit) ~ log2(25 / 0.01) -> ~11.28
    lvl_min = int(np.floor(np.log2(min_wave / (2 * dx)))) 
    lvl_max = int(np.ceil(np.log2(max_wave / (2 * dx))))
    
    # We need to decompose deep enough to see the max_wave
    decomp_level = max(lvl_max + 1, 12)
    max_possible = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len)
    decomp_level = min(decomp_level, max_possible)
    
    coeffs = pywt.wavedec(data, wavelet, level=decomp_level)
    new_coeffs = []
    
    # coeffs[0] is Approximation (The very long hills > 25m) -> Nuke it
    new_coeffs.append(np.zeros_like(coeffs[0]))
    
    for i in range(1, len(coeffs)):
        # Detail level index is relative to the total decomposition
        current_lvl = decomp_level - i + 1
        
        # Keep only the details that fall within our wavelength band
        if lvl_min <= current_lvl <= lvl_max:
            new_coeffs.append(coeffs[i])
        else:
            new_coeffs.append(np.zeros_like(coeffs[i]))
            
    return pywt.waverec(new_coeffs, wavelet)[:len(data)]

def calculate_iri_patch(spatial_profile, dx):
    if len(spatial_profile) < 2: return 0.0
    dt = dx / V_SIM_MS
    time_array = np.arange(len(spatial_profile)) * dt
    zeroed_profile = spatial_profile - spatial_profile[0]
    _, yout, _ = lsim(golden_car_system, U=zeroed_profile, T=time_array)
    return np.sum(np.abs(yout)) * dt

# ==========================================
# 3. PROCESSING PIPELINE
# ==========================================
def process_trip(trip_dir):
    imu_path = os.path.join(trip_dir, 'imu_speed_data.csv')
    iri_path = os.path.join(trip_dir, 'iri_sensor_data.csv')
    
    if not os.path.exists(imu_path) or not os.path.exists(iri_path): return False

    df = pd.merge(pd.read_csv(imu_path), pd.read_csv(iri_path), on=['sample_number', 'sim_time'])
    valid_mask = (df['pos_x'] != 0.0) & (df['road_elevation_left'] != 0.0)
    df = df[valid_mask].copy().reset_index(drop=True)
    if len(df) < 50: return False

    # --- STEP 1: TIME DOMAIN WAVELET LPF (1.11 Hz) ---
    t_array = df['sim_time'].values
    for side in ['left', 'right']:
        col = f'road_elevation_{side}'
        df[f'{col}_clean_time'] = wavelet_time_lpf(df[col].values, t_array, cutoff_hz=1.11)

    # --- STEP 2: SPATIAL TRANSFORMATION ---
    dx_arr = np.diff(df['pos_x'].values, prepend=df['pos_x'].values[0])
    dy_arr = np.diff(df['pos_y'].values, prepend=df['pos_y'].values[0])
    dz_arr = np.diff(df['pos_z'].values, prepend=df['pos_z'].values[0]) 
    df['cumulative_distance'] = np.cumsum(np.sqrt(dx_arr**2 + dy_arr**2 + dz_arr**2))

    max_dist = df['cumulative_distance'].max()
    spatial_grid = np.arange(0, max_dist, DX)
    change_mask = df['cumulative_distance'].diff() != 0
    change_mask.iloc[0] = True
    df_true = df[change_mask].copy()

    # --- STEP 3: SPATIAL DOMAIN WAVELET BPF (5.4m - 25m) ---
    z_final = {}
    for side in ['left', 'right']:
        # Interpolate cleaned time-data to spatial grid
        z_spatial_raw = interp1d(df_true['cumulative_distance'], 
                                 df_true[f'road_elevation_{side}_clean_time'], 
                                 kind='cubic', fill_value="extrapolate")(spatial_grid)
        
        # Apply the Band-Pass (This nukes hills AND the remaining micro-fuzz)
        z_final[side] = wavelet_spatial_bpf(detrend(z_spatial_raw), dx=DX, 
                                           min_wave=5.4, max_wave=25.0)

    # --- STEP 4: IRI CALCULATION ---
    samples_per_segment = int(SEGMENT_LENGTH_M / DX)
    segment_records = []
    
    for start_idx in range(0, len(spatial_grid), samples_per_segment):
        end_idx = start_idx + samples_per_segment
        if end_idx > len(spatial_grid): break 
            
        dist_km = (samples_per_segment * DX) / 1000.0
        iri_l = calculate_iri_patch(z_final['left'][start_idx:end_idx], DX) / dist_km
        iri_r = calculate_iri_patch(z_final['right'][start_idx:end_idx], DX) / dist_km
        
        segment_records.append({
            'start_dist': spatial_grid[start_idx],
            'end_dist': spatial_grid[end_idx - 1],
            'IRI': (iri_l + iri_r) / 2.0
        })
        
    segments_df = pd.DataFrame(segment_records)
    if segments_df.empty: return False

    # Map back and Save
    bin_edges = np.array(list(segments_df['start_dist']) + [segments_df['end_dist'].iloc[-1] + 0.001])
    df['IRI'] = segments_df['IRI'].values[np.clip(np.searchsorted(bin_edges, df['cumulative_distance'].values, side='right') - 1, 0, len(segments_df)-1)]

    output_path = os.path.join(trip_dir, 'readings.csv')
    df[['sample_number', 'ax', 'ay', 'az', 'wx', 'wy', 'wz', 'speed_ms', 'IRI']].to_csv(output_path, index=False)
    print(f"    [+] Saved readings.csv | Max 1m IRI: {segments_df['IRI'].max():.1f}")
    return True

def main():
    # Fix this path to point to your actual data folder
    base_dir = os.path.abspath('../../data/IRI') 
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    for car in os.listdir(base_dir):
        car_p = os.path.join(base_dir, car)
        if not os.path.isdir(car_p): continue
        for m_name in os.listdir(car_p):
            m_p = os.path.join(car_p, m_name)
            if not os.path.isdir(m_p): continue
            for trip in os.listdir(m_p):
                trip_dir = os.path.join(m_p, trip)
                if os.path.isdir(trip_dir) and trip.startswith('trip_'):
                    print(f"Processing: {trip_dir}")
                    process_trip(trip_dir)

if __name__ == '__main__':
    main()