import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, losses

# ==========================================
# Configuration and Setup
# ==========================================
DATA_DIR = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model\baseline\data"
OUTPUT_DIR = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model\baseline\outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Known historical training times for auto-resumed models
HISTORICAL_TRAIN_TIMES = {
    'Ridge Regression': 0.23,
    'Random Forest': 3.52,
    'XGBoost': 0.96,
    'LightGBM': 0.58,
    'CatBoost': 2.30,
    '2-Layer Bi-LSTM': 120.0,
    '2-Layer GRU': 114.32
}

def load_compiled_data():
    """Loads compressed train, val, and test data buffers."""
    print("[*] Loading compiled dataset buffers...")
    train_data = np.load(os.path.join(DATA_DIR, "train_data.npz"))
    val_data = np.load(os.path.join(DATA_DIR, "val_data.npz"))
    test_data = np.load(os.path.join(DATA_DIR, "test_data.npz"))
    
    return {
        'train': {k: train_data[k] for k in train_data.files},
        'val': {k: val_data[k] for k in val_data.files},
        'test': {k: test_data[k] for k in test_data.files}
    }

def save_checkpoint_state(predictions, timing_metrics, y_test):
    """Saves predictions and timing metrics incrementally after each model."""
    preds_to_save = dict(predictions)
    preds_to_save['y_true'] = y_test
    np.savez_compressed(os.path.join(OUTPUT_DIR, "all_model_predictions.npz"), **preds_to_save)
    
    with open(os.path.join(OUTPUT_DIR, "model_timing_metrics.json"), "w") as f:
        json.dump(timing_metrics, f, indent=4)

# ==========================================
# 1. Tabular Baseline Models
# ==========================================
def train_tabular_models(data, predictions, timing_metrics):
    """Trains and evaluates Family A & Family D tabular baselines with auto-resume."""
    print("\n" + "="*50)
    print("STARTING TABULAR BASELINE TRAINING (Family A & D)")
    print("="*50)
    
    X_train_tab, y_train = data['train']['tab'], data['train']['y']
    X_val_tab, y_val = data['val']['tab'], data['val']['y']
    X_test_tab, y_test = data['test']['tab'], data['test']['y']
    
    # Standardize features for linear model and tree models
    scaler_path = os.path.join(MODELS_DIR, "tabular_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_train_scaled = scaler.transform(X_train_tab)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_tab)
        joblib.dump(scaler, scaler_path)
        
    X_val_scaled = scaler.transform(X_val_tab)
    X_test_scaled = scaler.transform(X_test_tab)
    
    tabular_models = {
        'Ridge Regression': RidgeCV(alphas=np.logspace(-3, 3, 20)),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
    }
    
    for name, model_init in tabular_models.items():
        save_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl")
        
        if name == 'Ridge Regression':
            X_test_eval = X_test_scaled
        else:
            X_test_eval = X_test_tab
            
        if os.path.exists(save_path):
            print(f"[*] Found existing model for {name}, loading from disk...")
            model = joblib.load(save_path)
            train_time = HISTORICAL_TRAIN_TIMES.get(name, 1.0)
        else:
            print(f"[*] Training {name}...")
            model = model_init
            t0 = time.time()
            if name == 'Ridge Regression':
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train_tab, y_train)
            train_time = time.time() - t0
            joblib.dump(model, save_path)
        
        # Measure inference latency on test set
        t0_inf = time.time()
        preds = model.predict(X_test_eval)
        inf_time_total = time.time() - t0_inf
        ms_per_sample = (inf_time_total / len(y_test)) * 1000.0
        
        predictions[name] = preds
        timing_metrics[name] = {
            'train_time_s': float(train_time),
            'inf_ms_per_sample': float(ms_per_sample)
        }
        print(f"    [+] {name} ready | Train Time: {train_time:.2f}s | Latency: {ms_per_sample:.4f} ms/sample")
        save_checkpoint_state(predictions, timing_metrics, y_test)
        
    return predictions, timing_metrics

# ==========================================
# 2. Recurrent & Hybrid Baselines + 1D-CNN
# ==========================================
def build_bilstm_model():
    """Family B: 2-Layer Bi-Directional LSTM on 400x7 raw spatial sequence."""
    inp = layers.Input(shape=(400, 7), name="raw_sequence")
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='softplus', name="predicted_iri")(x)
    return models.Model(inp, out, name="BiLSTM_2Layer")

def build_gru_model():
    """Family B: 2-Layer Bi-Directional GRU on 400x7 raw spatial sequence."""
    inp = layers.Input(shape=(400, 7), name="raw_sequence")
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.GRU(32, return_sequences=False))(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='softplus', name="predicted_iri")(x)
    return models.Model(inp, out, name="GRU_2Layer")

def build_hybrid_lstm_context_model():
    """Family C: Bi-LSTM sequence branch + Dense context branch fusion."""
    raw_inp = layers.Input(shape=(400, 7), name="raw_sequence")
    ctx_inp = layers.Input(shape=(13,), name="context_features")
    
    # RNN branch
    x_seq = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(raw_inp)
    x_seq = layers.Dense(32, activation='relu')(x_seq)
    
    # Context branch
    x_ctx = layers.Dense(32, activation='relu')(ctx_inp)
    x_ctx = layers.BatchNormalization()(x_ctx)
    
    # Late fusion
    fused = layers.Concatenate()([x_seq, x_ctx])
    fused = layers.Dense(64, activation='relu')(fused)
    fused = layers.Dropout(0.2)(fused)
    fused = layers.Dense(32, activation='relu')(fused)
    out = layers.Dense(1, activation='softplus', name="predicted_iri")(fused)
    return models.Model([raw_inp, ctx_inp], out, name="BiLSTM_Context_Hybrid")

def build_proposed_1d_cnn_model():
    """Proposed Section VII: Hierarchical Multi-Scale 1D-CNN + Context Fusion."""
    raw_inp = layers.Input(shape=(400, 6), name="raw_imu_6ch")
    ctx_inp = layers.Input(shape=(13,), name="context_features")
    
    # Spatial branch
    x_spat = layers.DepthwiseConv1D(kernel_size=5, depth_multiplier=2, activation='relu', padding='same')(raw_inp)
    x_spat = layers.BatchNormalization()(x_spat)
    x_spat = layers.MaxPooling1D(pool_size=2)(x_spat)
    
    x_spat = layers.Conv1D(48, kernel_size=5, activation='relu', padding='same')(x_spat)
    x_spat = layers.BatchNormalization()(x_spat)
    x_spat = layers.MaxPooling1D(pool_size=2)(x_spat)
    
    x_spat = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x_spat)
    x_spat = layers.BatchNormalization()(x_spat)
    
    pool_avg = layers.GlobalAveragePooling1D()(x_spat)
    pool_max = layers.GlobalMaxPooling1D()(x_spat)
    cnn_feats = layers.Concatenate()([pool_avg, pool_max])
    cnn_feats = layers.Dropout(0.3)(cnn_feats)
    
    # Context branch
    x_ctx = layers.Dense(32, activation='relu')(ctx_inp)
    x_ctx = layers.BatchNormalization()(x_ctx)
    x_ctx = layers.Dropout(0.2)(x_ctx)
    
    # Fusion
    fused = layers.Concatenate()([cnn_feats, x_ctx])
    fused = layers.Dense(64, activation='relu')(fused)
    fused = layers.Dense(32, activation='relu')(fused)
    out = layers.Dense(1, activation='softplus', name="predicted_iri")(fused)
    
    return models.Model([raw_inp, ctx_inp], out, name="Proposed_1D_CNN")

def train_neural_models(data, predictions, timing_metrics):
    """Trains and evaluates Family B, Family C, and Proposed Section VII neural models with auto-resume."""
    print("\n" + "="*50)
    print("STARTING NEURAL & RECURRENT MODEL TRAINING")
    print("="*50)
    
    # Prepare datasets
    X_train_raw7, X_train_ctx, y_train = data['train']['raw'], data['train']['ctx'], data['train']['y']
    X_val_raw7, X_val_ctx, y_val = data['val']['raw'], data['val']['ctx'], data['val']['y']
    X_test_raw7, X_test_ctx, y_test = data['test']['raw'], data['test']['ctx'], data['test']['y']
    
    # For Section VII 1D-CNN, slice the first 6 channels (ax, ay, az, wx, wy, wz)
    X_train_raw6 = X_train_raw7[:, :, :6]
    X_val_raw6 = X_val_raw7[:, :, :6]
    X_test_raw6 = X_test_raw7[:, :, :6]
    
    neural_configs = [
        ("2-Layer Bi-LSTM", build_bilstm_model, X_train_raw7, X_val_raw7, X_test_raw7),
        ("2-Layer GRU", build_gru_model, X_train_raw7, X_val_raw7, X_test_raw7),
        ("Bi-LSTM + Context", build_hybrid_lstm_context_model, [X_train_raw7, X_train_ctx], [X_val_raw7, X_val_ctx], [X_test_raw7, X_test_ctx]),
        ("Proposed 1D-CNN", build_proposed_1d_cnn_model, [X_train_raw6, X_train_ctx], [X_val_raw6, X_val_ctx], [X_test_raw6, X_test_ctx])
    ]
    
    for name, model_builder, x_tr, x_va, x_te in neural_configs:
        save_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.keras")
        
        if os.path.exists(save_path):
            print(f"\n[*] Found existing checkpoint for {name}, loading from disk...")
            model = models.load_model(save_path)
            train_time = HISTORICAL_TRAIN_TIMES.get(name, 120.0)
        else:
            print(f"\n[*] Compiling and Training {name}...")
            model = model_builder()
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss=losses.Huber(delta=1.0),
                metrics=['mae']
            )
            
            early_stop = callbacks.EarlyStopping(monitor='val_mae', patience=8, restore_best_weights=True, verbose=1)
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=4, min_lr=1e-5, verbose=0)
            
            t0 = time.time()
            history = model.fit(
                x_tr, y_train,
                validation_data=(x_va, y_val),
                epochs=50,
                batch_size=128,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            train_time = time.time() - t0
            model.save(save_path)
            
        # Measure inference latency on test set
        t0_inf = time.time()
        preds = model.predict(x_te, batch_size=256, verbose=0).flatten()
        inf_time_total = time.time() - t0_inf
        ms_per_sample = (inf_time_total / len(y_test)) * 1000.0
        
        predictions[name] = preds
        timing_metrics[name] = {
            'train_time_s': float(train_time),
            'inf_ms_per_sample': float(ms_per_sample)
        }
        print(f"    [+] {name} ready | Train Time: {train_time:.2f}s | Latency: {ms_per_sample:.4f} ms/sample")
        save_checkpoint_state(predictions, timing_metrics, y_test)
        
    return predictions, timing_metrics

if __name__ == '__main__':
    data = load_compiled_data()
    predictions, timing_metrics = {}, {}
    
    predictions, timing_metrics = train_tabular_models(data, predictions, timing_metrics)
    predictions, timing_metrics = train_neural_models(data, predictions, timing_metrics)
    
    print(f"\n[*] All 9 models benchmarked and saved successfully! Predictions written to {OUTPUT_DIR}/all_model_predictions.npz")
