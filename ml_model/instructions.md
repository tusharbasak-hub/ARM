# IRI Background Model - Implementation Guide

This guide details how to implement the `iri_background_model.tflite` model in a mobile application or edge device. The model predicts the International Roughness Index (IRI) of a road by analyzing IMU and speed data.

## 1. Data Collection Requirements

To use this model, you need to collect data from three primary sensors: the **Accelerometer**, the **Gyroscope**, and a **Speed Source** (like GPS or OBD-II).

### Sensor Data & Units

| Feature      | Sensor        | Description                                 | Unit      |
| :----------- | :------------ | :------------------------------------------ | :-------- |
| `ax`       | Accelerometer | Lateral Acceleration (Left/Right)           | `m/s²` |
| `ay`       | Accelerometer | Longitudinal Acceleration (Forward/Braking) | `m/s²` |
| `az`       | Accelerometer | Vertical Acceleration (Up/Down)             | `m/s²` |
| `wx`       | Gyroscope     | Pitch (Rotation around Lateral axis)        | `rad/s` |
| `wy`       | Gyroscope     | Roll (Rotation around Longitudinal axis)    | `rad/s` |
| `wz`       | Gyroscope     | Yaw (Rotation around Vertical axis)         | `rad/s` |
| `speed_ms` | GPS / OBD2    | Current Vehicle Speed                       | `m/s`   |

### Coordinate System (Crucial!)

The model was trained on a standardized **smartphone orientation**. If your device is mounted differently, you **must** remap the axes to match:

- **X-axis (Lateral):** Points out the left/right sides of the vehicle.
- **Y-axis (Longitudinal):** Points forward out the windshield / backward out the rear.
- **Z-axis (Vertical):** Points straight up to the sky / down to the road.

### Sampling Frequency

- **Suggested Frequency:** 100 Hz.
- model is trained on frequencies ranging from 10 - 30 Hz, higher the better
- Since the model operates in the **spatial domain** (distance) rather than the **time domain**, minor fluctuations in polling rates are acceptable, provided you accurately timestamp the data to calculate distance.

---

## 2. Data Preprocessing Pipeline

The model does **not** take a stream of raw time-series data. It expects a **spatially windowed** sequence.

### Step A: Time-to-Distance Mapping

For every sample received, calculate the distance traveled since the last sample using the speed and the time delta:

```python
dt = current_timestamp - previous_timestamp
dx = speed_ms * dt
cumulative_distance += dx
```

### Step B: Spatial Windowing (5.0 Meters)

Extract a window of exactly **5.0 meters** of traveled distance. Wait until your cumulative distance hits 5.0m before running an inference.

### Step C: Spatial Resampling (100 Steps)

Once you have 5.0m of data, you must resample it into exactly **100 spatial points**. This essentially means evaluating the sensor values every 5 centimeters. Use linear interpolation to convert your irregular time-based arrays into a fixed `(100, 7)` matrix.
The final order of features in the matrix must be: `[ax, ay, az, wx, wy, wz, speed_ms]`.

### Step D: Wavelet Filtering (⚠️ Important Quirk)

*Note for Developers:* The original training code intended to use a Daubechies 4 (`db4`) Wavelet Low-Pass Filter on the 6 IMU channels. However, **due to a bug in the training pipeline, the filter was bypassed and the model was trained on raw, unfiltered (but interpolated) data.**

- **Action:** For this specific `.tflite` model, **do not** apply the wavelet filter. Pass the raw interpolated `(100, 7)` data directly to match how the model was trained.

### Step E: Contextual Features Extraction

The model uses a secondary input array of shape `(7,)` containing aggregated statistics for the 5.0m window. Calculate these from your `(100, 7)` window matrix:

1. `avg_speed_ms`: The mean of the `speed_ms` column.
2. `mean_az`: The mean of the `az` column.
3. `mean_ay`: The mean of the `ay` column.
4. `mcr_az`: The Mean Crossing Rate of `az`. (Count how many times the sign of `az` changes in the 100 steps, then divide by 100).
5. `veh_type_1`: One-hot encoding for vehicle type. (Set to `0` for generic use).
6. `veh_type_2`: One-hot encoding for vehicle type. (Set to `0` for generic use).
7. `veh_type_3`: One-hot encoding for vehicle type. (Set to `1` as a default fallback).

*Example Context Array:* `[avg_speed, mean_az, mean_ay, mcr_az, 0.0, 0.0, 1.0]`

---

## 3. Inference Implementation

You will need the `tflite-runtime` or standard `tensorflow` library (or the TFLite package for Android/iOS if building natively).

### Python Inference Example

```python
import numpy as np
import tensorflow as tf

# 1. Load the TFLite Model
interpreter = tf.lite.Interpreter(model_path="iri_background_model.tflite")
interpreter.allocate_tensors()

# 2. Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# TFLite may reorder inputs. We must map our data to the correct tensor index based on name.
raw_input_index = None
ctx_input_index = None

for detail in input_details:
    if "raw_imu_speed" in detail['name']:
        raw_input_index = detail['index']
    elif "context_stats" in detail['name']:
        ctx_input_index = detail['index']

# 3. Prepare your data (Mock Data shown here)
# raw_window shape must be (1, 100, 7), dtype float32
raw_window = np.zeros((1, 100, 7), dtype=np.float32) 

# context_stats shape must be (1, 7), dtype float32
context_stats = np.array([[12.5, 9.8, 0.1, 0.45, 0.0, 0.0, 1.0]], dtype=np.float32)

# 4. Set Tensors
interpreter.set_tensor(raw_input_index, raw_window)
interpreter.set_tensor(ctx_input_index, context_stats)

# 5. Invoke the Model
interpreter.invoke()

# 6. Retrieve the Prediction
predicted_iri = interpreter.get_tensor(output_details[0]['index'])

print(f"Predicted IRI (m/km): {predicted_iri[0][0]:.2f}")
```

### Output Interpretation

The model outputs a single continuous `float32` value representing the International Roughness Index (IRI) in **meters per kilometer (m/km)**. Higher values indicate rougher roads.
