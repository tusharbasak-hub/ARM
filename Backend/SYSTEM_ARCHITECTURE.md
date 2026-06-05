# Detailed System Architecture

This document provides a comprehensive overview of the Road Quality Monitoring System's architecture, specifically detailing the Data Collection pipeline, International Roughness Index (IRI) calculation methodology, Machine Learning model design, and Backend infrastructure.

## 1. Data Collection Architecture (BeamNG.tech Simulation)
**Source:** `ml_model/data/simulation/scripts/IRI/collect.ipynb`

The data collection subsystem is designed to simulate realistic vehicle dynamics and gather high-fidelity sensor readings using the BeamNG.tech physics simulator.

### Architecture Highlights
* **Multithreaded Polling System:** Utilizes a highly synchronized multithreaded architecture with thread locks (`bng_lock`, `data_lock`) to poll data simultaneously without blocking the simulator.
* **Sensor Suite:**
  * **Advanced IMU (100Hz):** Captures raw and smoothed acceleration (ax, ay, az) and angular velocity (wx, wy, wz).
  * **Dual Lidar (100Hz):** Mounted left and right pointing downwards to measure exact road elevation (Z-axis absolute values).
  * **Dashcam (20Hz):** Captures visual frames of the road synchronized with the physics data.
* **Data Flow:**
  1. Thread workers continuously poll IMU, Lidar, and Camera data.
  2. Data is transformed (e.g., IMU axes re-routed to match smartphone orientation, Z-elevation safely extracted from the point cloud).
  3. A main strict metronome loop (running exactly at 100Hz) synchronizes all sensor states and dumps them into `imu_speed_data.csv` and `iri_sensor_data.csv`.
  4. An asynchronous Image Saver Queue processes dashcam frames off the main thread to prevent IO blocking.

---

## 2. Target Variable (IRI) Generation Pipeline
**Source:** `ml_model/data/simulation/scripts/IRI/iri_simple.py`

This pipeline is responsible for converting raw BeamNG telemetry into standardized International Roughness Index (IRI) labels using World Bank/IRC parameters.

### Processing Engine
* **Golden Car Quarter-Car Model:** Simulates standard suspension using a State-Space representation.
  * Parameters: C=6.0, K1=63.3, K2=653.0, MU=0.15.
* **Wavelet Processing (Dual Domain):**
  1. **Time Domain LPF (1.11 Hz):** Applies a Discrete Wavelet Transform (`db4` wavelet) to filter out high-frequency sensor noise.
  2. **Spatial Domain BPF (5.4m - 25m):** After converting time-series data to a uniform 1cm spatial grid (using cumulative distance), it applies a Band-Pass Wavelet filter to isolate the specific wavelengths (bumps/dips) that affect human ride comfort, ignoring hills and micro-fuzz.
* **IRI Calculation:** Simulates the "Golden Car" over each 1-meter segment of the spatially-filtered profile and computes the accumulated suspension movement per kilometer.
* **Output:** Merges the original IMU data with the calculated IRI for each spatial patch into a final `readings.csv`.

---

## 3. Machine Learning Model Architecture
**Source:** `ml_model/src/model/IRI/model_final copy.ipynb`

The ML model is designed to predict the IRI directly from noisy IMU and GPS speed data without requiring expensive Lidar setups.

### Preprocessing & Feature Engineering
* **Spatial Windowing:** Transforms time-domain data into the spatial domain using a sliding 5-meter window with a 2-meter stride. Resampled via 1D linear interpolation to exactly 100 uniformly spaced data points per window.
* **Wavelet Filtering:** Applies Level-2 'db4' wavelet filtering on the spatial signal to isolate low-frequency dynamics.
* **Class Balancing (The "W" Strategy):** Solves heavy dataset imbalance by randomly dropping 70% of "smooth road" data (IRI < 2) and applying a 10x oversampling multiplier to edge-case anomalies (high speed + severe degradation).
* **Parallel Extraction:** Utilizes `joblib` for parallel processing, with deterministic routing to train/val/test splits to prevent data leakage.

### Dual-Input Network Design
The network processes two distinct data types before merging:
1. **Spatial Feature Extractor:** Takes raw (100, 7) arrays. Uses `DepthwiseConv1D` to analyze individual sensor channels before mixing, followed by standard `Conv1D`, `MaxPooling1D`, and `GlobalMaxPooling1D` to extract structural bump patterns.
2. **Contextual Network:** Takes a (7,) array of trip stats (average speed, one-hot vehicle type). Uses Dense layers to provide a baseline understanding of the driving scenario.
3. **Feature Fusion:** Concatenates both branches and passes through final Dense layers to output the predicted IRI.

### Context-Aware Custom Loss Function (`boss_level_iri_loss_with_hunter`)
Designed to handle specific physical edge cases:
* **Stable Baseline:** Huber loss combined with Log-Cosh for stable gradients.
* **Pothole Penalty:** Multiplies error by 3.0 if the actual road is severely degraded (IRI > 4.0).
* **The "Hunter" (False Alarm Penalty):** Multiplies loss by 6.0 if the model hallucinates a high IRI (predicts > 2.0) on a smooth road while the car is moving slowly (< 20 m/s) or accelerating hard. This requires "smuggling" speed and acceleration into the loss function via a stacked `y_true_combo` array.

### Deployment
* The final trained Keras model undergoes Post-Training Quantization to convert float32 weights into a lighter format, saving as `iri_background_model.tflite` for edge inference on mobile devices with near-zero accuracy loss.

---

## 4. Backend Infrastructure
**Source:** `Backend/docs/ARCHITECTURE.md`

The backend provides scalable, real-time crowdsourcing capabilities.

### Architecture Overview Flow
1. **Mobile App (TinyML):** On-device calibration and ML inference outputs Road Quality (0-3), GPS, and Speed.
2. **REST API (Node.js/Express):** Handles Auth (JWT) and receives observation data.
3. **Core Processing Layer:**
   - **Map Matching Service (OSRM):** Snaps GPS points to exact road segments.
   - **Geohash Service:** Calculates precision 6 Geohashes (1.2km × 0.61km areas) for spatial segmentation. Neighboring regions are also fetched for a smooth map experience.
   - **Aggregation Service:** Uses time decay and weighted averages to update road quality scores based on recent observations.
4. **Socket.IO Real-time Server:** Users are grouped into regional rooms based on their Geohash. Broadcasts road quality updates *only* to the relevant regional room.
5. **Data Storage Layer:**
   - **MongoDB (Persistent Domain Data):** Stores `users` (with device IDs for anonymous auth), `observations` (GeoJSON locations, TTL indexed), and `roadSegments`.
   - **Redis (Ephemeral Session State):** Manages active sessions, region members, and heartbeats with auto-expiring TTLs.

### Scalability Architecture
* **Horizontal Scaling:** API designed statelessly to support multiple servers balanced via sticky sessions, using a Redis Adapter to synchronize Socket.IO events across nodes.
* **Database Sharding:** MongoDB is sharded using the `regionId` (Geohash) as the shard key, ensuring localized data resides on the same shard (e.g., all regions starting with "tt" in one shard).
