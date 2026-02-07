# Technical Documentation: Synthetic Pothole Data Pipeline

## 1. Overview

This module is responsible for ingesting raw accelerometer/gyroscope data, normalizing it, and generating a **synthetic training dataset** that simulates realistic road transitions. It outputs a time-series dataset formatted for a Hybrid CNN-MLP model.

**Key Objectives:**

* **Normalization:** Unify sampling rates (100Hz **$\to$** 10Hz) and units.
* **Scenario Synthesis:** Create "continuous driving" scenarios by stitching disparate sensor clips.
* **Leakage Prevention:** Enforce a strict file-level and geographic-level split between Train and Test sets.

---

## 2. Data Ingestion & Preprocessing

### Input Formats

1. **Raw Class Data:** CSV files containing 100Hz sensor readings.
2. **Manual "Good Road" Data:** A specific 10Hz CSV (`Yashobhoomi.csv`) containing GPS-tagged smooth road data.

### Preprocessing Logic (`load_and_split_pools`)

* **Gravity Correction:** Adds **$9.80665 m/s^2$** to the **$a_z$** component of raw files to align with standard IMU frames.
* **Downsampling:** Decimates 100Hz data to 10Hz using array slicing (`::10`) to preserve peak signal characteristics without smoothing out high-frequency impacts.
* **Segment Extraction:** Applies a dictionary of `FILE_RULES` to slice specific time ranges from raw files, assigning ground-truth labels (0–4) to valid segments only.

---

## 3. Train / Test Splitting Strategy

To ensure zero data leakage and robust evaluation, the pipeline uses a  **Stratified Source Split** :

1. **Stratified File Splitting:**
   * Files are grouped by their *maximum severity label* (e.g., all files containing "Class 4" potholes).
   * One file from **each** severity group is randomly withheld for the  **Test Set** .
   * The remaining files form the  **Training Set** .
   * *Result:* The model is tested on physical potholes it has never seen before.
2. **Geographic Splitting (Good Road):**
   * The `Yashobhoomi` dataset is split based on GPS coordinates.
   * **Segment A (Train):** Lat/Lon range **$[28.408, 76.927] \to [28.405, 76.919]$**.
   * **Segment B (Test):** Lat/Lon range **$[28.517, 77.022] \to [28.463, 76.968]$**.
   * *Result:* "Good road" noise patterns in Test are geographically distinct from Train.

---

## 4. Synthetic Scenario Generation

The `synthesize_windows` function generates data by simulating a car driving over different road surfaces.

### Stitching Logic

* **Scenario:** Transitions are created by concatenating clips in a `Good -> Bad -> Good` (or random) sequence.
* **Smoothing (`smooth_transition`):** To prevent artificial "jerk" (infinite acceleration) at the cut points, the system applies a **linear blend** over a 5-sample overlap region.
  * *Formula:* **$X_{new} = (1-\alpha)X_{clipA} + \alpha X_{clipB}$**

### Speed Injection (`generate_synthetic_speed`)

Since raw speed data may not match the desired training scenario, speed is synthetically overwritten:

* **Profile:** Random Walk noise is added to a base target speed to simulate natural throttle variation.
* **Modes:**
  * `fast_bad`: Simulates hitting a pothole at high speed (10–20 m/s).
  * `slow_good`: Simulates traffic on smooth roads (2–8 m/s).
  * `normal`: Random variation (2–20 m/s).

---

## 5. Windowing & Output

### Sliding Window Configuration

* **Window Size:** 2.0 seconds (20 samples at 10Hz).
* **Stride:** 0.5 seconds (5 samples), providing 75% overlap for data augmentation.

### Labeling Strategy

* **Max-Severity Logic:** The label for a 2-second window is the **maximum** label value present in that window.
  * *Example:* If a window contains 1.5s of "Good Road (0)" and 0.5s of "Large Pothole (4)", the entire window is labeled  **4** .

### Output Schema (CSV)

The output files (`train_windows.csv`, `test_windows.csv`) follow a "Long Format":

| **Column** | **Type** | **Description**                                      |
| ---------------- | -------------- | ---------------------------------------------------------- |
| `window_id`    | Int            | Unique ID for the 2-second event. Groups 20 rows together. |
| `step`         | Int            | Time step index (0–19) within the window.                 |
| `ax, ay, az`   | Float          | Accelerometer data (**$m/s^2$**).                  |
| `wx, wy, wz`   | Float          | Gyroscope data (**$rad/s$**).                      |
| `speed`        | Float          | Synthetic GPS speed (**$m/s$**).                   |
| `label`        | Int            | Severity Class (0–4). Constant for the whole window.      |

---

## 6. Code Structure Summary

| **Function**           | **Responsibility**                                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `load_and_split_pools`     | Loads CSVs, groups by label, executes Stratified & Geographic splits.                                            |
| `smooth_transition`        | Blends two sensor arrays to remove discontinuity artifacts.                                                      |
| `generate_synthetic_speed` | Creates realistic speed profiles with noise.                                                                     |
| `synthesize_windows`       | The main engine: Stitches clips**$\to$**Injects Speed**$\to$**Slides Window**$\to$**Saves CSV. |
