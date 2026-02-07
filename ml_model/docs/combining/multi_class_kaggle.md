# Data Processing and Labeling Strategy

This document explains the "combining" phase of my data preparation pipeline, where I process raw sensor data from multiple folders (PVS 1-9) into a single, clean dataset for model training.

## 1. Centering the MPU Data

The raw dataset provided MPU (accelerometer and gyroscope) readings for the **Left** and **Right** sides of the dashboard. However, for my application, I need the sensor data to represent the **Center** of the dashboard of the vehicle.

I applied standard rigid body mechanics. Since the dashboard creates a rigid connection between the two sensors:

1. **Linear Acceleration**: The acceleration at the midpoint is the average of the accelerations at the two ends.
2. **Angular Velocity**: Rotational velocity is constant across a rigid body. I average the left and right values to reduce sensor noise.
3. **GPS**: Latitude, longitude, and speed are also averaged to smooth out GPS drift.

**Visualizing the Logic:**

$$
\begin{array}{c}
\text{Left Sensor} \quad \xrightarrow{\quad d \quad} \quad \text{Midpoint} \quad \xleftarrow{\quad d \quad} \quad \text{Right Sensor} \\
\bigg\downarrow & \bigg\downarrow & \bigg\downarrow \\
\vec{a}_{L}, \vec{\omega}_{L} & \vec{a}_{mid} = \frac{\vec{a}_{L} + \vec{a}_{R}}{2} & \vec{a}_{R}, \vec{\omega}_{R} \\
& \vec{\omega}_{mid} \approx \frac{\vec{\omega}_{L} + \vec{\omega}_{R}}{2} &
\end{array}
$$

*The midpoint readings are derived by averaging the "noisy" endpoints to get a stable center reading.*

This step resulted in a generated file `dataset_gps_mpu_mid.csv` for each folder.

## 2. Renaming Columns

The original column headers (e.g., `acc_x_dashboard`, `gyro_x_dashboard`) were verbose and difficult to read. I renamed them to standard scientific subscripts:

* `ax`, `ay`, `az` (Acceleration)
* `wx`, `wy`, `wz` (Angular Velocity / Omega)

## 3. Removing Unused Files

I cleaned up the directories by deleting files that are outside the scope of this project (like `video_dataset_left.mp4`, `map.html`, etc.), keeping only the essential CSVs and the right-side environment video for verification.

## 4-7. Road Quality Labeling Logic

I transformed the raw boolean flags from `dataset_labels.csv` into a single multi-class target variable `RoadQuality` (0-3).

The labels were assigned based on a priority system. I check for the worst conditions first (Priority 3). If a row matches a higher priority mask, it is assigned that label, and subsequent lower-priority checks ignore it.

**Label Definition:**

* **0**: Smooth road (Highway/Good City Road)
* **1**: Rough but good (Patched road)
* **2**: Cobblestone / Bad
* **3**: Unpaved / Dirt / Very Bad

**Logic Equations:**

**Priority 3 (Worst Quality): dirt road (small village road vibes)**

$$
\text{Quality} = 3 \iff \left[ (\text{Cobble} \land \neg \text{NoSpeedBump}) \land (\text{BadLeft} \lor \text{BadRight}) \right] \lor (\text{DirtRoad})
$$

**Priority 2: a road ruined in rain**

$$
\text{Quality} = 2 \iff (\text{Cobble}) \land (\text{Quality is unset})
$$

**Priority 0 (Best Quality): yamuna or delhi-mumbai expressway level**

$$
\text{Quality} = 0 \iff (\text{Paved} \land \text{Asphalt} \land \text{GoodLeft} \land \text{GoodRight} \land \text{NoSpeedBump}) \land (\text{Quality is unset})
$$

**Priority 1 (Average Quality): average indian badly patched road** 

$$
\text{Quality} = 1 \iff (\text{Paved} \land \text{Asphalt}) \land (\text{Quality is unset})
$$

## 8. Downsampling Strategy

The dataset was recorded at a high frequency of **100Hz**, but standard mobile phones typically collect sensor data reliability at **10Hz**. To make my model lightweight and realistic for mobile deployment, I needed to downsample.

**Logic:**
Instead of simply taking every 10th row (which might drift if the sampling rate wasn't perfectly consistent), I used a time-delta approach. I iterate through the dataset and only keep a new row if its timestamp is at least **0.1 seconds** greater than the timestamp of the last kept row. This guarantees a consistent 10Hz sampling rate relative to real-time.

## 9. Combining Data

I merged the processed `dataset_gps_mpu_mid.csv` files from all PVS folders into two files: PVS 1, 2, 4, 5, 7, 8 into train and 3, 6, 9 into test at `ml_model\data\combined\multi_class_kaggle`
splited this way to make sure that model is tested on a completly new road, but it should be trained on all vehicle types
| Dataset | Vehicle             | Driver   | Scenario   | Distance |
|----------|---------------------|----------|------------|----------|
| PVS 1    | Volkswagen Saveiro  | Driver 1 | Scenario 1 | 13.81 km |
| PVS 2    | Volkswagen Saveiro  | Driver 1 | Scenario 2 | 11.62 km |
| PVS 3    | Volkswagen Saveiro  | Driver 1 | Scenario 3 | 10.72 km |
| PVS 4    | Fiat Bravo          | Driver 2 | Scenario 1 | 13.81 km |
| PVS 5    | Fiat Bravo          | Driver 2 | Scenario 2 | 11.63 km |
| PVS 6    | Fiat Bravo          | Driver 2 | Scenario 3 | 10.73 km |
| PVS 7    | Fiat Palio          | Driver 3 | Scenario 1 | 13.78 km |
| PVS 8    | Fiat Palio          | Driver 3 | Scenario 2 | 11.63 km |
| PVS 9    | Fiat Palio          | Driver 3 | Scenario 3 | 10.74 km |


## 10. Unit Conversion

Mobile devices (Android/iOS) typically provide angular velocity in **radians/second**, but the Kaggle dataset used **degrees/second**. To ensure compatibility with my app's input, I converted all gyro readings:

$$
\omega_{rad} = \omega_{deg} \times \frac{\pi}{180}
$$

## 11. Visualization

Finally, I plotted the sensor values and GPS tracks, color-coding the points by their assigned `RoadQuality`. This allowed me to visually verify that the logic in steps 4-7 correctly identified the road conditions observed in the reference videos.
