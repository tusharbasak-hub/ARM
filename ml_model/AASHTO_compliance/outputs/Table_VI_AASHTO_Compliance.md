# Table VI: AASHTO R 56 Compliance & Regression Performance Comparison

| Family                      | Model Architecture                   |   MAE (m/km) |   RMSE (m/km) |   Pearson r |   R2 Score |   AASHTO R 56 Compliance (%) |
|:----------------------------|:-------------------------------------|-------------:|--------------:|------------:|-----------:|-----------------------------:|
| Linear Baselines            | Ridge Regression                     |       2.7988 |        5.812  |      0.4714 |    -1.4232 |                        11.35 |
| Tabular Ensembles           | Random Forest                        |       2.2102 |        3.1294 |      0.6742 |     0.2975 |                        15.71 |
| Tabular Ensembles           | XGBoost                              |       2.1154 |        3.0397 |      0.6881 |     0.3372 |                        16.71 |
| Tabular Ensembles           | LightGBM                             |       2.075  |        2.9793 |      0.695  |     0.3633 |                        16.5  |
| Tabular Ensembles           | CatBoost                             |       2.0303 |        2.8926 |      0.7038 |     0.3998 |                        15.34 |
| Recurrent & Sequential RNNs | 2-Layer Bi-LSTM                      |       2.3748 |        3.4643 |      0.3819 |     0.139  |                        11.19 |
| Recurrent & Sequential RNNs | 2-Layer GRU                          |       2.254  |        3.4181 |      0.4264 |     0.1619 |                        13.72 |
| Hybrid Late-Fusion          | Bi-LSTM + Context                    |       2.2984 |        3.4304 |      0.4184 |     0.1558 |                        12.82 |
| Convolutional Networks      | Standard 1D-CNN (Uncalibrated)       |       1.7051 |        3.0169 |      0.6495 |     0.3471 |                        17.97 |
| Convolutional Networks      | Proposed 1D-CNN + PICA (Section VII) |       1.6283 |        2.4468 |      0.7553 |     0.5705 |                        18.39 |

---
*Note: AASHTO R 56 compliance defines the percentage of test sections where absolute error is within max(0.25 m/km, 10% of True IRI).*
