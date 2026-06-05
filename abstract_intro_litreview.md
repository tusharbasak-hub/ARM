# Real-Time Crowdsourced Road Surface Evaluation with Multimodal Navigation: A Spatially-Invariant IRI Pipeline, Dual-Input Deep Learning, and Ride-Comfort-Aware Routing

---

## Abstract

Urban road networks suffer from a persistent condition-monitoring deficit: professional inertial profilometry vehicles are too expensive for frequent city-wide deployment, while existing smartphone-based crowdsourcing systems are impaired by three fundamental and unresolved limitations. First, time-domain IMU signals are inherently speed-dependent — the same physical road feature manifests at different temporal frequencies depending on vehicle velocity, causing state-of-the-art models to produce erroneous, discontinuous quality estimates at speed-change events (the "staircase artifact"). Second, existing systems lack a principled model of temporal road-state evolution, causing repaired defects to persist on the map indefinitely and newly appearing defects to take an unreasonably long time to surface. Third, while several works detect road anomalies, none integrates this detection into an end-to-end system that uses road quality as a live, first-class routing criterion alongside distance, presenting it to the user as a multimodal, route-comparison navigation experience together with a personal ride history.

This paper presents a complete, deployed, end-to-end system that resolves all three limitations. Calibrated ground-truth International Roughness Index (IRI) labels are obtained deterministically from the BeamNG.tech soft-body physics engine, where a Quarter-Car Golden Car simulation at the ISO-standard 80 km/h provides perfectly synchronised 100 Hz dual-LiDAR road elevation profiles and 6-DoF IMU readings, decoupling training data from vehicle-specific suspension characteristics. A dual-domain wavelet processing pipeline — a 1.11 Hz time-domain low-pass filter for chassis resonance suppression followed by a 5.4–25.0 m spatial-domain band-pass filter for macro-slope removal — re-indexes all signals from the time domain onto a uniform 1 cm spatial grid, structurally eliminating speed dependency. A dual-input deep neural network of only 11,853 parameters fuses a depthwise + standard 1D CNN spatial feature extractor with a contextual MLP that encodes mean speed and vehicle type, trained under a composite Huber/Log-Cosh loss with inverse-class-frequency pothole weighting and a false-alarm suppression penalty. The model achieves **MAE = 0.33 m/km**, **RMSE = 1.18 m/km**, and **R² = 0.76** and is deployed on Android as a quantised TensorFlow Lite model for real-time background inference. The distributed backend — built on Node.js, MongoDB, and Redis — aggregates crowdsourced observations with exponential time decay ($\tau = 24$ h), achieving a self-healing map in which repaired roads are automatically down-weighted without administrator intervention. Sub-second live updates are routed to clients via Geohash-precision-6 WebSocket namespacing. A multimodal routing service scores Mapbox-derived candidate routes by distance-weighted road quality, presents the user with a ranked route comparison (smoothest vs. fastest vs. balanced), and stores each completed journey as a personal ride-history record with an overall Ride Comfort Score — enabling retrospective analysis of a user's cumulative road experience. To our knowledge, this is the first system to unify simulation-derived IRI ground truth, spatial-domain signal processing, temporally adaptive crowdsourcing, and ride-comfort-aware multimodal navigation in a single, end-to-end deployed platform.

**Index Terms** — International Roughness Index, road surface monitoring, crowdsourcing, smartphone sensing, spatial domain processing, 1D convolutional neural network, real-time systems, infrastructure-aware routing, ride comfort score, temporal decay, Geohashing, BeamNG simulation.

---

## I. Introduction

### A. The Global Road Deterioration Problem

Road pavement is the most capital-intensive component of urban infrastructure. In India alone, the road network spans over 6.3 million kilometres [CITE-MoRTH], yet a large fraction of this network — particularly secondary and tertiary roads — receives structural condition assessment no more frequently than once every three to five years, if at all [CITE-WorldBank]. The World Bank estimates that the cumulative economic cost of poor road condition to vehicle owners exceeds **$500 billion annually** worldwide, through accelerated tyre wear, fuel overconsumption, drivetrain damage, and road accident costs attributable to surface defects [CITE-WorldBank2019]. At the extremes of the spectrum, potholes — full-depth structural failures caused by the interaction of water infiltration, traffic loading, and thermal cycling — are responsible for thousands of vehicle damage incidents and road accidents annually, and their formation can progress from surface cracking to a full pothole within a matter of weeks under heavy precipitation [CITE-NRDA].

The international standard for quantifying road roughness is the **International Roughness Index (IRI)**, defined by the World Bank Quarter-Car model and standardised in ISO 8608 and ASTM E1926 [CITE-Sayers1986]. IRI represents the cumulative vertical displacement of a standardised quarter-car suspension traversing a road at 80 km/h, expressed in metres per kilometre (m/km). Professional measurement uses inertial laser profilometry vehicles whose procurement and operational costs are prohibitive for routine city-scale deployment. This creates a fundamental monitoring gap: the condition of the road network degrades continuously between surveys, and defects go unreported for months.

The challenge of closing this monitoring gap — with high spatial resolution, at low cost, and in near-real-time — motivates this work.

### B. The Smartphone Sensing Opportunity

Every smartphone in use today is a capable mobile sensor platform: it contains a multi-axis MEMS inertial measurement unit (IMU), a GNSS receiver, and a persistent mobile data connection. With over 6.8 billion smartphones in global use [CITE-Statista2024], any vehicle in which a smartphone is mounted constitutes a potential road condition probe. If even a small fraction of the daily vehicle trips on a road network contribute passive IMU measurements, the aggregate creates a continuously updated, spatially dense picture of road surface condition at near-zero marginal cost — a paradigm known as **participatory or opportunistic sensing** [CITE-Ganti2011].

This vision has motivated a growing body of research since Mohan et al.'s foundational Nericell system (2008) [CITE-Mohan2008]. However, despite more than fifteen years of effort, no existing system achieves the full promise of this paradigm. Three fundamental limitations persistently appear across the literature and remain unresolved.

### C. Three Unsolved Limitations of the State of the Art

**Limitation 1 — Speed-Dependent Signal Distortion and the Staircase Artifact.**

IMU sensors produce measurements as a function of time. Road roughness, however, is a geometric property of a spatial surface. The relationship between the two is mediated by vehicle speed $v$: a surface undulation of spatial wavelength $\lambda$ (metres) produces a temporal frequency $f = v / \lambda$ in the IMU signal. This means that the **same physical road feature appears at different temporal frequencies at different speeds.** At $v = 10$ km/h, a 5-metre undulation appears at $\approx 0.56$ Hz; at $v = 50$ km/h, the same feature appears at $\approx 2.78$ Hz — a five-fold shift. Any time-domain filter or time-domain machine learning model (including convolutional neural networks applied to raw IMU time series) is therefore inherently speed-dependent.

The practical consequence is the "staircase artifact": at locations where vehicle speed changes abruptly — intersections, pedestrian crossings, traffic lights, roundabouts — time-domain processing pipelines produce discontinuous, step-change jumps in estimated road quality even on physically uniform surfaces. This is because the pipeline's effective spatial frequency resolution changes with speed, and the transition appears as a sudden change in the measured roughness signal. Mukherjee et al. [CITE] acknowledge speed as a confounding factor and apply post-hoc normalisation; Sattar et al. [CITE-Sattar2018] restrict experiments to highway driving at near-constant speed; the comprehensive review by the MDPI 2024 survey [CITE-Comprehensive2024] identifies speed variation as one of the two primary open challenges in the field. No prior system has structurally eliminated this artifact through a full spatial-domain signal processing pipeline.

**Limitation 2 — No Temporal Road State Evolution (The Stale Map Problem).**

A road is not a static object. Its condition evolves continuously: a pothole that appears in February after winter frost-damage may be repaired by the municipal authority in March. A smooth surface may develop cracking by summer under heavy truck traffic. Any crowdsourced road condition map must model this temporal evolution — otherwise, the map becomes progressively stale and misleading.

Yet no existing smartphone-based road monitoring system implements a principled temporal decay model. Once a defect observation is stored, it persists indefinitely. Citizen reporting platforms such as SeeClickFix and FixMyStreet require explicit administrator action to close a report [CITE-Goodall2023]. Academic systems including iDriveSense [CITE-Kalim2018] and Roadroid [CITE-Eriksson2013] store observations without temporal weighting. Waze's pothole reporting, evaluated by Goodall (2023) [CITE-Goodall2023], clears reports after 30 minutes unless confirmed — a heuristic that bears no relationship to the actual timescale of road repair. The result is that in all prior systems, **a road repaired today will still show as damaged on the map for an indeterminate period,** severely degrading user trust and map utility.

**Limitation 3 — Absence of End-to-End Multimodal Navigation Integration.**

The ultimate utility of a road condition sensing system is not just awareness — it is the ability to act on that awareness. A driver who knows that Route A has a pothole cluster 2 km ahead should be able to choose Route B, with full knowledge of the trade-off between the extra distance and the improved ride quality. This is the multi-criteria route planning problem: finding a route that optimises a composite objective of travel distance, travel time, and infrastructure quality.

Despite this obvious utility, no existing peer-reviewed system delivers a complete, live, navigation-integrated deployment in which road quality derived from crowdsourced smartphone sensing is used as a first-class routing criterion. iDriveSense [CITE-Kalim2018] proposes a fuzzy-logic route scoring mechanism but relies on manually annotated road quality maps rather than a continuously updated crowdsourced layer, and is not deployed as a working mobile application. The system of Dewangan et al. [CITE] and the pothole avoidance framework of Balakuntala et al. [CITE] are conceptual frameworks without real-time backend integration. Commercial navigation systems (Google Maps, HERE, Waze) do not incorporate independently computed IRI-based quality scores. This gap — between a working crowdsourced sensing backend and a real, multimodal navigation experience for the end user — represents the most practically impactful limitation in the field.

### D. Additional Gaps in Prior Work

Beyond the three primary limitations, a systematic review of the literature reveals several further recurring weaknesses:

**Vehicle-Suspension Coupling and the Cross-Vehicle Generalisation Problem.** Virtually all empirical training datasets are collected from a single vehicle or a small number of homogeneous vehicles. The suspension system of a rigid heavy vehicle filters road surface inputs very differently from a soft-sprung passenger car: the same road profile produces different IMU signal amplitudes, damping profiles, and resonant frequencies across vehicle classes. A model trained on data from one vehicle type and deployed on another exhibits systematic, uncorrected bias [CITE-Mukherjee, CITE-Sattar2018]. Only a handful of works attempt cross-vehicle normalisation, and none solve the problem from first principles.

**Absence of Physically Calibrated Ground Truth.** The majority of published systems label their training data with subjective human ratings (good/fair/poor), manual event annotations, or binary thresholds on the z-axis accelerometer [CITE-Mednis2011, CITE-Eriksson2013]. These labels are not reproducible, not physically calibrated, and not inter-comparable across studies. The IRI is the internationally standardised metric of road roughness [CITE-Sayers1986], but obtaining true IRI ground truth in the field requires co-registered laser profilometry — a logistical challenge that most papers do not address. This absence of calibrated ground truth makes rigorous comparison between published systems impossible.

**Computational Deployability.** Systems that achieve high detection accuracy often do so at the cost of models too large or computationally expensive for continuous background inference on a smartphone [CITE-Comprehensive2024]. Large LSTM networks, multi-scale ResNets, and attention-based transformers may achieve strong performance on server-side batch evaluation but are impractical as always-on on-device sensors due to battery and thermal constraints.

**No Personal Ride History or Longitudinal Comfort Analytics.** No existing system maintains a per-user ride history that allows individuals to understand their cumulative road experience over time — which routes they regularly travel, how the quality of those routes has evolved, and what their average Ride Comfort Score is. This longitudinal dimension is entirely absent from the academic literature, despite its obvious value for urban mobility analytics and municipal reporting.

### E. Contributions of This Paper

This paper presents a complete, deployed, end-to-end system for real-time crowdsourced road surface evaluation and multimodal navigation. Our specific contributions, each addressing one of the gaps identified above, are as follows:

1. **Speed-invariant spatial-domain signal processing.** We introduce a dual-domain wavelet pipeline that rigorously transforms all IMU and LiDAR signals from the irregular time domain onto a uniform 1 cm spatial grid, applying a 1.11 Hz time-domain low-pass filter for chassis resonance suppression and a 5.4–25.0 m spatial-domain band-pass filter for macro-slope removal. This structural reformulation eliminates speed dependency and the staircase artifact.

2. **Physics-simulation-derived, vehicle-agnostic ground truth via BeamNG.tech.** We exploit the BeamNG.tech soft-body physics simulator — the same platform used for ADAS research and autonomous driving validation [CITE-BeamNG] — to generate synchronised (IMU, LiDAR profile) pairs across multiple vehicle models at the ISO-standard 80 km/h, providing deterministic, noise-free IRI labels that cannot be practically obtained from field surveys.

3. **Dual-input deep neural network with physically motivated architecture.** We propose an 11,853-parameter model combining depthwise-separable 1D convolutions (for per-channel spatial feature extraction) with a contextual MLP (for speed- and vehicle-type-conditioned regression), trained under a composite Huber/Log-Cosh loss function with inverse-class-frequency pothole weighting and a false-alarm suppression mechanism. The model is deployed on Android via TensorFlow Lite post-training quantisation.

4. **Temporally adaptive crowdsourced map with self-healing consistency.** We implement an exponential time-decay aggregation model ($w \propto e^{-t/\tau}$) in which every observation's contribution to a road segment's quality score diminishes with age. This enables repaired road defects to be automatically and continuously down-weighted without any manual intervention — a property we term **self-healing map consistency**, not previously described in the literature.

5. **Sub-second real-time broadcast via Geohash WebSocket namespacing.** The distributed backend routes road quality events exclusively to clients whose geographic viewport overlaps the affected area, using precision-6 Geohash strings as Socket.IO room identifiers. This achieves O(R) broadcast complexity (R = clients in a ≈ 1.2 km × 0.61 km cell) without global broadcast overhead.

6. **Multimodal ride-comfort-aware routing.** A routing service retrieves multiple candidate routes from the Mapbox Directions API, scores each by a distance-weighted composite of IRI-based road quality and route length, and presents the user with a ranked multi-route comparison. Users can select their preferred balance between ride comfort and distance — directly addressing the multi-criteria route planning gap identified above.

7. **Personal Ride Comfort Score and trip history.** Each completed journey is stored as a user-associated record containing the route geometry, mean IRI, and an overall Ride Comfort Score derived from the spatial quality distribution of the traversed road. This enables longitudinal personal mobility analytics and provides municipalities with user-linked pavement condition reports.

### F. Paper Organisation

Section II surveys related work across road surface monitoring, crowdsourcing architectures, and infrastructure-aware routing. Section III details the complete methodology and system architecture. Section IV presents experimental results. Section V discusses current limitations and future work — including the Pothole Classification Model under active development. Section VI concludes.

---

## II. Related Work

### A. Traditional Road Condition Assessment

The established method for road condition assessment is the **inertial laser profilometer**: a vehicle-mounted system that uses laser displacement sensors and a high-precision IMU to measure the road surface profile at driving speeds of 60–100 km/h, computing IRI in accordance with ASTM E1926 [CITE-Sayers1986]. These systems achieve high accuracy (IRI precision ±0.1 m/km) but cost upward of $200,000 USD per vehicle, require skilled operators, and are logistically suited only to periodic surveys — not continuous monitoring. Some agencies use Response Type Road Roughness Measurement Systems (RTRRMS), such as bump integrators mounted in standardised vehicles, which are cheaper but provide lower accuracy and require correlation calibration [CITE-Sayers1986].

Ground-penetrating radar, 3D LiDAR scanning, and photogrammetric inspection from unmanned aerial vehicles (UAVs) have been proposed for crack detection and structural assessment [CITE-Sattar2018]; these methods provide rich surface geometry but at high per-km cost and low temporal frequency, making city-scale continuous monitoring impractical.

### B. Smartphone-Based Road Surface Monitoring

#### B.1 Threshold and Rule-Based Detection

The earliest smartphone sensing approaches used simple threshold rules on the z-axis accelerometer to flag individual pothole events. Mednis et al. [CITE-Mednis2011] demonstrated that z-axis threshold crossings on an Android phone could detect potholes with true positive rates of up to 90% in controlled conditions — a foundational result that established the feasibility of smartphone-based detection. Mohan et al. (Nericell, 2008) [CITE-Mohan2008] expanded this to a multi-modal approach using the accelerometer, microphone, and GPS, detecting road roughness, potholes, speed bumps, and braking events. These threshold methods are fast and interpretable but are inherently speed-dependent (a larger z-axis spike is produced at higher speed by the same pothole) and produce high false-alarm rates at speed-change events and over urban railway crossings.

#### B.2 Signal Processing and IRI Estimation

A second class of methods applies classical signal processing to derive IRI-correlated metrics from smartphone accelerometer data. Eriksson et al. (Pothole Patrol, 2008) [CITE-Eriksson2008] used a Fourier-based spectral analysis to classify road surface quality from accelerometer data collected by taxis in Boston. Perttunen et al. [CITE-Perttunen2011] applied a zero-phase Butterworth band-pass filter in the time domain and showed that the filtered vertical acceleration correlates with subjective pavement ratings. Roadroid [CITE-Eriksson2013] — the most widely deployed commercial smartphone-based IRI system — uses a proprietary algorithm on smartphone vertical acceleration and has been validated against laser profilometry with correlation coefficients of 0.6–0.8 depending on vehicle type and speed range.

A recurring finding in this literature is that speed variation is the dominant source of error. Mukherjee et al. [CITE-Mukherjee2022] explicitly model speed as a confounding variable and attempt post-hoc normalisation with partial success. Islam et al. [CITE] propose speed-adaptive filter cutoffs for time-domain filtering, improving performance over a wider speed range but not eliminating the fundamental speed-distortion problem. None of these works reformulate the problem in the spatial domain.

The proposal by Hanson et al. [CITE] and the method of González et al. [CITE] use the quarter-car model with smartphone data, converting time-domain acceleration to IRI via the Golden Car parameters — the same approach we adopt, but applied after spatial domain transformation rather than in the time domain. Our work shows that performing this conversion on spatially-reindexed data eliminates the speed-dependent bias that affects all prior implementations.

#### B.3 Machine Learning Approaches

More recent work applies supervised machine learning to road quality classification. Sattar et al. [CITE-Sattar2018] provide a comprehensive review of feature extraction and classification methods up to 2018, concluding that Support Vector Machines and Random Forests achieve the highest accuracy on curated datasets but that cross-vehicle generalisation remains an open problem. The automated pothole detection system of Mohamed et al. [CITE-Mohamed2020] employs SVM and Random Forest classifiers on smartphone vibration data, reporting precision of up to 96.8% in within-vehicle evaluation.

Deep learning methods have recently emerged. IRI-Net [CITE-IRI-Net] applies a fully connected neural network to averaged accelerometer features for IRI prediction. Convolutional approaches have been proposed for event detection [CITE-CNN-road], showing that CNNs can learn discriminative spatial patterns in accelerometer sequences. However, the critical observation — documented in the MDPI 2024 comprehensive review [CITE-Comprehensive2024] — is that these CNN models operate on time-domain sequences and are therefore as speed-dependent as their signal-processing predecessors. Our system is the first to train a 1D CNN on spatially-indexed sequences, decoupling the learned features from vehicle speed.

The work most architecturally related to ours is that of Jeong et al. (IRI-Net) [CITE] and the dual-channel architecture of Sattar et al. [CITE], which fuse multiple feature streams before regression. Our contribution extends this fusion paradigm by explicitly separating per-channel spatial feature extraction (depthwise convolution) from cross-channel pattern recognition (standard convolution) and from macroscopic statistical context (MLP), with a physically motivated justification for each architectural choice.

#### B.4 The Role of Synthetic and Simulation Data

The use of synthetic data to supplement or replace empirical field collection has grown substantially in machine learning for transportation. Sharifi Renani et al. [CITE-Sharifi2021] demonstrate that training IMU-based kinematic prediction models on synthetic data significantly outperforms training on field-collected data alone (RMSE reduced by 38–54%), validating the sim-to-real transfer paradigm for IMU applications. In the autonomous driving domain, BeamNG.tech has been adopted as a research simulation environment [CITE-BeamNG], with its Python interface BeamNGpy enabling programmatic sensor configuration, scenario scripting, and data collection [CITE-BeamNGpy]. To our knowledge, our work is the first to use BeamNG.tech specifically for the generation of synchronised (IMU, road profile LiDAR) training data for road roughness estimation — exploiting the simulator's soft-body physics fidelity to model vehicle suspension dynamics with sufficient accuracy for IRI-correlated ground truth.

### C. Crowdsourcing Architectures for Road Monitoring

#### C.1 Data Aggregation and Quality

Crowdsourced road quality maps require robust aggregation methods that handle observation noise, spatially non-uniform sampling density, and temporal staleness. Yi et al. [CITE] propose a Bayesian aggregation framework that weights observations by estimated reliability. Mirtabar et al. [CITE-Mirtabar2022] develop a crowdsourcing-based IRI platform that uses median aggregation within spatial bins to suppress outliers. The comprehensive review by Sattar et al. [CITE-Sattar2018] identifies temporal decay as a desirable but unimplemented feature in all surveyed systems.

The HDMAP crowdsourcing review by [CITE-HDMap2024] identifies "data freshness" as the primary open challenge for crowdsourced mapping systems in the autonomous driving context, noting that the state of the art relies on explicit versioning and administrator-driven invalidation rather than continuous temporal decay. Our work addresses this gap directly.

#### C.2 Geospatial Indexing and Real-Time Broadcast

Efficient geospatial indexing is foundational to any real-time road quality system. The Geohash algorithm [CITE-Niemeyer2008] encodes geographic coordinates as compact alphanumeric strings with a hierarchical spatial structure, making it well-suited for region-based pub/sub architectures. Commercial traffic platforms (Waze, HERE) use variants of geospatial cell indexing for efficient event routing, but do not publish their architectures in the academic literature. Our system's use of precision-6 Geohash strings as Socket.IO room names provides a formally documented, reproducible implementation of geospatial pub/sub with O(R) broadcast complexity, which we present as a reusable architectural pattern.

### D. Multimodal and Multi-Criteria Route Planning

#### D.1 Classical Route Planning

Classical route planning (Dijkstra, A*, Bellman-Ford) optimises a single scalar edge cost — typically travel time or distance [CITE-Dijkstra]. Multi-criteria extensions model Pareto-optimal route sets under multiple objectives (time, cost, comfort), but the road condition dimension is absent from standard navigation graph representations: OpenStreetMap, the primary open-source map base, carries no IRI or pothole data in its default schema.

#### D.2 Road-Quality-Aware Routing

iDriveSense [CITE-Kalim2018] is the most closely related work to our routing component. It applies a two-stage fuzzy logic system: the first stage classifies each road segment from crowdsensed IMU data into one of several quality classes; the second stage recommends routes by weighting segments by their quality classification. The key distinction from our system is that iDriveSense uses a static, pre-annotated road quality map rather than a live, continuously updated crowdsourced layer, and does not present the user with a multi-route comparison showing the explicit trade-off between distance and road quality.

Balakuntala et al. [CITE] propose a graph-theoretic framework for pothole-avoiding routing, but this is a conceptual framework without a deployed implementation. Dewangan et al. [CITE] describe an intelligent pothole detection and avoidance system but similarly lack end-to-end deployment. A patent by Ford Motor Company [CITE-Ford-patent] describes ride-quality-aware route planning using in-vehicle suspension data, but this is proprietary and vehicle-specific rather than crowdsourced and smartphone-based.

The GenMRP framework [CITE-GenMRP2026] addresses personalised multi-route planning using historical user preferences but does not incorporate road surface quality as an input dimension. Commercial navigation applications (Google Maps, Waze, Apple Maps) do not use independently computed IRI-based road quality as a routing criterion. Our system fills this gap with a fully deployed, IRI-informed, multi-route comparison interface.

#### D.3 Ride Comfort and Trip History

The concept of a per-trip ride comfort score has been explored in adjacent domains. Ridergo [CITE-Ridergo2021] estimates passenger comfort in ride-hailing vehicles by mapping smartphone accelerometer patterns to a comfort index, with application to driver rating. The Ride Report mobile application [CITE-RideReport] records per-trip bicycle comfort ratings and aggregates them into a colour-coded comfort map. The University of Birmingham's smartphone comfort measurement system [CITE-Birmingham] uses neural networks to map accelerometer data to ISO 2631-based ride comfort metrics for railway passengers.

Our Ride Comfort Score adapts these concepts to the automotive urban road context: rather than a subjective self-report or a single-axis comfort index, our score is derived directly from the IRI-based road quality of each metre of the traversed route, stored as a persistent per-user trip record, and queryable for longitudinal trend analysis. This combination of automated, objective scoring, spatial granularity, and longitudinal persistence is, to our knowledge, novel.

### E. Positioning of This Work

Table I summarises the key dimensions along which this work differs from the most relevant prior systems.

| Feature | Nericell [2008] | iDriveSense [2018] | Roadroid [2013] | Sattar Review [2018] | **Ours** |
|---|:---:|:---:|:---:|:---:|:---:|
| IRI-calibrated ground truth | ✗ | ✗ | Partial | ✗ | **✓** |
| Spatial-domain processing | ✗ | ✗ | ✗ | ✗ | **✓** |
| Staircase artifact eliminated | ✗ | ✗ | ✗ | ✗ | **✓** |
| Simulation-derived training data | ✗ | ✗ | ✗ | ✗ | **✓** |
| Temporal decay / self-healing map | ✗ | ✗ | ✗ | ✗ | **✓** |
| Real-time WebSocket broadcast | ✗ | ✗ | ✗ | ✗ | **✓** |
| Multi-route ride-comfort navigation | ✗ | Partial | ✗ | ✗ | **✓** |
| Personal ride history + comfort score | ✗ | ✗ | ✗ | ✗ | **✓** |
| On-device TFLite deployment | ✗ | ✗ | Partial | ✗ | **✓** |

*Table I: Comparison of the proposed system with representative prior work across key dimensions.*

The remainder of this paper details the technical architecture that realises each of these contributions.
