# Real-Time Crowdsourced Road Surface Evaluation with Temporal Decay: A Spatially-Invariant IRI Pipeline, Dual-Input Deep Learning, and Infrastructure-Aware Routing

---

## Abstract

Continuous, high-resolution monitoring of road surface condition across urban networks remains an unsolved challenge. Traditional inertial profiling equipment is prohibitively expensive for city-scale deployment, while existing smartphone-based crowdsourcing approaches suffer from two fundamental and largely unaddressed limitations: (1) speed-dependent distortion of time-domain sensor signals that causes the same physical road feature to appear at different frequencies depending on vehicle velocity—the root cause of the so-called "staircase artifact"—and (2) the absence of a temporally adaptive data model, meaning that once a road defect is logged it persists indefinitely in the system's map even after the defect has been repaired.

This paper presents a complete end-to-end system for real-time road surface evaluation that resolves both limitations. Ground-truth training data is generated deterministically using the BeamNG.tech soft-body physics simulator, where synchronized 100 Hz dual-LiDAR road elevation and 6-DoF IMU signals are collected from a Quarter-Car (Golden Car) simulation at the ISO-standard 80 km/h, establishing a vehicle-agnostic ground truth for International Roughness Index (IRI) computation. A two-stage dual-domain wavelet processing pipeline—comprising a time-domain low-pass filter (1.11 Hz cutoff) for chassis resonance suppression followed by a spatial-domain band-pass filter (5.4 m–25.0 m wavelength) for hill-slope removal—translates all signals from the time domain into a uniform 1 cm spatial grid, structurally eliminating speed dependency. A lightweight dual-input deep neural network (11,853 parameters) fuses 1D depthwise convolutional spatial features with a contextual MLP processing mean speed and vehicle-type statistics, trained under a composite Huber/Log-Cosh loss function with inverse class frequency weighting and a false-alarm suppression mechanism. The model achieves a Mean Absolute Error of **0.33 m/km**, RMSE of **1.18 m/km**, and R² of **0.76** on held-out test data, and is deployed on Android as a TensorFlow Lite model. The distributed backend implements a three-tier Node.js/MongoDB/Redis architecture in which crowdsourced road observations are map-matched via OSRM, aggregated with exponential time decay, broadcast to subscribed clients in sub-second latency via Geohash-namespaced WebSocket rooms, and applied to OSRM-based route scoring that recommends the smoothest path between any two points. The temporal decay model ensures that repaired potholes are naturally down-weighted and eventually removed from the live map without any manual intervention. Taken together, this system represents the first fully integrated pipeline combining physics-simulation-derived ground truth, spatial-domain signal processing, real-time distributed aggregation with temporal decay, and infrastructure-aware navigation.

**Index Terms** — International Roughness Index, road surface monitoring, crowdsourcing, smartphone sensing, spatial domain processing, deep learning, real-time systems, infrastructure-aware routing, temporal decay.

---

## I. Introduction

### A. The Global Road Deterioration Problem

Road infrastructure is the circulatory system of economic activity. The World Bank estimates that the global cost of poor road conditions to vehicle owners—through accelerated wear, fuel overconsumption, and tire damage—exceeds **\$500 billion annually** [CITE-WorldBank]. In rapidly urbanising economies, the gap between road network growth and maintenance capacity is widening: the Indian road network alone spans over 6.3 million km, making exhaustive professional inspection operationally impossible at any reasonable frequency [CITE]. The consequence is a systematic maintenance deficit in which defects go undetected for months or years, escalating from minor surface cracking into full-depth structural failures—potholes—that cause vehicle damage, accidents, and rider injury.

The conventional standard for road condition assessment is the **International Roughness Index (IRI)**, standardised by the World Bank and ISO 8608, which quantifies the vertical displacement experienced by a standardised quarter-car suspension system traversing a road at 80 km/h, expressed in metres of suspension stroke per kilometre of road (m/km). Professional measurement is performed by inertial laser profilometry vehicles costing upward of \$200,000 USD and requiring skilled operators, making city-wide surveys feasible only once every two to five years. This temporal resolution is wholly inadequate for tracking the rapid development and—critically—the repair of surface defects.

### B. The Promise and Limitations of Smartphone-Based Crowdsourcing

The proliferation of smartphones—each equipped with a multi-axis MEMS IMU (accelerometer + gyroscope), GPS, and persistent network connectivity—has motivated a substantial body of research into crowdsourced road condition monitoring [CITE-Sattar2018, CITE-Mednis2011, CITE-Perttunen2011]. The fundamental appeal is clear: if every vehicle traversing a road network passively measures and reports surface roughness, a complete, continuously updated road condition map can be assembled at near-zero marginal cost.

However, the literature reveals two **fundamental and persistent limitations** that no prior system has simultaneously resolved:

**Limitation 1 — Speed-Dependent Signal Distortion (The "Staircase Artifact").**
Smartphone IMU sensors record data as a function of *time*, but road roughness is a property of *space*. A surface undulation of spatial wavelength $\lambda$ produces a temporal frequency in the IMU signal of $f = v / \lambda$, where $v$ is the vehicle speed. At $v = 10$ km/h, a 5-metre wavelength undulation appears at $\approx 0.55$ Hz; at $v = 50$ km/h, the same undulation appears at $\approx 2.78$ Hz. Time-domain filters and time-domain machine learning models (including convolutional neural networks operating on raw time series) are consequently speed-dependent: a model trained predominantly on highway data fails on urban low-speed data, and vice versa. This manifests as the "staircase artifact"—abrupt discontinuities in estimated road quality scores at speed-change events (intersections, traffic lights, roundabouts) even on physically homogeneous road surfaces [CITE-Perttunen, CITE-Sattar]. Existing literature either ignores this problem, applies post-hoc speed normalisation with limited effectiveness, or restricts studies to a narrow, controlled speed range [CITE-Mukherjee2022].

**Limitation 2 — The Absence of Temporal Road State Evolution.**
No existing crowdsourced road quality system implements a principled model of **road state temporal evolution**. Once a pothole is detected and logged, it persists in the system's database until a human administrator manually removes it—even if the pothole was repaired the following day. Conversely, a newly appearing defect on a previously "good" road segment cannot degrade the segment's quality score instantaneously; it must accumulate sufficient observations to overcome the historical inertia of the aggregate score. Platforms such as SeeClickFix and FixMyStreet rely on manual citizen confirmation or administrative action to close reports [CITE-Goodall2023]; academic systems such as iDriveSense [CITE-Kalim2018] and the system of Mohan et al. [CITE-Mohan2008] do not model temporal decay at all. The result is that these maps become **progressively stale**—a well-maintained road may carry a "bad" annotation for months after repair, eroding user trust and utility.

### C. Additional Limitations in the Prior Art

Beyond these two primary limitations, a survey of the literature reveals several further gaps:

**Vehicle-suspension coupling.** Empirical datasets collected from one vehicle type cannot be directly used to train models for another. The suspension system of a rigid commercial truck filters road surface inputs very differently from that of a soft-sprung passenger car—yet virtually all existing systems collect training data from a single vehicle and make no attempt to model or compensate for this cross-vehicle variability [CITE-Mukherjee, CITE-Sattar2018].

**Absence of standardised ground truth.** The IRI is a physically meaningful, internationally standardised metric grounded in a known dynamical model. Yet the majority of published smartphone-based systems use subjective human labels (good/fair/poor), manual event annotations, or single-device z-axis threshold crossings as "ground truth" [CITE-Mednis, CITE-Eriksson]. These labels are not reproducible, not physically calibrated, and not comparable across studies.

**Lack of end-to-end integration.** Prior systems focus on one component of the pipeline—a detection algorithm, a mobile app, or a visualisation layer—without delivering a complete, deployed stack from sensor to routable infrastructure-quality map. iDriveSense [CITE] addresses routing but uses fuzzy system scoring without a learned model. The system of Mohan et al. (Nericell, 2008) [CITE] pioneered in-car sensing but predates modern deep learning. More recent learning-based approaches [CITE-Sattar2018, CITE-Comprehensive2024] demonstrate detection accuracy but do not produce a navigable, real-time-updated road quality layer.

**Computational deployability.** Systems employing large deep learning models (ResNets, LSTMs, attention mechanisms) report high accuracy in controlled evaluations but are unsuitable for continuous background inference on a smartphone, where battery consumption and thermal constraints are binding [CITE-Comprehensive2024].

### D. Contributions of This Work

This paper presents **RoadSense** (name), a complete, deployed, end-to-end system for real-time crowdsourced road surface evaluation that addresses all of the above limitations. Our specific contributions are:

1. **Speed-invariant spatial domain processing.** We introduce a two-stage dual-domain wavelet processing pipeline that rigorously transforms all IMU and LiDAR signals from the time domain into a uniform 1 cm spatial grid, structurally eliminating speed-dependent distortion and the staircase artifact. This is, to our knowledge, the first road roughness pipeline to implement fully spatial-domain processing on crowdsourced smartphone data.

2. **Physics-simulation-derived, vehicle-agnostic ground truth.** We exploit the BeamNG.tech soft-body physics simulation engine to generate perfectly synchronised, noise-free pairs of (IMU signal, LiDAR road profile) across multiple vehicle models and map environments. This provides calibrated IRI labels that are impossible to obtain from field surveys without expensive profilometry equipment, and decouples the training data from any single vehicle's suspension characteristics.

3. **Dual-input deep neural network with a physically-motivated architecture.** We propose a 11,853-parameter model that separates spatial feature extraction (depthwise + standard 1D CNN) from macroscopic contextual processing (MLP), fuses them at a learned representation layer, and is trained under a composite Huber/Log-Cosh loss function augmented with inverse class frequency weighting for severe anomaly amplification and a false-alarm suppression penalty for dynamic-event robustness.

4. **Temporally adaptive crowdsourced road quality map.** We implement an exponential time-decay aggregation model in which every observation contributes to a road segment's quality score with a weight that diminishes with age ($w \propto e^{-t/\tau}$, $\tau = 24$ h). This ensures that repaired road defects are automatically and continuously down-weighted without administrator intervention—a property we term **self-healing map consistency.**

5. **Sub-second real-time broadcast via Geohash WebSocket namespacing.** The distributed backend routes road quality events to connected clients using precision-6 Geohash strings as Socket.IO room names, achieving O(R) broadcast complexity (where R is the number of clients in a geographic cell ≈ 1.2 km × 0.61 km) without iterating over all connected sockets.

6. **Infrastructure-aware route scoring with semantic cache freshness.** A routing service fetches Mapbox candidate routes, scores each by distance-weighted road quality, and recommends the smoothest path. Route cache validity is determined not by time-to-live alone, but by comparing the current IRI scores of the route's key segments against their cached snapshots—invalidating the cache whenever a significant road quality change is detected.

### E. Paper Organisation

The remainder of this paper is organised as follows. Section II surveys the related work in smartphone-based road monitoring, crowdsourcing architectures, and infrastructure-aware routing. Section III details the complete methodology and system architecture, encompassing simulation-based data acquisition, signal processing, deep learning design, and the distributed backend. Section IV presents experimental results and performance evaluation against the state of the art. Section V discusses limitations and future directions, including the Pothole Classification Model currently under development. Section VI concludes.

---

*This paper is structured such that Sections III onward are self-contained for readers primarily interested in the technical pipeline. Readers primarily interested in the system-level design may proceed directly to Section III-D.*
