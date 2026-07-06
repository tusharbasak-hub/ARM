import { loadTensorflowModel } from 'react-native-fast-tflite';
import { ENV } from '../config/env';

// Math helpers
const mean = (data: number[]) => data.reduce((a, b) => a + b, 0) / data.length;

const std = (data: number[], m?: number) => {
    const mu = m !== undefined ? m : mean(data);
    const sumSqDiff = data.reduce((a, b) => a + Math.pow(b - mu, 2), 0);
    return Math.sqrt(sumSqDiff / data.length);
};

const rms = (data: number[]) => {
    const sumSq = data.reduce((a, b) => a + Math.pow(b, 2), 0);
    return Math.sqrt(sumSq / data.length);
};

// 1D Linear Interpolation helper
function interpolate1D(x: number[], y: number[], xGrid: number[]): number[] {
    const yGrid: number[] = [];
    let i = 0;
    for (let j = 0; j < xGrid.length; j++) {
        const target = xGrid[j];
        // Find index i in x such that x[i] <= target <= x[i+1]
        while (i < x.length - 2 && x[i + 1] < target) {
            i++;
        }
        const x0 = x[i];
        const x1 = x[i + 1];
        const y0 = y[i];
        const y1 = y[i + 1];
        if (x1 === x0) {
            yGrid.push(y0);
        } else {
            yGrid.push(y0 + ((y1 - y0) * (target - x0)) / (x1 - x0));
        }
    }
    return yGrid;
}

class MLService {
    private iriModel: any = null;
    private potholeModel: any = null;
    private isReady: boolean = false;

    // Hardcoded Isotonic Calibration LUT (81 points matching simulated training distribution)
    private readonly LUT_X = [
        1.4086923599243164, 1.7440783977508545, 1.7578213214874268, 1.842975378036499,
        1.8473470211029053, 1.850132942199707, 1.8503165245056152, 2.0337424278259277,
        2.034315347671509, 2.14422869682312, 2.1448543071746826, 2.646420478820801,
        2.646545171737671, 2.782649517059326, 2.7841274738311768, 2.785505771636963,
        2.798475742340088, 2.7987818717956543, 3.4157867431640625, 3.419206142425537,
        3.4557509422302246, 3.457141399383545, 3.8421335220336914, 3.8446102142333984,
        4.201412200927734, 4.201869487762451, 4.53278112411499, 4.538265705108643,
        4.707102298736572, 4.707160472869873, 4.888235569000244, 4.895236492156982,
        5.206006050109863, 5.2062087059021, 5.209521770477295, 5.21237325668335,
        5.400198459625244, 5.400581359863281, 5.99685525894165, 5.998824119567871,
        6.003285884857178, 6.006974220275879, 6.182613849639893, 6.18587589263916,
        6.206119060516357, 6.209420204162598, 6.368969440460205, 6.37297248840332,
        6.800901412963867, 6.8156585693359375, 6.837174892425537, 6.839789390563965,
        7.005046844482422, 7.009982109069824, 7.093731880187988, 7.097972869873047,
        7.420625686645508, 7.423173904418945, 8.234336853027344, 8.238245010375977,
        8.259381294250488, 8.262492179870605, 8.405502319335938, 8.408745765686035,
        9.010956764221191, 9.04541015625, 9.266136169433594, 9.269000053405762,
        9.646008491516113, 9.695860862731934, 10.979141235351562, 11.015422821044922,
        15.754457473754883, 15.78569221496582, 20.249418258666992, 20.576433181762695,
        21.86021614074707, 22.171323776245117, 23.77276611328125, 27.686208724975586,
        38.985233306884766
    ];

    private readonly LUT_Y = [
        1.2128446102142334, 1.2128446102142334, 1.3427366018295288, 1.3427366018295288,
        1.349787950515747, 1.349787950515747, 1.3977473974227905, 1.3977473974227905,
        1.8458579778671265, 1.8458579778671265, 1.8902190923690796, 1.8902190923690796,
        2.1944472789764404, 2.1944472789764404, 2.299156904220581, 2.5858237743377686,
        2.5858237743377686, 2.5971808433532715, 2.5971808433532715, 2.605844736099243,
        2.605844736099243, 3.253041982650757, 3.253041982650757, 3.437206983566284,
        3.437206983566284, 3.7107958793640137, 3.7107958793640137, 4.061951637268066,
        4.061951637268066, 4.340411186218262, 4.340411186218262, 4.519742012023926,
        4.519742012023926, 4.627760410308838, 4.627760410308838, 4.728625297546387,
        4.728625297546387, 5.007259368896484, 5.007259368896484, 5.308963298797607,
        5.308963298797607, 5.342557430267334, 5.342557430267334, 5.394349575042725,
        5.394349575042725, 5.399829387664795, 5.399829387664795, 5.556457042694092,
        5.556457042694092, 6.468915462493897, 6.468915462493897, 6.580991744995117,
        6.580991744995117, 6.958472728729248, 6.958472728729248, 7.956695079803467,
        7.956695079803467, 8.10103702545166, 8.10103702545166, 8.519607543945312,
        8.519607543945312, 9.348237991333008, 9.348237991333008, 9.480485916137695,
        9.480485916137695, 10.133979797363281, 10.133979797363281, 11.378344535827637,
        11.378344535827637, 12.98142147064209, 12.98142147064209, 13.616571426391602,
        13.616571426391602, 13.738725662231445, 13.738725662231445, 14.436870574951172,
        14.436870574951172, 15.114957809448242, 15.114957809448242, 15.571969032287598,
        15.571969032287598
    ];

    // Hardcoded context normalization parameters from context_scaler.pkl for the pothole classifier
    private readonly POTHOLE_CONTEXT_MEAN = [0.0, 12.464910954632682, 0.9999906116945316, 4.2572717833764555];
    private readonly POTHOLE_CONTEXT_SCALE = [1.0, 6.405128570025835, 6.002955638863807e-06, 0.8918762876362021];

    async initialize() {
        try {
            console.log('Initializing Dual-Model ML Service...');

            // Load both TFLite models
            this.iriModel = await loadTensorflowModel(ENV.ML.IRI_MODEL_FILE);
            this.potholeModel = await loadTensorflowModel(ENV.ML.POTHOLE_MODEL_FILE);

            this.isReady = true;
            console.log('Both ML Models loaded successfully');
        } catch (error) {
            console.error('Failed to load ML models:', error);
        }
    }

    getReadyStatus() {
        return this.isReady;
    }

    /**
     * Pipeline 1: Continuous IRI Regression (Every 100m)
     */
    async estimateIri(windowData: any[]): Promise<number | null> {
        if (!this.isReady || !this.iriModel) {
            console.warn('IRI Model not ready');
            return null;
        }

        try {
            const N = windowData.length;
            if (N < 2) return null;

            // 1. Extract raw timestamp, speeds, and IMU columns
            const times = windowData.map(s => s.timestamp);
            const speeds = windowData.map(s => s.speed || 0);

            // Compute distance array d
            const d: number[] = [0.0];
            for (let i = 1; i < N; i++) {
                const dt = (times[i] - times[i - 1]) / 1000.0; // in seconds
                const speed = speeds[i];
                d.push(d[i - 1] + speed * dt);
            }

            // Target uniform spatial grid of 400 steps over 100m (0.00, 0.25, ..., 99.75)
            const xGrid = Array.from({ length: 400 }, (_, idx) => idx * 0.25);

            // Resample each axis
            const axResampled = interpolate1D(d, windowData.map(s => s.ax || 0), xGrid);
            const ayResampled = interpolate1D(d, windowData.map(s => s.ay || 0), xGrid);
            const azResampled = interpolate1D(d, windowData.map(s => s.az || 0), xGrid);
            const wxResampled = interpolate1D(d, windowData.map(s => s.wx || 0), xGrid);
            const wyResampled = interpolate1D(d, windowData.map(s => s.wy || 0), xGrid);
            const wzResampled = interpolate1D(d, windowData.map(s => s.wz || 0), xGrid);
            const speedResampled = interpolate1D(d, speeds, xGrid);

            // 2. Physics-Informed Speed Normalization for Vertical Accel (az)
            // Reference speed: 80 km/h = 22.22 m/s
            const normAz = new Float32Array(400);
            for (let k = 0; k < 400; k++) {
                const vSafe = Math.max(speedResampled[k], 5.0);
                const speedScale = Math.pow(22.22 / vSafe, 2);
                normAz[k] = azResampled[k] * speedScale;
            }

            // Populate Raw IMU Input Buffer [1, 400, 6] (Float32Array: 2400 floats)
            const rawImuBuffer = new Float32Array(1 * 400 * 6);
            for (let k = 0; k < 400; k++) {
                const base = k * 6;
                rawImuBuffer[base + 0] = axResampled[k]; // ax
                rawImuBuffer[base + 1] = ayResampled[k]; // ay
                rawImuBuffer[base + 2] = normAz[k];      // az (speed-normalized)
                rawImuBuffer[base + 3] = wxResampled[k]; // wx
                rawImuBuffer[base + 4] = wyResampled[k]; // wy
                rawImuBuffer[base + 5] = wzResampled[k]; // wz
            }

            // 3. Context Features extraction [1, 13]
            const speedMean = mean(speedResampled);
            const speedStd = std(speedResampled, speedMean);
            const rmsAz = rms(Array.from(normAz)) + 1e-6;
            const rmsAy = rms(ayResampled) + 1e-6;
            const varAz = Math.pow(std(Array.from(normAz)), 2);

            const absAz = Array.from(normAz).map(Math.abs);
            const maxAbsAz = Math.max(...absAz);
            const crestFactorAz = maxAbsAz / rmsAz;

            // Zero crossing / Mean crossing rate (MCR)
            let zeroCrossings = 0;
            for (let k = 1; k < 400; k++) {
                if ((normAz[k] >= 0 && normAz[k - 1] < 0) || (normAz[k] < 0 && normAz[k - 1] >= 0)) {
                    zeroCrossings++;
                }
            }
            const mcrAz = zeroCrossings / 400.0;
            const p2pAz = Math.max(...Array.from(normAz)) - Math.min(...Array.from(normAz));

            const rmsWz = rms(wzResampled) + 1e-6;
            const rmsWy = rms(wyResampled) + 1e-6;
            const meanAbsAx = mean(axResampled.map(Math.abs));

            // Spectral energy ratios (approximate based on speed as in Swift/Kotlin guidelines)
            const energyRatio1to4 = speedMean > 0 ? 0.35 : 0.0;
            const energyRatio4to15 = speedMean > 0 ? 0.45 : 0.0;

            const contextStats = new Float32Array([
                speedMean,
                speedStd,
                rmsAz,
                rmsAy,
                varAz,
                crestFactorAz,
                mcrAz,
                p2pAz,
                rmsWz,
                rmsWy,
                meanAbsAx,
                energyRatio1to4,
                energyRatio4to15
            ]);

            // 4. Run IRI regressor inference
            const results = await this.iriModel.run([rawImuBuffer, contextStats]);

            if (results && results[0]) {
                const predictedLogIri = results[0][0];

                // 5. Inverse Exponential Transformation
                const rawIri = Math.exp(predictedLogIri) - 1.0;

                // 6. Piecewise Linear Isotonic Calibration LUT
                if (rawIri <= this.LUT_X[0]) return this.LUT_Y[0];
                if (rawIri >= this.LUT_X[this.LUT_X.length - 1]) return this.LUT_Y[this.LUT_Y.length - 1];

                for (let i = 0; i < this.LUT_X.length - 1; i++) {
                    if (rawIri >= this.LUT_X[i] && rawIri <= this.LUT_X[i + 1]) {
                        const slope = (this.LUT_Y[i + 1] - this.LUT_Y[i]) / (this.LUT_X[i + 1] - this.LUT_X[i]);
                        return this.LUT_Y[i] + slope * (rawIri - this.LUT_X[i]);
                    }
                }
                return rawIri;
            }

            return null;
        } catch (error) {
            console.error('IRI prediction failed:', error);
            return null;
        }
    }

    /**
     * Pipeline 2: Pothole / Road Quality Classification (Sliding Window of 128 samples)
     */
    async classifyRoad(rawAzBuffer: number[], speeds: number[], vehicleId: number): Promise<number | null> {
        if (!this.isReady || !this.potholeModel) {
            console.warn('Pothole model not ready');
            return null;
        }

        try {
            const N = rawAzBuffer.length;
            if (N !== 128) {
                console.warn(`Pothole classifier requires exactly 128 samples, got ${N}`);
                return null;
            }

            // 1. Jerk approximation (1st-order difference)
            const sig = new Float32Array(128);
            sig[0] = 0.0;
            for (let i = 1; i < 128; i++) {
                sig[i] = rawAzBuffer[i] - rawAzBuffer[i - 1];
            }

            // 2. Z-Score Standardization
            const meanSig = mean(Array.from(sig));
            const stdSig = std(Array.from(sig), meanSig) + 1e-6;
            const sigNorm = new Float32Array(128);
            for (let i = 0; i < 128; i++) {
                sigNorm[i] = (sig[i] - meanSig) / stdSig;
            }

            // 3. Populate Vibration Input Buffer [1, 2, 128]
            const vibrationBuffer = new Float32Array(256);
            // Channel 0: Rectified Magnitude
            for (let i = 0; i < 128; i++) {
                vibrationBuffer[i] = Math.abs(sigNorm[i]);
            }
            // Channel 1: Numerical Gradient (Central Difference)
            vibrationBuffer[128] = sigNorm[1] - sigNorm[0];
            for (let i = 1; i < 127; i++) {
                vibrationBuffer[128 + i] = (sigNorm[i + 1] - sigNorm[i - 1]) / 2.0;
            }
            vibrationBuffer[255] = sigNorm[127] - sigNorm[126];

            // 4. Physical features & Context scaling
            const rmsVal = rms(Array.from(sigNorm)) + 1e-6;
            const maxAbs = Math.max(...Array.from(sigNorm).map(Math.abs));
            const crestFactor = maxAbs / rmsVal;
            const speedMs = mean(speeds);

            const rawContext = [
                vehicleId, // SUV (0), Hatchback (1), Sedan (2)
                speedMs,
                rmsVal,
                crestFactor
            ];

            const scaledContext = new Float32Array(4);
            for (let j = 0; j < 4; j++) {
                scaledContext[j] = (rawContext[j] - this.POTHOLE_CONTEXT_MEAN[j]) / this.POTHOLE_CONTEXT_SCALE[j];
            }

            // 5. Run classification model
            const results = await this.potholeModel.run([vibrationBuffer, scaledContext]);

            if (results && results[0]) {
                const logits = results[0];

                // Softmax
                const maxLogit = Math.max(...logits);
                const exps = logits.map((val: number) => Math.exp(val - maxLogit));
                const sumExps = exps.reduce((a: number, b: number) => a + b, 0);
                const probs = exps.map((val: number) => val / sumExps);

                // Argmax
                let predClass = 0;
                let maxProb = probs[0];
                for (let i = 1; i < 4; i++) {
                    if (probs[i] > maxProb) {
                        maxProb = probs[i];
                        predClass = i;
                    }
                }

                // 6. Safety Gate Confidence Thresholding
                if (predClass >= 2 && maxProb < 0.82) {
                    predClass = 0; // Downgrade low-confidence pothole alarm to Excellent
                }

                return predClass;
            }

            return null;
        } catch (error) {
            console.error('Pothole classification failed:', error);
            return null;
        }
    }
}

export const mlService = new MLService();
