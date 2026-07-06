import { mlService } from './mlService';

export interface SensorReading {
  ax: number; ay: number; az: number;
  wx: number; wy: number; wz: number;
  speed:     number;
  timestamp: number;
  location:  { latitude: number; longitude: number; heading: number };
}

class WindowManager {
  // IRI Regressor buffer (distance-based)
  private iriBuffer: SensorReading[] = [];
  private iriDistance = 0.0;
  private lastIriTimestamp: number | null = null;
  private onIriReadyCallback: ((iriScore: number, lastReading: SensorReading) => void) | null = null;

  // Pothole Classifier buffer (time-based)
  private potholeBuffer: SensorReading[] = [];
  private potholePredictionsQueue: number[] = [];
  private onPotholeReadyCallback: ((potholeClass: number, lastReading: SensorReading) => void) | null = null;
  
  private vehicleId = 0;
  private samplesSinceLastPotholeInference = 0;
  
  // Throttle interval for pothole model (run every 10 samples at 50Hz = every 200ms)
  private readonly POTHOLE_INFERENCE_STRIDE = 10;
  // Sliding queue size for majority vote (1.5 seconds at 10Hz would be 15, at 5Hz / 200ms it is 15 predictions = 3 seconds, which is perfect)
  private readonly MAJORITY_VOTE_QUEUE_SIZE = 15;

  init(
    onIriReady: (iriScore: number, lastReading: SensorReading) => void,
    onPotholeReady: (potholeClass: number, lastReading: SensorReading) => void,
    vehicleId: number
  ) {
    this.onIriReadyCallback = onIriReady;
    this.onPotholeReadyCallback = onPotholeReady;
    this.vehicleId = vehicleId;

    this.iriBuffer = [];
    this.iriDistance = 0.0;
    this.lastIriTimestamp = null;

    this.potholeBuffer = [];
    this.potholePredictionsQueue = [];
    this.samplesSinceLastPotholeInference = 0;
  }

  add(reading: SensorReading) {
    // 1. IRI Model Distance Tracking & Accumulation
    if (reading.speed >= 1.0) { // Stationary filtering (discard readings when speed < 1.0 m/s)
      if (this.lastIriTimestamp !== null) {
        const dt = (reading.timestamp - this.lastIriTimestamp) / 1000.0; // seconds
        const deltaDist = reading.speed * dt; // meters
        this.iriDistance += deltaDist;
      }
      this.iriBuffer.push(reading);
      this.lastIriTimestamp = reading.timestamp;

      // When cumulative distance reaches 100.0 meters, trigger IRI regression
      if (this.iriDistance >= 100.0) {
        const currentIriBuffer = [...this.iriBuffer];
        const lastReading = reading;
        
        // Reset IRI accumulator
        this.iriBuffer = [];
        this.iriDistance = 0.0;
        this.lastIriTimestamp = null;

        // Run IRI model asynchronously
        mlService.estimateIri(currentIriBuffer).then((iriScore) => {
          if (iriScore !== null && this.onIriReadyCallback) {
            this.onIriReadyCallback(iriScore, lastReading);
          }
        }).catch(err => console.error('[IRI Inference Error]:', err));
      }
    } else {
      // Vehicle is stationary, pause distance accumulation
      this.lastIriTimestamp = null;
    }

    // 2. Pothole Model Rolling Buffer & Inference
    this.potholeBuffer.push(reading);
    if (this.potholeBuffer.length > 128) {
      this.potholeBuffer.shift(); // maintain sliding window of exactly 128 samples
    }

    if (this.potholeBuffer.length === 128) {
      this.samplesSinceLastPotholeInference++;
      
      // Run pothole inference every POTHOLE_INFERENCE_STRIDE samples (200ms)
      if (this.samplesSinceLastPotholeInference >= this.POTHOLE_INFERENCE_STRIDE) {
        this.samplesSinceLastPotholeInference = 0;

        const currentPotholeBuffer = [...this.potholeBuffer];
        const rawAz = currentPotholeBuffer.map(s => s.az);
        const speeds = currentPotholeBuffer.map(s => s.speed);
        const lastReading = reading;

        mlService.classifyRoad(rawAz, speeds, this.vehicleId).then((predClass) => {
          if (predClass !== null) {
            // Push to sliding queue for temporal smoothing majority vote
            this.potholePredictionsQueue.push(predClass);
            if (this.potholePredictionsQueue.length > this.MAJORITY_VOTE_QUEUE_SIZE) {
                this.potholePredictionsQueue.shift();
            }

            // Calculate majority vote
            const smoothedClass = this.getMajorityVote(this.potholePredictionsQueue);
            
            if (this.onPotholeReadyCallback) {
                this.onPotholeReadyCallback(smoothedClass, lastReading);
            }
          }
        }).catch(err => console.error('[Pothole Inference Error]:', err));
      }
    }
  }

  private getMajorityVote(queue: number[]): number {
    if (queue.length === 0) return 0;
    
    const counts: Record<number, number> = {};
    let maxClass = queue[0];
    let maxCount = 0;
    
    queue.forEach((val) => {
      counts[val] = (counts[val] || 0) + 1;
      if (counts[val] > maxCount) {
        maxCount = counts[val];
        maxClass = val;
      }
    });
    
    return maxClass;
  }

  reset() {
    this.iriBuffer = [];
    this.iriDistance = 0.0;
    this.lastIriTimestamp = null;
    this.potholeBuffer = [];
    this.potholePredictionsQueue = [];
    this.samplesSinceLastPotholeInference = 0;
  }
}

export const windowManager = new WindowManager();
