export interface SensorReading {
  ax: number; ay: number; az: number;
  wx: number; wy: number; wz: number;
  speed:     number;
  timestamp: number;
  location:  { latitude: number; longitude: number };
}

const WINDOW_SIZE = 20; // 20 samples @ 10 Hz = 2 seconds

class WindowManager {
  private buffer: SensorReading[] = [];
  private cb: ((w: SensorReading[]) => void) | null = null;

  init(onWindowReady: (w: SensorReading[]) => void) {
    this.cb     = onWindowReady;
    this.buffer = [];
  }

  add(reading: SensorReading) {
    this.buffer.push(reading);
    if (this.buffer.length >= WINDOW_SIZE) {
      const win = [...this.buffer];
      this.buffer = [];
      this.cb?.(win);
    }
  }

  reset() { this.buffer = []; }
}

export const windowManager = new WindowManager();
