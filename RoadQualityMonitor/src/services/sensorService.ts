import { accelerometer, gyroscope, setUpdateIntervalForType, SensorTypes } from 'react-native-sensors';
import Geolocation from 'react-native-geolocation-service';
import { map } from 'rxjs/operators';
import { Platform, PermissionsAndroid } from 'react-native';

const SAMPLING_INTERVAL_MS = 100; // 10Hz

class SensorService {
    private accelSubscription: any = null;
    private gyroSubscription: any = null;
    
    private currentAccel = { x: 0, y: 0, z: 0 };
    private currentGyro = { x: 0, y: 0, z: 0 };
    private currentSpeed = 0;
    private currentLocation = { latitude: 0, longitude: 0, heading: -1 };

    private onReadingCallback: ((reading: any) => void) | null = null;
    private intervalId: any = null;
    private isCollecting = false;

    async requestPermissions() {
        if (Platform.OS === 'android') {
            await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION);
        }
    }

    startCollection(onReading: (reading: any) => void) {
        if (this.isCollecting) return;
        this.onReadingCallback = onReading;
        this.isCollecting = true;

        // Configure Sensors
        setUpdateIntervalForType(SensorTypes.accelerometer, SAMPLING_INTERVAL_MS);
        setUpdateIntervalForType(SensorTypes.gyroscope, SAMPLING_INTERVAL_MS);

        // Subscribe to Sensors
        this.accelSubscription = accelerometer.subscribe(({ x, y, z }) => {
            this.currentAccel = { x, y, z };
        });

        this.gyroSubscription = gyroscope.subscribe(({ x, y, z }) => {
            this.currentGyro = { x, y, z };
        });

        // Watch Location (for speed and coords)
        Geolocation.watchPosition(
            (position) => {
                this.currentLocation = {
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude,
                    heading: position.coords.heading ?? -1
                };
                this.currentSpeed = position.coords.speed || 0; // Speed in m/s
                if (this.currentSpeed < 0) this.currentSpeed = 0;
            },
            (error) => console.log(error),
            { enableHighAccuracy: true, distanceFilter: 0, interval: 1000, fastestInterval: 1000 }
        );

        // Start specific sampling loop to sync data readings
        // Although sensors emit independently, we want to sample specific snapshots at 10Hz
        this.intervalId = setInterval(() => {
            this.emitReading();
        }, SAMPLING_INTERVAL_MS);
    }

    emitReading() {
        if (!this.onReadingCallback) return;

        const reading = {
            ax: this.currentAccel.x,
            ay: this.currentAccel.y,
            az: this.currentAccel.z,
            wx: this.currentGyro.x,
            wy: this.currentGyro.y,
            wz: this.currentGyro.z,
            speed: this.currentSpeed,
            timestamp: Date.now(),
            location: { ...this.currentLocation }
        };

        this.onReadingCallback(reading);
    }

    stopCollection() {
        this.isCollecting = false;
        if (this.accelSubscription) this.accelSubscription.unsubscribe();
        if (this.gyroSubscription) this.gyroSubscription.unsubscribe();
        if (this.intervalId) clearInterval(this.intervalId);
        Geolocation.stopObserving();
    }
}

export const sensorService = new SensorService();
