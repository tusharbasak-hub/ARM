package com.example.better_roads.service

import android.app.*
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.Build
import android.os.IBinder
import android.os.Looper
import androidx.core.app.NotificationCompat
import com.example.better_roads.MainActivity
import com.example.better_roads.ml.RoadQualityModel
import com.google.android.gms.location.*
import java.util.concurrent.TimeUnit
import kotlin.math.*

class RoadQualityService : Service(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationCallback: LocationCallback

    private lateinit var model: RoadQualityModel
    
    // Data Buffers
    private val sensorData = mutableListOf<SensorSample>()
    private var lastLocation: Location? = null
    private var cumulativeDistance = 0f
    private var currentSpeedMs = 0f

    private val CHANNEL_ID = "RoadQualityServiceChannel"
    private val NOTIFICATION_ID = 1

    data class SensorSample(
        val timestamp: Long,
        val ax: Float, val ay: Float, val az: Float,
        val wx: Float, val wy: Float, val wz: Float,
        val speedMs: Float,
        val distance: Float
    )

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, createNotification())

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        
        model = RoadQualityModel(this)

        startSensors()
        startLocationUpdates()
    }

    private fun startSensors() {
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
    }

    private fun startLocationUpdates() {
        val locationRequest = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000)
            .setMinUpdateIntervalMillis(500)
            .build()

        locationCallback = object : LocationCallback() {
            override fun onLocationResult(locationResult: LocationResult) {
                for (location in locationResult.locations) {
                    onNewLocation(location)
                }
            }
        }

        try {
            fusedLocationClient.requestLocationUpdates(locationRequest, locationCallback, Looper.getMainLooper())
        } catch (e: SecurityException) {
            e.printStackTrace()
        }
    }

    private var currentAx = 0f
    private var currentAy = 0f
    private var currentAz = 0f
    private var currentWx = 0f
    private var currentWy = 0f
    private var currentWz = 0f

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            // Mapping: X=Lateral, Y=Longitudinal, Z=Vertical
            // Standard Android: X=Lateral (left-right), Y=Longitudinal (up-down on screen), Z=Vertical (out of screen)
            // Assuming phone is portrait in holder:
            currentAx = event.values[0]
            currentAy = event.values[1]
            currentAz = event.values[2]
        } else if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            currentWx = event.values[0]
            currentWy = event.values[1]
            currentWz = event.values[2]
        }

        val currentTime = System.currentTimeMillis()
        sensorData.add(SensorSample(currentTime, currentAx, currentAy, currentAz, currentWx, currentWy, currentWz, currentSpeedMs, cumulativeDistance))
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun onNewLocation(location: Location) {
        if (lastLocation != null) {
            val dx = location.distanceTo(lastLocation!!)
            cumulativeDistance += dx
        }
        lastLocation = location
        currentSpeedMs = location.speed

        if (cumulativeDistance >= 5.0f) {
            processWindow()
        }
    }

    private fun processWindow() {
        val windowData = sensorData.filter { it.distance <= 5.0f }.toList()
        if (windowData.size < 10) return // Not enough data

        // 1. Interpolate to 100 points
        val resampled = resample(windowData, 100)

        // 2. Prepare context stats
        val contextStats = prepareContextStats(resampled)

        // 3. Inference
        val roadQuality = model.predict(resampled, contextStats)

        // 4. Save to DB and notify listeners (TBD)
        onRoadQualityPredicted(roadQuality, lastLocation)

        // Clean up buffer
        sensorData.removeIf { it.distance <= 5.0f }
        // Adjust remaining distances
        for (i in sensorData.indices) {
            // This is a bit complex, simpler to reset cumulativeDistance and shift.
        }
        cumulativeDistance -= 5.0f
        for (i in sensorData.indices) {
            // Not quite right, need to fix distance logic
        }
        // Simplified: Clear all and start fresh for next 5m
        sensorData.clear()
        cumulativeDistance = 0f
    }

    private fun resample(data: List<SensorSample>, steps: Int): Array<FloatArray> {
        val result = Array(steps) { FloatArray(7) }
        val maxDist = 5.0f
        val stepSize = maxDist / (steps - 1)

        for (i in 0 until steps) {
            val targetDist = i * stepSize
            // Find samples to interpolate between
            val nextIdx = data.indexOfFirst { it.distance >= targetDist }.coerceAtLeast(1).coerceAtMost(data.size - 1)
            val prevIdx = nextIdx - 1
            
            val s1 = data[prevIdx]
            val s2 = data[nextIdx]
            
            val t = if (s2.distance != s1.distance) (targetDist - s1.distance) / (s2.distance - s1.distance) else 0f
            
            result[i][0] = s1.ax + t * (s2.ax - s1.ax)
            result[i][1] = s1.ay + t * (s2.ay - s1.ay)
            result[i][2] = s1.az + t * (s2.az - s1.az)
            result[i][3] = s1.wx + t * (s2.wx - s1.wx)
            result[i][4] = s1.wy + t * (s2.wy - s1.wy)
            result[i][5] = s1.wz + t * (s2.wz - s1.wz)
            result[i][6] = s1.speedMs + t * (s2.speedMs - s1.speedMs)
        }
        return result
    }

    private fun prepareContextStats(resampled: Array<FloatArray>): FloatArray {
        var sumSpeed = 0f
        var sumAz = 0f
        var sumAy = 0f
        var signChanges = 0
        
        for (i in resampled.indices) {
            sumSpeed += resampled[i][6]
            sumAz += resampled[i][2]
            sumAy += resampled[i][1]
            if (i > 0) {
                if (resampled[i][2].sign != resampled[i-1][2].sign) {
                    signChanges++
                }
            }
        }
        
        return floatArrayOf(
            sumSpeed / resampled.size,
            sumAz / resampled.size,
            sumAy / resampled.size,
            signChanges.toFloat() / resampled.size,
            0.0f, // veh_type_1
            0.0f, // veh_type_2
            1.0f  // veh_type_3 (default fallback)
        )
    }

    private fun onRoadQualityPredicted(level: Int, location: Location?) {
        // Broadcast or Save to DB
        val intent = Intent("com.example.better_roads.ROAD_QUALITY_UPDATE")
        intent.putExtra("level", level)
        intent.putExtra("lat", location?.latitude ?: 0.0)
        intent.putExtra("lng", location?.longitude ?: 0.0)
        sendBroadcast(intent)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val serviceChannel = NotificationChannel(
                CHANNEL_ID,
                "Road Quality Monitoring Service",
                NotificationManager.IMPORTANCE_DEFAULT
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(serviceChannel)
        }
    }

    private fun createNotification(): Notification {
        val notificationIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, notificationIntent,
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Road Quality Monitoring")
            .setContentText("Recording sensors and GPS...")
            .setSmallIcon(android.R.drawable.ic_menu_mylocation)
            .setContentIntent(pendingIntent)
            .build()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        fusedLocationClient.removeLocationUpdates(locationCallback)
        model.close()
    }
}
