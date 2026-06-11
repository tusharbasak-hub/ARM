package com.example.better_roads.service

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.better_roads.api.ObservationRequest
import com.example.better_roads.api.RoadApiService
import com.example.better_roads.data.RoadDatabase
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.text.SimpleDateFormat
import java.util.*

class SyncWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {

    private val db = RoadDatabase.getDatabase(context)
    private val api = Retrofit.Builder()
        .baseUrl("http://10.0.2.2:5000/api/") // Adjust for local testing (10.0.2.2 is host for emulator)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
        .create(RoadApiService::class.java)

    override suspend fun doWork(): Result {
        val unsynced = db.roadPointDao().getUnsyncedPoints()
        if (unsynced.isEmpty()) return Result.success()

        val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
        sdf.timeZone = TimeZone.getTimeZone("UTC")

        val syncedIds = mutableListOf<Long>()
        for (point in unsynced) {
            try {
                val request = ObservationRequest(
                    latitude = point.latitude,
                    longitude = point.longitude,
                    roadQuality = point.qualityLevel,
                    speed = 0f, // Need to store speed in DB if important
                    timestamp = sdf.format(Date(point.timestamp))
                )
                val response = api.submitObservation(request)
                if (response.isSuccessful) {
                    syncedIds.add(point.id)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        if (syncedIds.isNotEmpty()) {
            db.roadPointDao().markAsSynced(syncedIds)
        }

        return if (syncedIds.size == unsynced.size) Result.success() else Result.retry()
    }
}
