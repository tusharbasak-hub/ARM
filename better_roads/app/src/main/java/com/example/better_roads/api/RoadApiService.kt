package com.example.better_roads.api

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST
import retrofit2.http.GET
import retrofit2.http.Query

data class ObservationRequest(
    val latitude: Double,
    val longitude: Double,
    val roadQuality: Int,
    val speed: Float,
    val timestamp: String,
    val deviceMetadata: Map<String, String> = mapOf("platform" to "android")
)

data class ObservationResponse(
    val success: Boolean,
    val message: String
)

data class RoadSegmentsResponse(
    val success: Boolean,
    val data: RoadSegmentsData
)

data class RoadSegmentsData(
    val roadSegments: List<RoadSegment>
)

data class RoadSegment(
    val roadSegmentId: String,
    val geometry: Geometry,
    val aggregatedQualityScore: Double
)

data class Geometry(
    val type: String,
    val coordinates: List<List<Double>>
)

interface RoadApiService {
    @POST("observations")
    suspend fun submitObservation(@Body request: ObservationRequest): Response<ObservationResponse>

    @GET("roads/nearby")
    suspend fun getNearbyRoads(
        @Query("lat") lat: Double,
        @Query("lng") lng: Double,
        @Query("radius") radius: Int = 5000
    ): Response<RoadSegmentsResponse>
}
