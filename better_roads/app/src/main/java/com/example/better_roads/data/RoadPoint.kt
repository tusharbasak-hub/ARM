package com.example.better_roads.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "road_points")
data class RoadPoint(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val latitude: Double,
    val longitude: Double,
    val qualityLevel: Int,
    val timestamp: Long,
    val isSynced: Boolean = false
)
