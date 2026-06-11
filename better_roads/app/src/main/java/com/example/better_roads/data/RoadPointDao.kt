package com.example.better_roads.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface RoadPointDao {
    @Insert
    suspend fun insert(point: RoadPoint)

    @Query("SELECT * FROM road_points ORDER BY timestamp DESC")
    fun getAllPoints(): Flow<List<RoadPoint>>

    @Query("SELECT * FROM road_points WHERE isSynced = 0")
    suspend fun getUnsyncedPoints(): List<RoadPoint>

    @Query("UPDATE road_points SET isSynced = 1 WHERE id IN (:ids)")
    suspend fun markAsSynced(ids: List<Long>)
}
