package com.example.better_roads.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [RoadPoint::class], version = 1, exportSchema = false)
abstract class RoadDatabase : RoomDatabase() {
    abstract fun roadPointDao(): RoadPointDao

    companion object {
        @Volatile
        private var INSTANCE: RoadDatabase? = null

        fun getDatabase(context: Context): RoadDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    RoadDatabase::class.java,
                    "road_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
