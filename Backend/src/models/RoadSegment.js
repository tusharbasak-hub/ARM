const mongoose = require("mongoose");

const roadSegmentSchema = new mongoose.Schema({
  roadSegmentId: {
    type: String,
    required: true,
    unique: true
  },

  geometry: {
    type: {
      type: String,
      enum: ["LineString"],
      default: "LineString"
    },
    coordinates: {
      type: [[Number]],
      required: true
    }
  },

  centerPoint: {
    type: {
      type: String,
      enum: ["Point"],
      default: "Point"
    },
    coordinates: {
      type: [Number] // [lng, lat]
    }
  },

  aggregatedQualityScore: {
    type: Number,
    min: 0,
    max: 3,
    default: null
  },

  // FIX B: Add fields used by aggregationService
  confidenceScore: {
    type: Number,
    min: 0,
    max: 1,
    default: 0
  },

  qualityDistribution: {
    excellent: { type: Number, default: 0 },
    good: { type: Number, default: 0 },
    bad: { type: Number, default: 0 },
    worst: { type: Number, default: 0 }
  },

  observationCount: {
    type: Number,
    default: 0
  },

  // Patch-based length tracking (in meters)
  len1: {
    type: Number,
    default: 0,
    min: 0
  },

  len2: {
    type: Number,
    default: 0,
    min: 0
  },

  len3: {
    type: Number,
    default: 0,
    min: 0
  },

  segmentLength: {
    type: Number,
    default: 100,
    min: 1
  },

  regionId: {
    type: String,
    required: true,
    index: true
  },

  lastUpdated: Date

}, { timestamps: true });

// centerPoint 2dsphere index (regionId already indexed in schema field)
roadSegmentSchema.index({ centerPoint: "2dsphere" });

module.exports = mongoose.model("RoadSegment", roadSegmentSchema);
