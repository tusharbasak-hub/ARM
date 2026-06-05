// ==========================================
// FILE: Backend/src/models/RoadSegment.js
// ==========================================

/**
 * RoadSegment Model — Hybrid Non-Breaking Architecture
 *
 * Represets a road stretch. Combines legacy visualization fields (geometry, regionId)
 * with the new high-performance Hybrid Milestone Aggregation tracking arrays.
 */

'use strict';

const mongoose = require('mongoose');

const { Schema } = mongoose;

const IRI_CATEGORY = Object.freeze({
  GOOD:     'green', 
  MODERATE: 'yellow',
  BAD:      'orange',
});

const IRI_THRESHOLD_GOOD     = 1.5;
const IRI_THRESHOLD_MODERATE = 2.5;

// ── GeoJSON LineString Schema ─────────────────────────────────────────────
const GeoLineStringSchema = new Schema(
  {
    type: {
      type:     String,
      enum:     ['LineString'],
      default:  'LineString',
      required: true,
    },
    coordinates: {
      type:     [[Number]], // [[lng, lat], [lng, lat], ...]
      required: true,
    },
  },
  { _id: false }
);

// ── Main Hybrid Schema ────────────────────────────────────────────────────
const RoadSegmentSchema = new Schema(
  {
    name: {
      type: String,
      trim: true,
    },

    // Legacy support for your route controllers/map queries
    roadSegmentId: {
      type:  String,
      index: true,
    },

    // Region ID used for bounding-box and geohash viewports
    regionId: {
      type:  String,
      required: true,
      index: true,
    },

    // Maps to your frontend map drawing layer
    geometry: {
      type:     GeoLineStringSchema,
      required: true,
      index:    '2dsphere',
    },

    // Dual-Compatibility Score Fields (No breakdown on endpoints)
    aggregatedQualityScore: {
      type:    Number,
      default: 1.0,
    },
    observationCount: {
      type:    Number,
      default: 0,
    },

    // ── New Aggregation Queue Parameters ──────────────────────────────────
    iriStats: {
      pendingDevices: { type: [String], default: [] }, // Unique user milestone array
      sampleCount:    { type: Number, default: 0 },    // Syncs with unique users count
      average:        { type: Number, default: 0 },    // Syncs with aggregatedQualityScore
      lastUpdated:    { type: Date,   default: null },
    },

    iriCategory: {
      type:    String,
      enum:    Object.values(IRI_CATEGORY),
      default: IRI_CATEGORY.GOOD,
    },

    centerPoint: {
      type: [Number], // [lng, lat] for map center viewport calculation
    },

    lastObservationAt: {
      type:  Date,
      index: true,
    },
  },
  {
    timestamps: true,
    versionKey: false,
  }
);

// ── Static helper ─────────────────────────────────────────────────────────
RoadSegmentSchema.statics.iriToCategory = function (averageIri) {
  if (averageIri < IRI_THRESHOLD_GOOD)     return IRI_CATEGORY.GOOD;
  if (averageIri < IRI_THRESHOLD_MODERATE) return IRI_CATEGORY.MODERATE;
  return IRI_CATEGORY.BAD;
};

// ── Pre-save trigger: sync values automatically to maintain compatibility ─
RoadSegmentSchema.pre('save', function (next) {
  // Database fields fallback synchronisation strategy
  this.aggregatedQualityScore = this.iriStats.average;
  this.observationCount = this.iriStats.sampleCount;
  
  // Recalculate color string
  this.iriCategory = this.constructor.iriToCategory(this.iriStats.average);
  next();
});

const RoadSegment = mongoose.model('RoadSegment', RoadSegmentSchema);

module.exports = RoadSegment;
module.exports.IRI_CATEGORY           = IRI_CATEGORY;
module.exports.IRI_THRESHOLD_GOOD     = IRI_THRESHOLD_GOOD;
module.exports.IRI_THRESHOLD_MODERATE = IRI_THRESHOLD_MODERATE;