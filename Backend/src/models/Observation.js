// ==========================================
// FILE: Backend/src/models/Observation.js
// ==========================================

/**
 * Observation Model — Dual-Model ML Pipeline
 *
 * Stores a single edge-device reading that contains outputs from two parallel
 * TensorFlow Lite models running on the mobile client:
 *
 *   1. Continuous IRI Model  → iriScore (Float, roughness in m/km)
 *   2. Pothole Classification → hasPothole (Boolean) + potholeConfidence (Float)
 *
 * Geospatial Design:
 *   • location  – GeoJSON Point for the exact GPS coordinate of this reading.
 *                 Indexed as a 2dsphere so MongoDB can run $near / $geoWithin
 *                 queries natively without any in-memory JS math.
 *
 * Filtering Rule (enforced at write-time by the controller, not here):
 *   A DB write is only triggered when potholeConfidence > 0.75 OR the
 *   reading belongs to a significant batch.  That gate lives in
 *   observationController.js; this model stores whatever passes it.
 */

'use strict';

const mongoose = require('mongoose');

const { Schema } = mongoose;

// ── Sub-schema: GeoJSON Point ──────────────────────────────────────────────
const GeoPointSchema = new Schema(
  {
    type: {
      type:    String,
      enum:    ['Point'],
      default: 'Point',
      required: true,
    },
    // [longitude, latitude]  ← GeoJSON standard order
    coordinates: {
      type:     [Number],
      required: true,
    },
  },
  { _id: false }
);

// ── Main Schema ────────────────────────────────────────────────────────────
const ObservationSchema = new Schema(
  {
    // ── Road-segment reference ───────────────────────────────────────────
    roadSegmentId: {
      type:     Schema.Types.ObjectId,
      ref:      'RoadSegment',
      required: true,
      index:    true,
    },

    // ── Device / session metadata ────────────────────────────────────────
    deviceId: {
      type:  String,
      index: true,
    },
    sessionId: {
      type:  String,
      index: true,
    },

    // ── Exact GPS coordinate (GeoJSON Point) ─────────────────────────────
    // Native 2dsphere index enables $near / $geoWithin without JS math.
    location: {
      type:     GeoPointSchema,
      required: true,
      index:    '2dsphere',
    },

    // Convenience flat-fields so controllers can read lat/lng directly
    // without destructuring the GeoJSON object every time.
    latitude: {
      type:     Number,
      required: true,
    },
    longitude: {
      type:     Number,
      required: true,
    },

    // ── Model 1 — Continuous IRI ─────────────────────────────────────────
    iriScore: {
      type:     Number,   // roughness in m/km; lower = smoother
      required: true,
      min:      0,
    },

    // ── Model 2 — Pothole Classification ────────────────────────────────
    hasPothole: {
      type:     Boolean,
      required: true,
      default:  false,
    },
    potholeConfidence: {
      type:    Number,    // 0.0 – 1.0
      default: 0,
      min:     0,
      max:     1,
    },

    // ── Derived map-marker type ──────────────────────────────────────────
    // Computed once at creation to avoid recalculation on every read.
    //   'pothole'  – hasPothole === true
    //   'perfect'  – hasPothole === false && iriScore < 1.0
    //   'normal'   – everything else (used internally; not broadcast)
    markerType: {
      type: String,
      enum: ['pothole', 'perfect', 'normal'],
    },

    // ── Timestamp ────────────────────────────────────────────────────────
    recordedAt: {
      type:    Date,
      default: Date.now,
      index:   true,
    },
  },
  {
    timestamps: true,   // adds createdAt / updatedAt
    versionKey: false,
  }
);

// ── Pre-save hook: derive markerType automatically ─────────────────────────
ObservationSchema.pre('save', function (next) {
  if (this.hasPothole) {
    this.markerType = 'pothole';
  } else if (this.iriScore < 1.0) {
    this.markerType = 'perfect';
  } else {
    this.markerType = 'normal';
  }
  next();
});

// ── Compound index for efficient "recent observations near a point" queries ─
ObservationSchema.index({ location: '2dsphere', recordedAt: -1 });

// ── Index for segment-level IRI aggregations ───────────────────────────────
ObservationSchema.index({ roadSegmentId: 1, recordedAt: -1 });

module.exports = mongoose.model('Observation', ObservationSchema);