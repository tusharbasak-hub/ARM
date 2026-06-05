// ==========================================
// FILE: Backend/src/services/aggregation.js
// ==========================================

/**
 * Aggregation Service — Advanced Exponential Decay & Crowd Consensus
 *
 * This service is the single source of truth for:
 * 1. Exponentially decayed rolling IRI average per RoadSegment
 * 2. Derived colour category (iriCategory)
 * 3. "segment-polyline-update" Socket.IO broadcast
 *
 * ── Design Principles ─────────────────────────────────────────────────────
 *
 * • EXPONENTIAL TIME DECAY (Self-Healing Map)
 * Observations age out over time. Formula: w = e^(-age / tau)
 * tau is 24 hours. Data older than 7 days is discarded entirely.
 * This ensures a newly repaired road instantly overrides old bad data.
 *
 * • CROWD CONSENSUS (1 User = 1 Vote)
 * Observations are first grouped by deviceId before averaging for the segment.
 * Prevents a single user from skewing the map by driving back and forth.
 *
 * • POTHOLES DO NOT AFFECT POLYLINE COLOUR
 * Segment colour is IRI-pure (general road wear).
 */

'use strict';

const mongoose = require('mongoose');
const Observation = require('../models/Observation');
const RoadSegment = require('../models/RoadSegment');
const { IRI_CATEGORY, IRI_THRESHOLD_GOOD, IRI_THRESHOLD_MODERATE } = require('../models/RoadSegment');
const { getIO } = require('../socket');

// ── Constants for Decay Math ──────────────────────────────────────────────
const TAU_MS = 24 * 60 * 60 * 1000;              // 24 Hours
const MAX_DATA_AGE_MS = 7 * 24 * 60 * 60 * 1000; // 7 Days cut-off

// ── IRI category derivation ────────────────────────────────────────────────

function deriveCategory(avg) {
  if (avg < IRI_THRESHOLD_GOOD)     return IRI_CATEGORY.GOOD;     // 'green'
  if (avg < IRI_THRESHOLD_MODERATE) return IRI_CATEGORY.MODERATE; // 'yellow'
  return IRI_CATEGORY.BAD;                                        // 'orange'
}

// ── Deduplication lock: prevent thundering-herd on the same segment ────────
const _inFlight = new Set();

// ── Core update function ───────────────────────────────────────────────────

/**
 * Recalculate IRI stats for a RoadSegment using Time Decay & Consensus, and broadcast.
 */
async function updateSegment(segmentId) {
  const key = String(segmentId);

  if (_inFlight.has(key)) return;
  _inFlight.add(key);

  try {
    const objectId = typeof segmentId === 'string'
      ? new mongoose.Types.ObjectId(segmentId)
      : segmentId;

    const cutoffDate = new Date(Date.now() - MAX_DATA_AGE_MS);

    // ── 1. Aggregate IRI stats from the DB (Decay + Consensus) ─────────────
    const [result] = await Observation.aggregate([
      {
        // Step A: Filter by segment AND discard dead data (older than 7 days)
        $match: {
          roadSegmentId: objectId,
          recordedAt: { $gte: cutoffDate }
        },
      },
      {
        // Step B: Calculate Age in milliseconds
        $addFields: {
          ageMs: { $subtract: [new Date(), '$recordedAt'] }
        }
      },
      {
        // Step C: Calculate Exponential Weight: w = e^(-ageMs / TAU_MS)
        $addFields: {
          weight: { 
            $exp: { $divide: [ { $multiply: [-1, '$ageMs'] }, TAU_MS ] } 
          }
        }
      },
      {
        // Step D: Calculate Weighted IRI
        $addFields: {
          weightedIri: { $multiply: ['$iriScore', '$weight'] }
        }
      },
      {
        // Step E: Group by User (deviceId) - 1 User = 1 Average Vote
        $group: {
          _id: { 
            deviceId: { $ifNull: ['$deviceId', '$_id'] }, 
            segmentId: '$roadSegmentId' 
          },
          userWeightedIriSum: { $sum: '$weightedIri' },
          userWeightSum:      { $sum: '$weight' }
        },
      },
      {
        // Step F: Group by Segment - Sum up all Unique Users' votes
        $group: {
          _id: '$_id.segmentId',
          totalWeightedIri: { $sum: '$userWeightedIriSum' },
          totalWeight:      { $sum: '$userWeightSum' },
          uniqueUsers:      { $sum: 1 }
        },
      },
      {
        // Step G: Final Math (Total Weighted IRI / Total Weight)
        $project: {
          averageIri: { 
            $cond: [
              { $eq: ['$totalWeight', 0] }, 
              0, 
              { $divide: ['$totalWeightedIri', '$totalWeight'] }
            ] 
          },
          sampleCount: '$uniqueUsers'
        }
      }
    ]);

    if (!result) {
      return;
    }

    const { averageIri, sampleCount } = result;
    const iriCategory = deriveCategory(averageIri);

    // ── 2. Persist to the segment document & Reset pending queue ───────────
    const updatedSegment = await RoadSegment.findByIdAndUpdate(
      objectId,
      {
        $set: {
          iriStats: {
            sampleCount,
            rollingSum: 0, 
            average:     averageIri,
            pendingDevices: [], // Clear queue array for next milestone counting loop
            lastUpdated: new Date(),
          },
          iriCategory,
          lastObservationAt: new Date(),
        },
      },
      { new: true, lean: true }
    );

    if (!updatedSegment) {
      console.warn(`[aggregation] Segment ${key} not found during update`);
      return;
    }

    // ── 3. Broadcast polyline update via Socket.IO ─────────────────────
    const io = getIO();
    io.emit('segment-polyline-update', {
      roadSegmentId: key,
      iriCategory,
      averageIri:   Math.round(averageIri * 1000) / 1000, 
      sampleCount,  // Unique Users count
      polyline:     updatedSegment.polyline,
      name:         updatedSegment.name,
      updatedAt:    updatedSegment.lastObservationAt,
    });
  } catch (err) {
    console.error(`[aggregation] updateSegment(${key}) error:`, err.message);
    throw err; 
  } finally {
    _inFlight.delete(key);
  }
}

// ── Bulk recalculation (maintenance / startup use) ─────────────────────────

async function recalculateAllSegments() {
  let processed = 0;
  let errors    = 0;

  const segmentIds = await Observation.distinct('roadSegmentId');
  console.info(`[aggregation] Recalculating ${segmentIds.length} segments with exponential decay…`);

  for (const id of segmentIds) {
    try {
      // eslint-disable-next-line no-await-in-loop
      await updateSegment(id);
      processed += 1;
    } catch (err) {
      errors += 1;
      console.error(`[aggregation] Failed for segment ${id}:`, err.message);
    }
  }

  console.info(`[aggregation] Done. processed=${processed} errors=${errors}`);
  return { processed, errors };
}

module.exports = {
  updateSegment,
  recalculateAllSegments,
  deriveCategory, 
};