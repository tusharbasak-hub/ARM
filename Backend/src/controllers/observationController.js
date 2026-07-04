// ==========================================
// FILE: Backend/src/controllers/observationController.js
// ==========================================

'use strict';

const Observation  = require('../models/Observation');
const RoadSegment  = require('../models/RoadSegment');
const aggregation  = require('../services/aggregation');
const { getIO }    = require('../socket');
const mapMatching  = require('../services/mapMatching');
const { getRegionId } = require('../utils/geohash');

// ── Constants ──────────────────────────────────────────────────────────────
const POTHOLE_CONFIDENCE_THRESHOLD = 0.75;
const CONSENSUS_MIN_USERS = 1;       // Testing/Single-User Mode (Production: 3)
const MILESTONE_THRESHOLD = 1;       // Testing/Single-User Mode (Production: 30)
const CONSENSUS_RADIUS_METERS = 10;
const CONSENSUS_TIME_WINDOW_MS = 24 * 60 * 60 * 1000;
const MATCH_MAX_DISTANCE_METERS = 80; // Max distance in meters to match observation to segment (optimized from 40m)

// ── Internal Helpers ───────────────────────────────────────────────────────

function passesConfidenceGate(hasPothole, potholeConfidence) {
  if (!hasPothole) return true;
  return potholeConfidence > POTHOLE_CONFIDENCE_THRESHOLD;
}

async function resolveSegment(lat, lng, roadSegmentId, sessionId, heading) {
  if (roadSegmentId) {
    const seg = await RoadSegment.findById(roadSegmentId).lean();
    if (seg) return seg;
  }

  const nearest = await RoadSegment.findOne({
    geometry: {
      $near: {
        $geometry: { type: 'Point', coordinates: [lng, lat] },
        $maxDistance: MATCH_MAX_DISTANCE_METERS,
      },
    },
  }).lean();

  if (nearest) return nearest;

  // Calculate travel direction
  let travelDirection = null;

  // 1. Try to compute from GPS heading if valid (heading >= 0)
  if (heading !== undefined && heading !== null && heading >= 0 && heading <= 360) {
    const headingRad = (heading * Math.PI) / 180;
    travelDirection = [Math.sin(headingRad), Math.cos(headingRad)];
  }

  // 2. Fallback to calculating from previous observation in same session
  if (!travelDirection && sessionId) {
    const prevObs = await Observation.findOne({ sessionId })
      .sort({ recordedAt: -1 })
      .lean();
    if (prevObs) {
      const dx = lng - prevObs.longitude;
      const dy = lat - prevObs.latitude;
      const dist = Math.hypot(dx, dy);
      if (dist > 1e-6) {
        travelDirection = [dx / dist, dy / dist];
      }
    }
  }

  // Dynamically match and create a new RoadSegment if none exists within 80m
  try {
    const matched = await mapMatching.matchPoint(lat, lng, travelDirection);
    if (!matched) return null;

    let seg = await RoadSegment.findOne({ roadSegmentId: matched.roadSegmentId }).lean();
    if (!seg) {
      const regionId = getRegionId(matched.matchedLatitude, matched.matchedLongitude);
      const newSeg = new RoadSegment({
        name: matched.roadName || 'Unknown Road',
        roadSegmentId: matched.roadSegmentId,
        regionId,
        geometry: {
          type: 'LineString',
          coordinates: [
            [
              matched.matchedLongitude - 0.00045 * (matched.direction ? matched.direction[0] : 1),
              matched.matchedLatitude - 0.00045 * (matched.direction ? matched.direction[1] : 0)
            ],
            [
              matched.matchedLongitude + 0.00045 * (matched.direction ? matched.direction[0] : 1),
              matched.matchedLatitude + 0.00045 * (matched.direction ? matched.direction[1] : 0)
            ]
          ]
        },
        centerPoint: [matched.matchedLongitude, matched.matchedLatitude],
        aggregatedQualityScore: 1.0,
        observationCount: 0,
        iriStats: {
          pendingDevices: [],
          sampleCount: 0,
          average: 1.0,
          lastUpdated: new Date(),
        }
      });
      const savedDoc = await newSeg.save();
      seg = savedDoc.toObject();

      // Broadcast the newly created segment via Socket.IO
      const io = getIO();
      io.emit('segment-polyline-update', {
        roadSegmentId: String(seg._id),
        iriCategory:   seg.iriCategory,
        averageIri:    seg.iriStats?.average ?? 0,
        sampleCount:   seg.iriStats?.sampleCount ?? 0,
        polyline:      seg.geometry,
        name:          seg.name,
        updatedAt:     seg.updatedAt,
      });
    }
    return seg;
  } catch (err) {
    console.error('[resolveSegment] Error dynamically creating road segment:', err.message);
    return null;
  }
}

async function saveObservation(reading, segment) {
  const { latitude, longitude, iriScore, hasPothole = false, potholeConfidence = 0, deviceId, sessionId, recordedAt } = reading;

  const obs = new Observation({
    roadSegmentId: segment._id,
    deviceId,
    sessionId,
    latitude,
    longitude,
    location: { type: 'Point', coordinates: [longitude, latitude] },
    iriScore,
    hasPothole,
    potholeConfidence,
    recordedAt: recordedAt ? new Date(recordedAt) : new Date(),
  });

  await obs.save();
  return obs;
}

function emitMapPointEvent(observation) {
  if (observation.markerType !== 'pothole') return; 

  const io = getIO();
  io.emit('map-point-event', {
    type:     observation.markerType,
    location: { lat: observation.latitude, lng: observation.longitude },
    iriScore: observation.iriScore,
    hasPothole: observation.hasPothole,
    potholeConfidence: observation.potholeConfidence,
    timestamp: observation.recordedAt,
  });
}

async function verifyAndEmitPothole(observation) {
  if (observation.markerType !== 'pothole') return;

  const timeWindow = new Date(Date.now() - CONSENSUS_TIME_WINDOW_MS);
  
  const recentPotholes = await Observation.find({
    markerType: 'pothole',
    recordedAt: { $gte: timeWindow },
    location: {
      $near: {
        $geometry: { type: 'Point', coordinates: [observation.longitude, observation.latitude] },
        $maxDistance: CONSENSUS_RADIUS_METERS,
      },
    },
  }).select('deviceId').lean();

  const uniqueUsers = new Set();
  recentPotholes.forEach(p => { if (p.deviceId) uniqueUsers.add(p.deviceId); });
  if (observation.deviceId) uniqueUsers.add(observation.deviceId);

  if (uniqueUsers.size >= CONSENSUS_MIN_USERS) {
    console.log(`[Consensus] Pothole Verified! Votes: ${uniqueUsers.size}/${CONSENSUS_MIN_USERS}. Emitting Pin.`);
    emitMapPointEvent(observation);
  }
}

// ── Router Endpoint Handlers ───────────────────────────────────────────────

/**
 * Maps to: POST /api/observations
 */
async function submitObservation(req, res) {
  try {
    const { latitude, longitude, iriScore, hasPothole = false, potholeConfidence = 0, roadSegmentId, deviceId, sessionId, recordedAt, heading } = req.body;

    if (latitude === undefined || longitude === undefined) {
      return res.status(400).json({ error: 'latitude and longitude are required' });
    }
    if (iriScore === undefined || typeof iriScore !== 'number' || iriScore < 0) {
      return res.status(400).json({ error: 'iriScore must be a non-negative number' });
    }

    if (!passesConfidenceGate(hasPothole, potholeConfidence)) {
      return res.status(202).json({ accepted: false, reason: 'potholeConfidence below threshold.' });
    }

    const segment = await resolveSegment(latitude, longitude, roadSegmentId, sessionId, heading);
    if (!segment) {
      return res.status(404).json({ error: 'No road segment found near coordinates' });
    }

    const reading = { latitude, longitude, iriScore, hasPothole, potholeConfidence, deviceId, sessionId, recordedAt };
    const obs     = await saveObservation(reading, segment);

    verifyAndEmitPothole(obs).catch((err) => {
      console.error('[Background Consensus Error]:', err.message);
    });

    const updatedSegment = await RoadSegment.findByIdAndUpdate(
      segment._id,
      { $addToSet: { 'iriStats.pendingDevices': deviceId } },
      { new: true }
    );

    const pendingCount = updatedSegment.iriStats?.pendingDevices?.length || 0;

    if (pendingCount >= MILESTONE_THRESHOLD) {
      aggregation.updateSegment(segment._id).catch((err) => {
        console.error('[aggregation] updateSegment failed:', err.message);
      });
    }

    return res.status(201).json({
      success: true,
      observationId: obs._id,
      markerType: obs.markerType,
      roadSegmentId: segment._id,
    });
  } catch (err) {
    console.error('[Controller Error] submitObservation:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

/**
 * Maps to: POST /api/observations/patch
 * Handles batch array telemetry profiles smoothly.
 */
async function submitPatch(req, res) {
  try {
    const { observations: readings } = req.body;
    // Fallback if client sends raw array inside body instead of wrapper object
    const batchList = Array.isArray(readings) ? readings : (Array.isArray(req.body) ? req.body : null);

    if (!batchList || batchList.length === 0) {
      return res.status(400).json({ error: 'Observations patch payload must be a non-empty array' });
    }

    const saved = [];
    const skipped = [];
    const segmentDevicesMap = {};

    for (const reading of batchList) {
      if (!passesConfidenceGate(reading.hasPothole, reading.potholeConfidence)) {
        skipped.push({ latitude: reading.latitude, longitude: reading.longitude, reason: 'low confidence' });
        continue;
      }

      const segment = await resolveSegment(reading.latitude, reading.longitude, reading.roadSegmentId, reading.sessionId, reading.heading);
      if (!segment) {
        skipped.push({ latitude: reading.latitude, longitude: reading.longitude, reason: 'no segment found' });
        continue;
      }

      const obs = await saveObservation(reading, segment);
      saved.push(obs._id);
      
      const segKey = String(segment._id);
      if (!segmentDevicesMap[segKey]) segmentDevicesMap[segKey] = new Set();
      if (reading.deviceId) segmentDevicesMap[segKey].add(reading.deviceId);

      verifyAndEmitPothole(obs).catch((err) => {
        console.error('[Batch Background Consensus Error]:', err.message);
      });
    }

    for (const segId of Object.keys(segmentDevicesMap)) {
      const uniqueDevicesArr = Array.from(segmentDevicesMap[segId]);
      if (uniqueDevicesArr.length > 0) {
        const updatedSegment = await RoadSegment.findByIdAndUpdate(
          segId,
          { $addToSet: { 'iriStats.pendingDevices': { $each: uniqueDevicesArr } } },
          { new: true }
        );

        const pendingCount = updatedSegment.iriStats?.pendingDevices?.length || 0;
        if (pendingCount >= MILESTONE_THRESHOLD) {
          aggregation.updateSegment(segId).catch((err) => {
            console.error('[aggregation] Batch updateSegment failed:', err.message);
          });
        }
      }
    }

    return res.status(201).json({ success: true, savedCount: saved.length, skippedCount: skipped.length, savedIds: saved });
  } catch (err) {
    console.error('[Controller Error] submitPatch batch:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

/**
 * Maps to: GET /api/observations/history
 */
async function getObservationHistory(req, res) {
  try {
    const deviceId = req.user?.deviceId || req.query.deviceId; 
    if (!deviceId) return res.status(400).json({ error: 'Device identifier missing' });

    const history = await Observation.find({ deviceId })
      .sort({ recordedAt: -1 })
      .limit(100)
      .lean();

    return res.status(200).json({ success: true, count: history.length, data: history });
  } catch (err) {
    return res.status(500).json({ error: 'Internal server error' });
  }
}

/**
 * Maps to: GET /api/observations/recent
 * Returns real-time validated alerts directly for UI components
 */
async function getRecentAlerts(req, res) {
  try {
    const alerts = await Observation.find({ markerType: 'pothole' })
      .sort({ recordedAt: -1 })
      .limit(100)
      .lean();

    return res.status(200).json({ success: true, count: alerts.length, data: alerts });
  } catch (err) {
    return res.status(500).json({ error: 'Internal server error' });
  }
}

module.exports = {
  submitObservation,
  submitPatch,
  getObservationHistory,
  getRecentAlerts,
  POTHOLE_CONFIDENCE_THRESHOLD,
};