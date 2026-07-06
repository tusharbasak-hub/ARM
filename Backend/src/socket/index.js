// ==========================================
// FILE: Backend/src/socket/index.js
// ==========================================

'use strict';

const { Server }     = require('socket.io');
const RoadSegment    = require('../models/RoadSegment');
const Observation    = require('../models/Observation');

let _io = null;

function init(server, opts = {}) {
  if (_io) {
    console.warn('[socket] init() called more than once — reusing existing instance');
    return _io;
  }

  _io = new Server(server, {
    cors: {
      origin:  process.env.CORS_ORIGIN || '*',
      methods: ['GET', 'POST'],
    },
    transports: ['websocket', 'polling'],
    ...opts,
  });

  _io.on('connection', async (socket) => {
    console.info(`[socket] Client connected: ${socket.id}`);

    try {
      await _sendInitialState(socket);
    } catch (err) {
      console.error('[socket] Failed to send initial state:', err.message);
    }

    socket.on('disconnect', (reason) => {
      console.info(`[socket] Client disconnected: ${socket.id} (${reason})`);
    });

    socket.on('request-segment', async (data) => {
      try {
        const { segmentId } = data || {};
        if (!segmentId) return;

        const seg = await RoadSegment.findById(segmentId).lean();
        if (seg) {
          socket.emit('segment-polyline-update', {
            roadSegmentId: String(seg._id),
            iriCategory:   seg.iriCategory,
            averageIri:    seg.iriStats?.average ?? 0,
            sampleCount:   seg.iriStats?.sampleCount ?? 0,
            polyline:      seg.geometry, // Aligned to geometry structure
            name:          seg.name,
            updatedAt:     seg.lastObservationAt,
          });
        }
      } catch (err) {
        console.error('[socket] request-segment error:', err.message);
      }
    });
  });

  console.info('[socket] Socket.IO initialised');
  return _io;
}

function getIO() {
  if (!_io) {
    throw new Error('[socket] getIO() called before init(). Call init(server) first.');
  }
  return _io;
}

async function _sendInitialState(socket) {
  // Directly selects standardized geometry structure
  const segments = await RoadSegment.find({})
    .select('_id name geometry iriCategory iriStats lastObservationAt')
    .lean();

  socket.emit('initial-segments', {
    segments: segments.map((seg) => ({
      roadSegmentId: String(seg._id),
      name:          seg.name,
      iriCategory:   seg.iriCategory,
      averageIri:    seg.iriStats?.average ?? 0,
      sampleCount:   seg.iriStats?.sampleCount ?? 0,
      polyline:      seg.geometry, 
      updatedAt:     seg.lastObservationAt,
    })),
  });

  // Send all recent observations (potholes and IRI predictions) for dual-layer map plotting
  const recentPoints = await Observation.find({})
    .sort({ recordedAt: -1 })
    .limit(1000)
    .select('latitude longitude iriScore hasPothole potholeConfidence markerType recordedAt')
    .lean();

  socket.emit('initial-map-points', {
    points: recentPoints.map((obs) => ({
      type:              obs.markerType,
      location:          { lat: obs.latitude, lng: obs.longitude },
      iriScore:          obs.iriScore,
      hasPothole:        obs.hasPothole,
      potholeConfidence: obs.potholeConfidence,
      timestamp:         obs.recordedAt,
    })),
  });
}

module.exports = {
  init,
  getIO,
};