// services/routeScoring.js
const axios = require('axios');
const RoadSegment = require('../models/RoadSegment');
const { getRedisClient } = require('../config/redis');
const polyline = require('@mapbox/polyline');
const { getRegionId, getNeighbors } = require('../utils/geohash');

class RouteScoringService {
  constructor() {
    this.mapboxToken = process.env.MAPBOX_ACCESS_TOKEN;
    this.mapboxBaseUrl = 'https://api.mapbox.com/directions/v5/mapbox/driving';

    // Weights (tuneable)
    this.QUALITY_WEIGHT = Number(process.env.ROUTE_QUALITY_WEIGHT ?? 0.7);
    this.DISTANCE_WEIGHT = Number(process.env.ROUTE_DISTANCE_WEIGHT ?? 0.3);

    // Segment settings
    this.TARGET_SAMPLE_STEP_M = Number(process.env.ROUTE_SAMPLE_STEP_M ?? 75); // sample approx every 75m
    this.MATCH_TOLERANCE_M = Number(process.env.ROUTE_MATCH_TOLERANCE_M ?? 120); // nearest centerPoint within 120m

    // Missing data behavior
    // 0 best .. 3 worst
    this.DEFAULT_QUALITY_SCORE = Number(process.env.ROUTE_DEFAULT_QUALITY_SCORE ?? 1.0);

    // Distance guardrail: prevent silly long route
    // If a route is > (1 + 0.80) = 180% of shortest, penalize heavily
    this.MAX_OVERAGE_RATIO = Number(process.env.ROUTE_MAX_OVERAGE_RATIO ?? 0.8);
    this.OVERAGE_PENALTY = Number(process.env.ROUTE_OVERAGE_PENALTY ?? 0.35); // add to finalScore

    // Cache
    this.CACHE_TTL = Number(process.env.ROUTE_CACHE_TTL ?? 900); // 15 min recommended for "current time"
    this.FRESHNESS_SAMPLE_K = Number(process.env.ROUTE_FRESHNESS_SAMPLE_K ?? 8); // fingerprint top K segments
    this.SIGNIFICANT_SCORE_CHANGE = Number(process.env.ROUTE_SIGNIFICANT_SCORE_CHANGE ?? 0.25);
  }

  /* =========================
     Public APIs
  ========================= */

  async getAndScoreRoutes(source, destination, maxRoutes = 3) {
    if (!this.mapboxToken) throw new Error('Mapbox token missing');

    const cacheKey = this.generateCacheKey(source, destination, maxRoutes);

    const cached = await this.getCached(cacheKey);
    if (cached) return cached;

    // 1) Fetch candidate routes from Mapbox
    const mapboxRoutes = await this.fetchMapboxRoutes(source, destination, maxRoutes);
    if (!mapboxRoutes?.length) throw new Error('No routes found between source and destination');

    // 2) Decode & pre-sample points for each route
    const prepared = mapboxRoutes.map(r => this.prepareRouteGeometry(r));

    // 3) Prefetch candidate road segments ONCE (for all routes)
    const regionIds = this.collectAllRegions(prepared);
    const candidateSegments = await this.fetchCandidateSegments(regionIds);

    // 4) Score each route using in-memory matching
    const scored = prepared.map(p => this.scorePreparedRoute(p, candidateSegments));

    // 5) Normalize distance across candidates & compute finalScore
    this.applyFinalScoring(scored);

    // 6) Sort and create response
    scored.sort((a, b) => a.finalScore - b.finalScore);
    const result = this.prepareResult(scored, source, destination);

    // 7) Cache with fingerprint for freshness checks
    await this.setCached(cacheKey, result);

    return result;
  }

  // Optional endpoint use
  async scoreRoute(mapboxRoute) {
    const prepared = this.prepareRouteGeometry(mapboxRoute);
    const regionIds = this.collectAllRegions([prepared]);
    const candidateSegments = await this.fetchCandidateSegments(regionIds);
    const scored = this.scorePreparedRoute(prepared, candidateSegments);

    // For single route preview, finalScore is less meaningful (no competing routes),
    // still compute a "self-normalized" distance (0).
    scored.finalScore = (scored.normalizedQuality * this.QUALITY_WEIGHT);
    return scored;
  }

  getQualityRating(score) {
    if (score <= 0.5) return 'Excellent';
    if (score <= 1.5) return 'Good';
    if (score <= 2.5) return 'Fair';
    return 'Poor';
  }

  /* =========================
     Mapbox
  ========================= */

  async fetchMapboxRoutes(source, destination, maxRoutes) {
    try {
      const coordinates = `${source.longitude},${source.latitude};${destination.longitude},${destination.latitude}`;
      const url = `${this.mapboxBaseUrl}/${coordinates}`;

      const params = {
        access_token: this.mapboxToken,
        alternatives: true,
        geometries: 'polyline6',
        overview: 'full',
        steps: false
      };

      const res = await axios.get(url, { params, timeout: 10000 });

      if (res.data?.code !== 'Ok' || !Array.isArray(res.data.routes)) {
        throw new Error('Mapbox API returned no routes');
      }

      return res.data.routes.slice(0, maxRoutes);
    } catch (err) {
      if (err.response) {
        throw new Error(`Mapbox API error: ${err.response.data?.message || err.message}`);
      }
      throw new Error(`Failed to fetch routes: ${err.message}`);
    }
  }

  /* =========================
     Geometry & Sampling
  ========================= */

  prepareRouteGeometry(mapboxRoute) {
    const coords = polyline.decode(mapboxRoute.geometry, 6); // returns [lat,lng]
    if (!coords?.length) {
      return {
        geometry: mapboxRoute.geometry,
        distance: mapboxRoute.distance || 0,
        duration: mapboxRoute.duration || 0,
        sampledPoints: [],
        sampledEdges: [] // [{from,to,edgeLenM}]
      };
    }

    // Sample approximately every TARGET_SAMPLE_STEP_M along polyline
    const { points, edges } = this.sampleAlongPolyline(coords, this.TARGET_SAMPLE_STEP_M);

    return {
      geometry: mapboxRoute.geometry,
      distance: mapboxRoute.distance || (this.polylineLengthMeters(coords)),
      duration: mapboxRoute.duration || 0,
      sampledPoints: points, // [{lat,lng, regionId}]
      sampledEdges: edges     // [{aIdx,bIdx,lenM}]
    };
  }

  sampleAlongPolyline(latlngs, stepM) {
    const points = [];
    const edges = [];

    // Always include start
    let last = { lat: latlngs[0][0], lng: latlngs[0][1] };
    points.push(this.enrichPoint(last));

    let carry = 0;

    for (let i = 1; i < latlngs.length; i++) {
      const curr = { lat: latlngs[i][0], lng: latlngs[i][1] };
      let segLen = haversineMeters(last.lat, last.lng, curr.lat, curr.lng);

      // If too small, move on accumulating
      if (segLen <= 0.001) {
        last = curr;
        continue;
      }

      while (carry + segLen >= stepM) {
        // Interpolate new point on this segment
        const remain = stepM - carry;
        const t = remain / segLen;

        const interp = {
          lat: last.lat + (curr.lat - last.lat) * t,
          lng: last.lng + (curr.lng - last.lng) * t
        };

        const prevIdx = points.length - 1;
        points.push(this.enrichPoint(interp));
        const newIdx = points.length - 1;

        edges.push({ aIdx: prevIdx, bIdx: newIdx, lenM: stepM });

        // Now continue from interp point to curr
        last = interp;
        segLen = haversineMeters(last.lat, last.lng, curr.lat, curr.lng);
        carry = 0;
      }

      carry += segLen;
      last = curr;
    }

    // Include end point if not very close to last sampled
    const end = { lat: latlngs[latlngs.length - 1][0], lng: latlngs[latlngs.length - 1][1] };
    const lastP = points[points.length - 1];
    const tailLen = haversineMeters(lastP.lat, lastP.lng, end.lat, end.lng);

    if (tailLen > stepM * 0.35) {
      const prevIdx = points.length - 1;
      points.push(this.enrichPoint(end));
      const newIdx = points.length - 1;
      edges.push({ aIdx: prevIdx, bIdx: newIdx, lenM: tailLen });
    }

    return { points, edges };
  }

  enrichPoint(p) {
    const regionId = getRegionId(p.lat, p.lng);
    return { ...p, regionId };
  }

  polylineLengthMeters(latlngs) {
    let total = 0;
    for (let i = 1; i < latlngs.length; i++) {
      total += haversineMeters(latlngs[i - 1][0], latlngs[i - 1][1], latlngs[i][0], latlngs[i][1]);
    }
    return total;
  }

  /* =========================
     Candidate Segments Prefetch
  ========================= */

  collectAllRegions(preparedRoutes) {
    const regionSet = new Set();
    for (const r of preparedRoutes) {
      for (const p of r.sampledPoints) {
        regionSet.add(p.regionId);
        // include neighbors for better match
        const neighbors = getNeighbors(p.regionId);
        for (const n of neighbors) regionSet.add(n);
      }
    }
    return Array.from(regionSet);
  }

  async fetchCandidateSegments(regionIds) {
    // Prefetch segments for all relevant regions in ONE query
    // Keep it light: we only need centerPoint + score.
    const segments = await RoadSegment.find({
      regionId: { $in: regionIds }
    }).select('roadSegmentId centerPoint aggregatedQualityScore observationCount lastUpdated');

    // Convert to array of usable objects
    return segments.map(s => ({
      roadSegmentId: s.roadSegmentId,
      lat: s.centerPoint?.coordinates?.[1],
      lng: s.centerPoint?.coordinates?.[0],
      score: (s.aggregatedQualityScore ?? null),
      observationCount: s.observationCount ?? 0,
      lastUpdated: s.lastUpdated ?? null
    })).filter(x => typeof x.lat === 'number' && typeof x.lng === 'number');
  }

  /* =========================
     Scoring (In-memory nearest match)
  ========================= */

  scorePreparedRoute(preparedRoute, candidateSegments) {
    const { sampledPoints, sampledEdges } = preparedRoute;

    // Map each sampled point to nearest candidate segment
    // We'll then accumulate weighted quality per edge length.
    const matched = sampledPoints.map(p => {
      const nearest = findNearestInMemory(p.lat, p.lng, candidateSegments, this.MATCH_TOLERANCE_M);
      if (!nearest) {
        return {
          hasData: false,
          roadSegmentId: null,
          qualityScore: this.DEFAULT_QUALITY_SCORE,
          observationCount: 0,
          matchDistanceM: null
        };
      }

      const score = (typeof nearest.score === 'number') ? nearest.score : this.DEFAULT_QUALITY_SCORE;
      const hasData = (typeof nearest.score === 'number');

      return {
        hasData,
        roadSegmentId: nearest.roadSegmentId,
        qualityScore: score,
        observationCount: nearest.observationCount,
        matchDistanceM: nearest.distanceM
      };
    });

    // Now compute distance-weighted route quality using edges
    let weightedSum = 0;
    let lengthSum = 0;

    // Also compute segment-level stats
    const usedSegmentIds = new Set();
    const segmentScores = {}; // FIX D: Track each segment's score for cache fingerprinting
    let segmentsWithData = 0;

    for (const e of sampledEdges) {
      const segInfo = matched[e.bIdx]; // quality at end point of that edge
      const q = segInfo?.qualityScore ?? this.DEFAULT_QUALITY_SCORE;

      weightedSum += q * e.lenM;
      lengthSum += e.lenM;

      if (segInfo?.roadSegmentId) {
        usedSegmentIds.add(segInfo.roadSegmentId);
        segmentScores[segInfo.roadSegmentId] = segInfo.qualityScore; // Store score snapshot
      }
    }

    // count data points with real data (approx)
    for (const m of matched) if (m.hasData) segmentsWithData++;

    const roadQualityScore = lengthSum > 0 ? (weightedSum / lengthSum) : this.DEFAULT_QUALITY_SCORE;

    // normalize quality 0..1
    const normalizedQuality = clamp01(roadQualityScore / 3);

    const distanceKm = (preparedRoute.distance || 0) / 1000;

    return {
      geometry: preparedRoute.geometry,
      distance: preparedRoute.distance,
      duration: preparedRoute.duration,
      distanceKm,
      roadQualityScore,
      normalizedQuality,
      // filled later in applyFinalScoring:
      normalizedDistance: 0,
      finalScore: 0,
      segmentCount: usedSegmentIds.size,
      segmentsWithData,
      dataCompleteness: sampledPoints.length > 0 ? (segmentsWithData / sampledPoints.length) : 0,
      // Store some details for caching / debug
      segmentDetails: Array.from(usedSegmentIds).slice(0, 200), // cap to avoid huge payload
      segmentScores // FIX D: Include scores for cache fingerprinting
    };
  }

  applyFinalScoring(scoredRoutes) {
    // Normalize distance relative to current candidates
    const distances = scoredRoutes.map(r => r.distanceKm);
    const minD = Math.min(...distances);
    const maxD = Math.max(...distances);
    const denom = (maxD - minD) || 1e-6;

    for (const r of scoredRoutes) {
      r.normalizedDistance = clamp01((r.distanceKm - minD) / denom);

      // Base weighted score
      let score = (r.normalizedQuality * this.QUALITY_WEIGHT) +
        (r.normalizedDistance * this.DISTANCE_WEIGHT);

      // Guardrail: penalize silly long routes even if quality is good
      if (r.distanceKm > minD * (1 + this.MAX_OVERAGE_RATIO)) {
        score += this.OVERAGE_PENALTY;
      }

      r.finalScore = score;
    }
  }

  /* =========================
     Result + Recommendation
  ========================= */

  prepareResult(scoredRoutes, source, destination) {
    const routes = scoredRoutes.map((r, idx) => ({
      rank: idx + 1,
      distance: r.distance,
      distanceKm: Number(r.distanceKm.toFixed(2)),
      duration: r.duration,
      durationMinutes: Math.ceil((r.duration || 0) / 60),
      roadQualityScore: Number(r.roadQualityScore.toFixed(2)),
      qualityRating: this.getQualityRating(r.roadQualityScore),
      finalScore: Number(r.finalScore.toFixed(4)),
      geometry: r.geometry,
      segmentCount: r.segmentCount,
      segmentsWithData: r.segmentsWithData,
      dataCompleteness: Number((r.dataCompleteness * 100).toFixed(1))
    }));

    const bestRoute = routes[0];

    const reason = this.generateRecommendation(routes);

    // FIX D: Fingerprint stores per-segment score snapshots for accurate freshness checking
    const topSegments = (scoredRoutes[0]?.segmentDetails || []).slice(0, this.FRESHNESS_SAMPLE_K);
    const fingerprint = {
      segments: topSegments.map(segId => ({
        roadSegmentId: segId,
        scoreSnapshot: scoredRoutes[0]?.segmentScores?.[segId] ?? this.DEFAULT_QUALITY_SCORE
      })),
      bestRoadQualityScore: scoredRoutes[0]?.roadQualityScore ?? this.DEFAULT_QUALITY_SCORE // keep for backward compat
    };

    return {
      source,
      destination,
      routes,
      bestRoute: { ...bestRoute, reason },
      fingerprint,
      timestamp: new Date().toISOString()
    };
  }

  generateRecommendation(routes) {
    const best = routes[0];
    if (routes.length === 1) return `Only route available. Road quality: ${best.qualityRating}.`;

    const shortest = Math.min(...routes.map(r => r.distanceKm));
    const bestQuality = Math.min(...routes.map(r => r.roadQualityScore));

    const reasons = [];
    if (best.distanceKm === shortest) reasons.push('shortest distance');
    if (best.roadQualityScore === bestQuality) reasons.push(`best road quality (${best.qualityRating})`);

    if (reasons.length === 2) return `Optimal balance of ${reasons[0]} and ${reasons[1]}.`;
    if (reasons.length === 1) return `Best overall choice due to ${reasons[0]}.`;
    return `Best weighted combination of distance and road quality.`;
  }

  /* =========================
     Cache (Redis) + Freshness
  ========================= */

  generateCacheKey(source, destination, maxRoutes) {
    const src = `${source.latitude.toFixed(4)},${source.longitude.toFixed(4)}`;
    const dst = `${destination.latitude.toFixed(4)},${destination.longitude.toFixed(4)}`;
    return `route:${src}:${dst}:mr${maxRoutes}`;
  }

  async getCached(cacheKey) {
    const redis = getRedisClient();
    if (!redis) return null;

    try {
      const raw = await redis.get(cacheKey);
      if (!raw) return null;

      const parsed = JSON.parse(raw);

      const ok = await this.isCacheFresh(parsed);
      if (!ok) {
        await redis.del(cacheKey);
        return null;
      }

      return parsed;
    } catch (err) {
      return null;
    }
  }

  async setCached(cacheKey, value) {
    const redis = getRedisClient();
    if (!redis) return;

    try {
      await redis.setEx(cacheKey, this.CACHE_TTL, JSON.stringify(value));
    } catch (err) {
      // ignore cache errors
    }
  }

  async isCacheFresh(cachedResult) {
    try {
      const fp = cachedResult?.fingerprint;
      // FIX D: Support both old (segmentIds) and new (segments array) fingerprint formats
      const segmentData = fp?.segments || [];
      const legacyIds = fp?.segmentIds || [];

      if (!segmentData.length && !legacyIds.length) return true;

      // Build query for all segment IDs
      const allIds = segmentData.length
        ? segmentData.map(s => s.roadSegmentId)
        : legacyIds;

      const segments = await RoadSegment.find({
        roadSegmentId: { $in: allIds }
      }).select('roadSegmentId aggregatedQualityScore');

      const currentScores = new Map(segments.map(s => [s.roadSegmentId, s.aggregatedQualityScore]));

      // Compare each segment's current score to its own snapshot (not to bestRoadQualityScore)
      if (segmentData.length) {
        for (const { roadSegmentId, scoreSnapshot } of segmentData) {
          const current = currentScores.get(roadSegmentId);
          if (typeof current !== 'number') continue;

          const diff = Math.abs(current - scoreSnapshot);
          if (diff > this.SIGNIFICANT_SCORE_CHANGE) return false;
        }
      } else {
        // Legacy format fallback: compare to bestRoadQualityScore
        for (const id of legacyIds) {
          const current = currentScores.get(id);
          if (typeof current !== 'number') continue;
          const diff = Math.abs(current - fp.bestRoadQualityScore);
          if (diff > this.SIGNIFICANT_SCORE_CHANGE) return false;
        }
      }

      return true;
    } catch (err) {
      // On error, keep cache (avoid server load spikes)
      return true;
    }
  }
}

/* =========================
   Helpers
========================= */

function haversineMeters(lat1, lon1, lat2, lon2) {
  const R = 6371000;
  const toRad = d => (d * Math.PI) / 180;

  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function findNearestInMemory(lat, lng, segments, maxDistM) {
  let best = null;
  let bestD = Infinity;

  for (const s of segments) {
    const d = haversineMeters(lat, lng, s.lat, s.lng);
    if (d < bestD) {
      bestD = d;
      best = s;
    }
  }

  if (!best || bestD > maxDistM) return null;
  return { ...best, distanceM: bestD };
}

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

module.exports = new RouteScoringService();
