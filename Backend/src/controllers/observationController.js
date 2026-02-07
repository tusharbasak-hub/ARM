const Observation = require('../models/Observation');
const RoadSegment = require('../models/RoadSegment');
const User = require('../models/User');
const mapMatchingService = require('../services/mapMatching');
const aggregationService = require('../services/aggregation');
const { getRegionId } = require('../utils/geohash');

/**
 * Submit road quality observation
 */
exports.submitObservation = async (req, res, next) => {
    try {

        console.log(" Observation API HIT");
        console.log(" req.body:", req.body);
        console.log(" req.userId:", req.userId);

        const { latitude, longitude, roadQuality, speed, timestamp, deviceMetadata } = req.validatedData;

        console.log(" validatedData:", req.validatedData);

        const userId = req.userId;

        // Get region ID from coordinates
        const regionId = getRegionId(latitude, longitude);

        // Perform map matching
        const matchResult = await mapMatchingService.matchPoint(latitude, longitude);
        console.log(" Map Matching Result:", matchResult);

        if (!matchResult) {
            return res.status(400).json({
                success: false,
                message: 'Unable to match location to road network'
            });
        }

        // Create observation
        const observation = await Observation.create({
            userId,
            location: {
                type: "Point",
                coordinates: [longitude, latitude]
            },
            roadQuality,
            speed, // Store speed for aggregation speed-based weighting
            timestamp: new Date(timestamp),
            roadSegmentId: matchResult.roadSegmentId,
            regionId,
            matchingConfidence: matchResult.confidence // FIX A: Store confidence for aggregation filtering
        });


        // Update or create road segment
        let roadSegment = await RoadSegment.findOneAndUpdate(
            { roadSegmentId: matchResult.roadSegmentId },
            {
                $setOnInsert: {
                    roadSegmentId: matchResult.roadSegmentId,
                    geometry: {
                        type: "LineString",
                        coordinates: [
                            [matchResult.matchedLongitude, matchResult.matchedLatitude],
                            [longitude, latitude]
                        ]
                    },
                    centerPoint: {
                        type: "Point",
                        coordinates: [longitude, latitude]
                    },
                    regionId,
                    roadName: matchResult.roadName
                },
                $inc: { observationCount: 1 },
                $set: { lastUpdated: new Date() }
            },
            { upsert: true, new: true }
        );

        console.log(" RoadSegment saved:", roadSegment.roadSegmentId);
        // Trigger aggregation (async, don't wait)
        setImmediate(async () => {
            try {
                console.log(" Aggregation started for:", matchResult.roadSegmentId);
                const oldScore = roadSegment.aggregatedQualityScore;
                const aggregationResult = await aggregationService.aggregateRoadSegment(matchResult.roadSegmentId);
                console.log(" Aggregation Result:", aggregationResult);

                const io = req.app.get('io');
                if (io) {
                    // Broadcast individual observation alert (exact location for map markers)
                    // Only broadcast bad quality (2,3) to avoid noise
                    if (roadQuality >= 2) {
                        io.to(regionId).emit('observation-alert', {
                            observationId: observation._id,
                            location: { lat: latitude, lng: longitude },
                            roadQuality,
                            severity: roadQuality, // 2=bad, 3=worst
                            timestamp: observation.timestamp,
                            regionId
                        });
                    }

                    // Broadcast aggregated segment update if significant change
                    if (aggregationResult && aggregationService.shouldBroadcastUpdate(oldScore, aggregationResult.aggregatedQualityScore, aggregationResult.confidenceScore)) {
                        io.to(regionId).emit('road-quality-update', {
                            roadSegmentId: matchResult.roadSegmentId,
                            aggregatedQualityScore: aggregationResult.aggregatedQualityScore,
                            confidenceScore: aggregationResult.confidenceScore,
                            regionId,
                            lastUpdated: new Date()
                        });
                    }
                }
            } catch (err) {
                console.error('Aggregation error:', err);
            }
        });

        // Update user statistics
        await User.findByIdAndUpdate(userId, {
            $inc: { totalObservations: 1 },
            lastActive: new Date()
        });

        res.status(201).json({
            success: true,
            message: 'Observation submitted successfully',
            data: {
                observationId: observation._id,
                roadSegmentId: matchResult.roadSegmentId,
                matchingConfidence: matchResult.confidence,
                regionId
            }
        });
    } catch (error) {
        next(error);
    }
};

/**
 * Get user's observation history
 */
exports.getObservationHistory = async (req, res, next) => {
    try {
        const userId = req.userId;
        const { limit = 50, offset = 0 } = req.query;

        const observations = await Observation.find({ userId })
            .sort({ timestamp: -1 })
            .limit(parseInt(limit))
            .skip(parseInt(offset))
            .select('-__v');

        const total = await Observation.countDocuments({ userId });

        res.json({
            success: true,
            data: {
                observations,
                pagination: {
                    total,
                    limit: parseInt(limit),
                    offset: parseInt(offset)
                }
            }
        });
    } catch (error) {
        next(error);
    }
};

/**
 * Get recent bad road observations for map markers
 * Allows browsing users to see pothole locations even when not driving
 */
exports.getRecentAlerts = async (req, res, next) => {
    try {
        const { regionIds, minSeverity = 2, hoursBack = 72 } = req.validatedQuery;

        const cutoffTime = new Date(Date.now() - hoursBack * 60 * 60 * 1000);

        // Query recent bad observations (roadQuality >= minSeverity)
        const observations = await Observation.find({
            regionId: { $in: regionIds },
            roadQuality: { $gte: minSeverity },
            timestamp: { $gte: cutoffTime }
        })
            .select('location roadQuality timestamp regionId')
            .sort({ timestamp: -1 })
            .limit(500) // Cap to avoid huge payloads
            .lean();

        // Transform to client-friendly format
        const alerts = observations.map(obs => ({
            id: obs._id,
            location: {
                lat: obs.location.coordinates[1],
                lng: obs.location.coordinates[0]
            },
            severity: obs.roadQuality,
            timestamp: obs.timestamp,
            regionId: obs.regionId
        }));

        res.json({
            success: true,
            data: {
                alerts,
                count: alerts.length,
                queryParams: { regionIds, minSeverity, hoursBack }
            }
        });
    } catch (error) {
        next(error);
    }
};

/**
 * Submit road quality patch observation
 * Handles continuous bad road stretches without flooding the server
 */
exports.submitPatch = async (req, res, next) => {
    try {
        const {
            startLatitude,
            startLongitude,
            endLatitude,
            endLongitude,
            severity,
            patchLengthM,
            startTimestamp,
            endTimestamp,
            deviceMetadata
        } = req.validatedData;

        const userId = req.userId;

        // Calculate duration in seconds
        const durationSeconds = (new Date(endTimestamp) - new Date(startTimestamp)) / 1000;

        // RULE: Ignore patches that are both short (<5m) AND brief (<2s) - likely noise
        if (patchLengthM < 5 && durationSeconds < 2) {
            return res.status(400).json({
                success: false,
                message: 'Patch too short and brief - likely noise (minimum 5m or 2s required)'
            });
        }

        // Get region ID from START coordinates
        const regionId = getRegionId(startLatitude, startLongitude);

        // Perform map matching on START point only (fast, MVP approach)
        const matchResult = await mapMatchingService.matchPoint(startLatitude, startLongitude);

        if (!matchResult) {
            return res.status(400).json({
                success: false,
                message: 'Unable to match location to road network'
            });
        }

        // Determine if patch should affect segment scoring
        // Render on map if >= 5m OR >= 2s, but only affect scoring if >= 5m AND >= 2s
        const affectsScore = patchLengthM >= 5 && durationSeconds >= 2;

        // Update or create road segment with patch length counters
        const updateOps = {
            $setOnInsert: {
                roadSegmentId: matchResult.roadSegmentId,
                geometry: {
                    type: "LineString",
                    coordinates: [
                        [startLongitude, startLatitude],
                        [endLongitude, endLatitude]
                    ]
                },
                centerPoint: {
                    type: "Point",
                    coordinates: [startLongitude, startLatitude]
                },
                regionId,
                roadName: matchResult.roadName
            },
            $set: { lastUpdated: new Date() }
        };

        // Only increment length counters if patch affects score
        if (affectsScore) {
            // FIX C: Safe severity handling (validated but defensive)
            if (severity === 1) {
                updateOps.$inc = { len1: patchLengthM, observationCount: 1 };
            } else if (severity === 2) {
                updateOps.$inc = { len2: patchLengthM, observationCount: 1 };
            } else if (severity === 3) {
                updateOps.$inc = { len3: patchLengthM, observationCount: 1 };
            } else {
                // Unexpected severity value - increment observationCount only
                updateOps.$inc = { observationCount: 1 };
            }
        } else {
            // Still increment observation count for tracking, but not length
            updateOps.$inc = { observationCount: 1 };
        }

        const roadSegment = await RoadSegment.findOneAndUpdate(
            { roadSegmentId: matchResult.roadSegmentId },
            updateOps,
            { upsert: true, new: true }
        );

        // Trigger aggregation (async, don't wait)
        setImmediate(async () => {
            try {
                const oldScore = roadSegment.aggregatedQualityScore;
                const aggregationResult = await aggregationService.aggregateRoadSegment(matchResult.roadSegmentId);

                const io = req.app.get('io');
                if (io) {
                    // Broadcast patch alert (exact bad road stretch for map polyline)
                    if (affectsScore && severity >= 2) {
                        io.to(regionId).emit('patch-alert', {
                            startLocation: { lat: startLatitude, lng: startLongitude },
                            endLocation: { lat: endLatitude, lng: endLongitude },
                            severity,
                            patchLengthM,
                            timestamp: new Date(startTimestamp),
                            regionId
                        });
                    }

                    // Broadcast aggregated segment update if significant change
                    if (aggregationResult && aggregationService.shouldBroadcastUpdate(oldScore, aggregationResult.aggregatedQualityScore, aggregationResult.confidenceScore)) {
                        io.to(regionId).emit('road-quality-update', {
                            roadSegmentId: matchResult.roadSegmentId,
                            aggregatedQualityScore: aggregationResult.aggregatedQualityScore,
                            confidenceScore: aggregationResult.confidenceScore,
                            regionId,
                            lastUpdated: new Date()
                        });
                    }
                }
            } catch (err) {
                console.error('Patch aggregation error:', err);
            }
        });

        // Update user statistics
        await User.findByIdAndUpdate(userId, {
            $inc: { totalObservations: 1 },
            lastActive: new Date()
        });

        res.status(201).json({
            success: true,
            message: 'Patch observation submitted successfully',
            data: {
                roadSegmentId: matchResult.roadSegmentId,
                matchingConfidence: matchResult.confidence,
                regionId,
                patchLengthM,
                severity,
                affectsScore
            }
        });
    } catch (error) {
        next(error);
    }
};
