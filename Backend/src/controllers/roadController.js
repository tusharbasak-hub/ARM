const RoadSegment = require('../models/RoadSegment');
const { getRegionId, getNeighbors, calculateDistance } = require('../utils/geohash');

/**
 * Get road segments in a region
 */
exports.getRoadSegmentsByRegion = async (req, res, next) => {
    try {
        const { regionId } = req.params;
        const { includeNeighbors = 'true' } = req.query;

        let regionIds = [regionId];

        // Include neighboring regions for smoother map experience
        if (includeNeighbors === 'true') {
            regionIds = getNeighbors(regionId);
        }

        const roadSegments = await RoadSegment.find({
            regionId: { $in: regionIds },
            aggregatedQualityScore: { $ne: null } // Only return segments with data
        })
            .select('-__v')
            .sort({ lastUpdated: -1 });

        res.json({
            success: true,
            data: {
                regionId,
                roadSegments,
                count: roadSegments.length
            }
        });
    } catch (error) {
        next(error);
    }
};

/**
 * Get map segments for complete road visualization
 * Returns ALL segments (green/yellow/red) in viewport regions
 */
exports.getMapSegments = async (req, res, next) => {
    try {
        const { regionIds } = req.validatedQuery;

        // Fetch all road segments with quality data in given regions
        const roadSegments = await RoadSegment.find({
            regionId: { $in: regionIds },
            observationCount: { $gt: 0 } // Only segments with observations
        })
            .select('roadSegmentId geometry centerPoint aggregatedQualityScore observationCount regionId lastUpdated')
            .lean()
            .limit(1000); // Cap to prevent huge payloads

        // Transform to map-friendly format with color coding
        const segments = roadSegments.map(seg => {
            const score = seg.aggregatedQualityScore ?? 1.0; // Default to good if no score

            // Color logic: 0-1=green, 1-2=yellow, 2-3=red
            let color = 'green';
            let quality = 'good';
            if (score >= 2) {
                color = 'red';
                quality = 'bad';
            } else if (score >= 1) {
                color = 'orange';
                quality = 'moderate';
            }

            return {
                roadSegmentId: seg.roadSegmentId,
                geometry: seg.geometry, // LineString with coordinates
                score: Number(score.toFixed(2)),
                color,
                quality,
                observationCount: seg.observationCount,
                lastUpdated: seg.lastUpdated
            };
        });

        res.json({
            success: true,
            data: {
                segments,
                count: segments.length
            }
        });
    } catch (error) {
        next(error);
    }
};

/**
 * Get nearby road segments
 */
exports.getNearbyRoadSegments = async (req, res, next) => {
    try {
        const { lat, lng, radius } = req.validatedQuery;

        // Get region ID and neighbors
        const regionId = getRegionId(lat, lng);
        const regionIds = getNeighbors(regionId);

        // Find road segments in nearby regions
        const roadSegments = await RoadSegment.find({
            regionId: { $in: regionIds },
            aggregatedQualityScore: { $ne: null },
            geometry: {
                $near: {
                    $geometry: {
                        type: 'Point',
                        coordinates: [lng, lat]
                    },
                    $maxDistance: radius
                }
            }
        })
            .select('-__v')
            .limit(100);

        // Calculate actual distances
        const segmentsWithDistance = roadSegments.map(segment => {
            const coords = segment.geometry.coordinates[0];
            const distance = calculateDistance(lat, lng, coords[1], coords[0]);

            return {
                ...segment.toObject(),
                distance
            };
        });

        // Sort by distance
        segmentsWithDistance.sort((a, b) => a.distance - b.distance);

        res.json({
            success: true,
            data: {
                location: { latitude: lat, longitude: lng },
                radius,
                roadSegments: segmentsWithDistance,
                count: segmentsWithDistance.length
            }
        });
    } catch (error) {
        next(error);
    }
};

/**
 * Get specific road segment details
 */
exports.getRoadSegmentDetails = async (req, res, next) => {
    try {
        const { segmentId } = req.params;

        const roadSegment = await RoadSegment.findOne({ roadSegmentId: segmentId });

        if (!roadSegment) {
            return res.status(404).json({
                success: false,
                message: 'Road segment not found'
            });
        }

        res.json({
            success: true,
            data: {
                roadSegment
            }
        });
    } catch (error) {
        next(error);
    }
};

/**
 * Get road quality statistics for a region
 */
exports.getRegionStatistics = async (req, res, next) => {
    try {
        const { regionId } = req.params;

        const stats = await RoadSegment.aggregate([
            {
                $match: {
                    regionId,
                    aggregatedQualityScore: { $ne: null }
                }
            },
            {
                $group: {
                    _id: null,
                    totalSegments: { $sum: 1 },
                    averageQuality: { $avg: '$aggregatedQualityScore' },
                    totalObservations: { $sum: '$observationCount' },
                    excellentRoads: {
                        $sum: { $cond: [{ $lte: ['$aggregatedQualityScore', 0.5] }, 1, 0] }
                    },
                    goodRoads: {
                        $sum: {
                            $cond: [{
                                $and: [
                                    { $gt: ['$aggregatedQualityScore', 0.5] },
                                    { $lte: ['$aggregatedQualityScore', 1.5] }
                                ]
                            }, 1, 0]
                        }
                    },
                    badRoads: {
                        $sum: {
                            $cond: [{
                                $and: [
                                    { $gt: ['$aggregatedQualityScore', 1.5] },
                                    { $lte: ['$aggregatedQualityScore', 2.5] }
                                ]
                            }, 1, 0]
                        }
                    },
                    worstRoads: {
                        $sum: { $cond: [{ $gt: ['$aggregatedQualityScore', 2.5] }, 1, 0] }
                    }
                }
            }
        ]);

        res.json({
            success: true,
            data: {
                regionId,
                statistics: stats[0] || {
                    totalSegments: 0,
                    averageQuality: null,
                    totalObservations: 0,
                    excellentRoads: 0,
                    goodRoads: 0,
                    badRoads: 0,
                    worstRoads: 0
                }
            }
        });
    } catch (error) {
        next(error);
    }
};
